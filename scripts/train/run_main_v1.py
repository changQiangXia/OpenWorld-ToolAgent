#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_BOOTSTRAP))

from src.agent.main_v1_data import (
    batch_encode_inputs,
    batch_targets,
    build_retriever,
    build_tool_vocab,
    load_rows,
    sample_batch,
    set_global_seed,
)
from src.agent.main_v1_eval import evaluate_rows
from src.agent.main_v1_model import MainV1Model
from src.agent.runtime_utils import (
    compute_config_hash,
    ensure_dirs,
    ensure_within_root,
    load_yaml,
    make_exp_id,
    setup_logger,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train main_v1 open-world multimodal model on small samples.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, default=Path("configs/train/main_v1.yaml"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _resolve(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else (root / path)


def _pick_device(request: str) -> torch.device:
    req = str(request or "auto").lower()
    if req == "cpu":
        return torch.device("cpu")
    if req == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Config requested cuda but CUDA is not available")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()

    cfg_path = _resolve(args.train_config, project_root)
    cfg = load_yaml(cfg_path)

    cfg_hash = compute_config_hash({"train": cfg})
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))

    task_name = str(cfg.get("task_name", "main_v1"))
    model_name = str(cfg.get("model_name", "main_v1_mm_openworld"))
    exp_id = make_exp_id(task=task_name, model=model_name, seed=seed)

    output_dirs = cfg.get("output_dirs", {})
    logs_dir = _resolve(Path(output_dirs.get("logs", "outputs/logs")), project_root)
    reports_dir = _resolve(Path(output_dirs.get("reports", "outputs/reports")), project_root)
    preds_dir = _resolve(Path(output_dirs.get("predictions", "outputs/predictions")), project_root)
    ckpt_dir = _resolve(Path(output_dirs.get("checkpoints", "outputs/checkpoints")), project_root)

    ensure_dirs([logs_dir, reports_dir, preds_dir, ckpt_dir], project_root)
    for p in [logs_dir, reports_dir, preds_dir, ckpt_dir, cfg_path]:
        ensure_within_root(p, project_root)

    log_file = logs_dir / f"{exp_id}.log"
    report_file = reports_dir / f"{exp_id}.json"
    pred_file = preds_dir / f"{exp_id}_dev.jsonl"
    ckpt_file = ckpt_dir / f"{exp_id}.pt"

    logger = setup_logger(log_file, exp_id=exp_id, cfg_hash=cfg_hash, level=str(cfg.get("log_level", "INFO")))
    logger.info("start main_v1 training")
    logger.info("project_root=%s", project_root)
    logger.info("config=%s", cfg_path)
    logger.info("seed=%d", seed)

    if args.dry_run:
        logger.info("dry-run mode enabled; no train/eval executed")
        print(
            json.dumps(
                {
                    "exp_id": exp_id,
                    "config_hash": cfg_hash,
                    "dry_run": True,
                    "log_file": str(log_file),
                    "prediction_file": str(pred_file),
                    "report_file": str(report_file),
                    "checkpoint_file": str(ckpt_file),
                },
                ensure_ascii=True,
            )
        )
        return

    set_global_seed(seed)
    rng = random.Random(seed + 101)

    data_cfg = cfg.get("data", {})
    train_path = _resolve(Path(data_cfg.get("train_file", "data/splits/train.jsonl")), project_root)
    dev_path = _resolve(Path(data_cfg.get("dev_file", "data/splits/dev.jsonl")), project_root)
    for p in [train_path, dev_path]:
        ensure_within_root(p, project_root)

    training_cfg = cfg.get("training", {})
    max_train = int(training_cfg.get("max_samples_train", 0))
    max_dev = int(training_cfg.get("max_samples_dev", 0))
    train_rows = load_rows(train_path, limit=max_train if max_train > 0 else None)
    dev_rows = load_rows(dev_path, limit=max_dev if max_dev > 0 else None)
    if not train_rows or not dev_rows:
        raise RuntimeError("train/dev rows cannot be empty")

    unknown_token = str(cfg.get("unknown_token", "__unknown__"))
    tool_vocab = build_tool_vocab(train_rows, unknown_token=unknown_token)
    tool_to_idx = {tool: idx for idx, tool in enumerate(tool_vocab)}

    retr_cfg = cfg.get("retriever", {})
    retriever = build_retriever(
        project_root=project_root,
        tool_corpus_relpath=str(retr_cfg.get("tool_corpus_file", "data/processed/tool_corpus.jsonl")),
        tool_vocab=tool_vocab,
    )
    retr_top_k = int(retr_cfg.get("top_k", 4))

    model_cfg = cfg.get("model", {})
    text_dim = int(model_cfg.get("text_dim", 256))
    retrieval_dim = int(model_cfg.get("retrieval_dim", 128))
    hidden_dim = int(model_cfg.get("hidden_dim", 256))
    modality_dim = int(model_cfg.get("modality_dim", 24))
    dropout = float(model_cfg.get("dropout", 0.1))

    device = _pick_device(str(training_cfg.get("device", "auto")))
    model = MainV1Model(
        num_tools=len(tool_vocab),
        text_dim=text_dim,
        retrieval_dim=retrieval_dim,
        hidden_dim=hidden_dim,
        modality_dim=modality_dim,
        dropout=dropout,
    ).to(device)

    lr = float(training_cfg.get("lr", 1e-3))
    weight_decay = float(training_cfg.get("weight_decay", 0.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    max_steps = int(training_cfg.get("max_steps", 100))
    batch_size = int(training_cfg.get("batch_size", 16))
    eval_interval = int(training_cfg.get("eval_interval", 20))
    log_interval = int(training_cfg.get("log_interval", 10))
    grad_clip = float(training_cfg.get("grad_clip", 1.0))

    loss_cfg = cfg.get("loss", {})
    tool_loss_weight = float(loss_cfg.get("tool_loss_weight", 1.0))
    unknown_loss_weight = float(loss_cfg.get("unknown_loss_weight", 0.5))

    inf_cfg = cfg.get("inference", {})
    unknown_threshold = float(inf_cfg.get("unknown_threshold", 0.55))
    top_k_tools = int(inf_cfg.get("top_k_tools", 3))
    unknown_temperature = float(inf_cfg.get("unknown_temperature", 1.0))
    unknown_energy_weight = float(inf_cfg.get("unknown_energy_weight", 0.2))

    runtime_cfg = cfg.get("runtime", {})
    cost_per_request = float(runtime_cfg.get("cost_per_request_usd", 0.0))

    train_losses: List[float] = []
    last_eval_metrics: Dict[str, float] = {}

    for step in range(1, max_steps + 1):
        model.train()
        batch_rows = sample_batch(train_rows, batch_size=batch_size, rng=rng)

        query_features, retrieval_features, modality_ids, _ = batch_encode_inputs(
            rows=batch_rows,
            text_dim=text_dim,
            retrieval_dim=retrieval_dim,
            retriever=retriever,
            retriever_top_k=retr_top_k,
            device=device,
        )
        tool_targets, unknown_targets = batch_targets(
            rows=batch_rows,
            tool_to_idx=tool_to_idx,
            unknown_token=unknown_token,
            device=device,
        )

        outputs = model(
            query_features=query_features,
            retrieval_features=retrieval_features,
            modality_ids=modality_ids,
        )
        loss_dict = model.compute_loss(
            outputs=outputs,
            tool_targets=tool_targets,
            unknown_targets=unknown_targets,
            tool_loss_weight=tool_loss_weight,
            unknown_loss_weight=unknown_loss_weight,
        )

        optimizer.zero_grad(set_to_none=True)
        loss_dict["total_loss"].backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        train_losses.append(float(loss_dict["total_loss"].item()))

        if step % log_interval == 0 or step == 1:
            logger.info(
                "step=%d total_loss=%.6f tool_loss=%.6f unknown_loss=%.6f",
                step,
                float(loss_dict["total_loss"].item()),
                float(loss_dict["tool_loss"].item()),
                float(loss_dict["unknown_loss"].item()),
            )

        if step % eval_interval == 0 or step == max_steps:
            metrics, _ = evaluate_rows(
                model=model,
                rows=dev_rows,
                tool_vocab=tool_vocab,
                retriever=retriever,
                retriever_top_k=retr_top_k,
                text_dim=text_dim,
                retrieval_dim=retrieval_dim,
                unknown_token=unknown_token,
                unknown_threshold=unknown_threshold,
                top_k_tools=top_k_tools,
                unknown_temperature=unknown_temperature,
                unknown_energy_weight=unknown_energy_weight,
                exp_id=exp_id,
                cost_per_request_usd=cost_per_request,
                device=device,
            )
            last_eval_metrics = metrics
            logger.info("dev_metrics_step_%d=%s", step, json.dumps(metrics, sort_keys=True))

    torch.save(
        {
            "exp_id": exp_id,
            "model_state_dict": model.state_dict(),
            "tool_vocab": tool_vocab,
            "config": cfg,
            "config_hash": cfg_hash,
        },
        ckpt_file,
    )

    final_metrics, predictions = evaluate_rows(
        model=model,
        rows=dev_rows,
        tool_vocab=tool_vocab,
        retriever=retriever,
        retriever_top_k=retr_top_k,
        text_dim=text_dim,
        retrieval_dim=retrieval_dim,
        unknown_token=unknown_token,
        unknown_threshold=unknown_threshold,
        top_k_tools=top_k_tools,
        unknown_temperature=unknown_temperature,
        unknown_energy_weight=unknown_energy_weight,
        exp_id=exp_id,
        cost_per_request_usd=cost_per_request,
        device=device,
    )
    write_jsonl(pred_file, predictions)

    report_payload = {
        "exp_id": exp_id,
        "config_hash": cfg_hash,
        "config_path": str(cfg_path),
        "train_file": str(train_path),
        "dev_file": str(dev_path),
        "device": str(device),
        "tool_vocab_size": len(tool_vocab),
        "tool_vocab": tool_vocab,
        "training_summary": {
            "max_steps": max_steps,
            "batch_size": batch_size,
            "loss_last": train_losses[-1] if train_losses else None,
            "loss_mean": (sum(train_losses) / len(train_losses)) if train_losses else None,
            "last_eval_metrics": last_eval_metrics,
        },
        "metrics": final_metrics,
        "artifacts": {
            "log_file": str(log_file),
            "prediction_file": str(pred_file),
            "report_file": str(report_file),
            "checkpoint_file": str(ckpt_file),
        },
    }
    write_json(report_file, report_payload)

    logger.info("final_metrics=%s", json.dumps(final_metrics, sort_keys=True))
    logger.info("wrote_predictions=%s rows=%d", pred_file, len(predictions))
    logger.info("wrote_report=%s", report_file)
    logger.info("wrote_checkpoint=%s", ckpt_file)

    print(
        json.dumps(
            {
                "exp_id": exp_id,
                "report_file": str(report_file),
                "prediction_file": str(pred_file),
                "checkpoint_file": str(ckpt_file),
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
