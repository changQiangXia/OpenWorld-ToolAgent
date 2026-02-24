#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_BOOTSTRAP))

from src.agent.main_v1_data import build_retriever, load_rows
from src.agent.main_v1_eval import evaluate_rows
from src.agent.main_v1_model import MainV1Model
from src.agent.runtime_utils import latest_file, load_yaml, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate main_v1 checkpoint on a split jsonl.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, default=Path("configs/train/main_v1.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--split-name", type=str, default="dev", choices=["dev", "test", "train"])
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
    cfg = load_yaml(_resolve(args.train_config, project_root))

    output_dirs = cfg.get("output_dirs", {})
    ckpt_dir = _resolve(Path(output_dirs.get("checkpoints", "outputs/checkpoints")), project_root)
    preds_dir = _resolve(Path(output_dirs.get("predictions", "outputs/predictions")), project_root)
    reports_dir = _resolve(Path(output_dirs.get("reports", "outputs/reports")), project_root)

    if args.checkpoint is not None:
        ckpt_path = _resolve(args.checkpoint, project_root)
    else:
        latest = latest_file(ckpt_dir, "*.pt")
        if latest is None:
            raise FileNotFoundError(f"No checkpoint found under {ckpt_dir}")
        ckpt_path = latest

    split_path: Path
    if args.split_file is not None:
        split_path = _resolve(args.split_file, project_root)
    else:
        data_cfg = cfg.get("data", {})
        if args.split_name == "train":
            split_path = _resolve(Path(data_cfg.get("train_file", "data/splits/train.jsonl")), project_root)
        elif args.split_name == "test":
            split_path = _resolve(Path(data_cfg.get("test_file", "data/splits/test.jsonl")), project_root)
        else:
            split_path = _resolve(Path(data_cfg.get("dev_file", "data/splits/dev.jsonl")), project_root)

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    tool_vocab = list(checkpoint["tool_vocab"])
    exp_id = str(checkpoint.get("exp_id", ckpt_path.stem))

    training_cfg = cfg.get("training", {})
    device = _pick_device(str(training_cfg.get("device", "auto")))

    model_cfg = cfg.get("model", {})
    model = MainV1Model(
        num_tools=len(tool_vocab),
        text_dim=int(model_cfg.get("text_dim", 256)),
        retrieval_dim=int(model_cfg.get("retrieval_dim", 128)),
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        modality_dim=int(model_cfg.get("modality_dim", 24)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    retr_cfg = cfg.get("retriever", {})
    retriever = build_retriever(
        project_root=project_root,
        tool_corpus_relpath=str(retr_cfg.get("tool_corpus_file", "data/processed/tool_corpus.jsonl")),
        tool_vocab=tool_vocab,
    )

    inf_cfg = cfg.get("inference", {})
    unknown_token = str(cfg.get("unknown_token", "__unknown__"))
    rows = load_rows(split_path)

    metrics, predictions = evaluate_rows(
        model=model,
        rows=rows,
        tool_vocab=tool_vocab,
        retriever=retriever,
        retriever_top_k=int(retr_cfg.get("top_k", 4)),
        text_dim=int(model_cfg.get("text_dim", 256)),
        retrieval_dim=int(model_cfg.get("retrieval_dim", 128)),
        unknown_token=unknown_token,
        unknown_threshold=float(inf_cfg.get("unknown_threshold", 0.55)),
        top_k_tools=int(inf_cfg.get("top_k_tools", 3)),
        unknown_temperature=float(inf_cfg.get("unknown_temperature", 1.0)),
        unknown_energy_weight=float(inf_cfg.get("unknown_energy_weight", 0.2)),
        exp_id=exp_id,
        cost_per_request_usd=float(cfg.get("runtime", {}).get("cost_per_request_usd", 0.0)),
        device=device,
    )

    pred_path = preds_dir / f"{exp_id}_{args.split_name}_eval.jsonl"
    report_path = reports_dir / f"{exp_id}_{args.split_name}_eval.json"
    write_jsonl(pred_path, predictions)
    write_json(
        report_path,
        {
            "exp_id": exp_id,
            "checkpoint": str(ckpt_path),
            "split_file": str(split_path),
            "metrics": metrics,
            "artifacts": {
                "prediction_file": str(pred_path),
                "report_file": str(report_path),
            },
        },
    )

    print(
        json.dumps(
            {
                "report_file": str(report_path),
                "prediction_file": str(pred_path),
                "metrics": metrics,
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
