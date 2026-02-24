#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_BOOTSTRAP))

from src.agent.main_v1_data import compute_prediction_metrics, is_unknown_gold, normalize_gold_tools
from src.agent.qwen_vl_whisper import QwenVLWhisperToolSelector
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


def _resolve(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else (root / path)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _dedup(items: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        token = str(item).strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _load_tools_from_corpus(path: Path) -> List[str]:
    if not path.exists():
        return []
    tools: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                name = str(obj.get("name", "")).strip()
                if name:
                    tools.append(name)
    return _dedup(tools)


def _row_candidates(
    row: Dict[str, Any],
    fallback_tools: Sequence[str],
    unknown_token: str,
    candidate_limit: int,
) -> List[str]:
    raw = row.get("candidates")
    cands = [str(x) for x in raw] if isinstance(raw, list) else []
    cands = [x for x in cands if x and x != unknown_token]
    if not cands:
        cands = [str(x) for x in fallback_tools if str(x) and str(x) != unknown_token]
    cands = _dedup(cands)
    if candidate_limit > 0:
        cands = cands[:candidate_limit]
    return cands


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5-VL-7B + Whisper on benchmark split.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/eval/qwen25_vl_whisper.yaml"))
    parser.add_argument("--split-name", type=str, default=None)
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    cfg_path = _resolve(args.config, project_root)
    cfg = load_yaml(cfg_path)

    named_cfg = {"eval": cfg}
    cfg_hash = compute_config_hash(named_cfg)

    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    task_name = str(cfg.get("task_name", "open_world_tool_agent"))
    model_name = str(cfg.get("model_name", "qwen25_vl_whisper"))
    exp_id = make_exp_id(task=task_name, model=model_name, seed=seed)

    output_dirs = cfg.get("output_dirs", {}) if isinstance(cfg.get("output_dirs"), dict) else {}
    logs_dir = _resolve(Path(str(output_dirs.get("logs", "outputs/logs"))), project_root)
    preds_dir = _resolve(Path(str(output_dirs.get("predictions", "outputs/predictions"))), project_root)
    reports_dir = _resolve(Path(str(output_dirs.get("reports", "outputs/reports"))), project_root)
    ensure_dirs([logs_dir, preds_dir, reports_dir], project_root)
    for p in [logs_dir, preds_dir, reports_dir, cfg_path]:
        ensure_within_root(p, project_root)

    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data"), dict) else {}
    split_name = str(
        args.split_name
        if args.split_name is not None
        else data_cfg.get("split_name", "dev")
    )
    if split_name not in {"train", "dev", "test"}:
        raise ValueError(f"split_name must be one of train/dev/test, got: {split_name}")
    if args.split_file is not None:
        split_path = _resolve(args.split_file, project_root)
    else:
        split_template = str(data_cfg.get("split_file_template", "data/splits/{split}.jsonl"))
        split_path = _resolve(Path(split_template.format(split=split_name)), project_root)
    ensure_within_root(split_path, project_root)

    pred_file = preds_dir / f"{exp_id}_{split_name}_qwen_eval.jsonl"
    report_file = reports_dir / f"{exp_id}_{split_name}_qwen_eval.json"
    log_file = logs_dir / f"{exp_id}_{split_name}_qwen_eval.log"

    logger = setup_logger(log_file, exp_id=f"{exp_id}_{split_name}", cfg_hash=cfg_hash, level="INFO")
    logger.info("start qwen25-vl + whisper eval")
    logger.info("config=%s", cfg_path)
    logger.info("split=%s path=%s", split_name, split_path)

    rows = _read_jsonl(split_path)
    if not rows:
        raise RuntimeError(f"no rows found in split file: {split_path}")

    cfg_max_samples = int(data_cfg.get("max_samples", 0))
    user_max_samples = int(args.max_samples) if args.max_samples is not None else 0
    max_samples = user_max_samples if user_max_samples > 0 else cfg_max_samples
    if max_samples > 0:
        rows = rows[:max_samples]

    runtime_cfg = cfg.get("runtime", {}) if isinstance(cfg.get("runtime"), dict) else {}
    generation_cfg = cfg.get("generation", {}) if isinstance(cfg.get("generation"), dict) else {}
    models_cfg = cfg.get("models", {}) if isinstance(cfg.get("models"), dict) else {}
    asr_cfg = cfg.get("asr", {}) if isinstance(cfg.get("asr"), dict) else {}

    unknown_token = str(cfg.get("unknown_token", "__unknown__"))
    unknown_threshold = float(runtime_cfg.get("unknown_threshold", 0.55))
    candidate_limit = int(runtime_cfg.get("candidate_limit", 8))
    retriever_top_k = int(runtime_cfg.get("retriever_top_k", 4))
    cost_per_request = float(runtime_cfg.get("cost_per_request_usd", 0.0))

    tools = [str(x) for x in cfg.get("tools", []) if str(x)]
    if not tools:
        corpus_path = _resolve(Path("data/processed/tool_corpus.jsonl"), project_root)
        tools = _load_tools_from_corpus(corpus_path)
    if not tools:
        raise RuntimeError("No tools available. Set `tools` in config or provide tool corpus.")

    if args.dry_run:
        print(
            json.dumps(
                {
                    "exp_id": exp_id,
                    "dry_run": True,
                    "rows": len(rows),
                    "split_file": str(split_path),
                    "prediction_file": str(pred_file),
                    "report_file": str(report_file),
                    "log_file": str(log_file),
                    "tools": tools,
                },
                ensure_ascii=True,
            )
        )
        return

    selector = QwenVLWhisperToolSelector(
        project_root=project_root,
        qwen_model_dir=_resolve(Path(str(models_cfg.get("qwen_model_dir", "models/Qwen2.5-VL-7B-Instruct"))), project_root),
        whisper_model_dir=_resolve(Path(str(models_cfg.get("whisper_model_dir", "models/whisper-large-v3"))), project_root),
        device=str(runtime_cfg.get("device", "cuda:0")),
        dtype=str(runtime_cfg.get("dtype", "bfloat16")),
        media_mode=str(runtime_cfg.get("media_mode", "multimodal")),
        video_frame_strategy=str(runtime_cfg.get("video_frame_strategy", "first_frame")),
        trust_remote_code=bool(models_cfg.get("trust_remote_code", True)),
        use_flash_attention_2=bool(models_cfg.get("use_flash_attention_2", True)),
        load_in_4bit=bool(models_cfg.get("load_in_4bit", False)),
        max_new_tokens=int(generation_cfg.get("max_new_tokens", 96)),
        do_sample=bool(generation_cfg.get("do_sample", False)),
        temperature=float(generation_cfg.get("temperature", 0.0)),
        top_p=float(generation_cfg.get("top_p", 1.0)),
        asr_chunk_length_s=float(asr_cfg.get("chunk_length_s", 30.0)),
        asr_batch_size=int(asr_cfg.get("batch_size", 8)),
        asr_language=str(asr_cfg.get("language")) if "language" in asr_cfg else None,
    )

    predictions: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        cands = _row_candidates(
            row=row,
            fallback_tools=tools,
            unknown_token=unknown_token,
            candidate_limit=candidate_limit,
        )
        pred = selector.predict(
            row=row,
            candidate_tools=cands,
            unknown_token=unknown_token,
            unknown_threshold=unknown_threshold,
        )

        gold_tools = normalize_gold_tools(row, unknown_token=unknown_token)
        predictions.append(
            {
                "exp_id": exp_id,
                "id": str(row.get("id", "")),
                "split": str(row.get("split", split_name)),
                "query": str(row.get("query", "")),
                "modality": str(row.get("modality", "unknown")),
                "ambiguity_type": str(row.get("ambiguity_type", "unknown")),
                "tool_status": str(row.get("tool_status", "unknown")),
                "mapping_type": str(row.get("mapping_type", "one_to_one")),
                "gold_tools": gold_tools,
                "gold_tool": gold_tools[0],
                "pred_tool": pred.pred_tool,
                "pred_tools": pred.pred_tools,
                "pred_tool_scores": [float(x) for x in pred.pred_tool_scores],
                "unknown_prob": float(pred.unknown_prob),
                "confidence": float(pred.confidence),
                "is_unknown_pred": bool(pred.is_unknown_pred),
                "is_unknown_gold": bool(is_unknown_gold(gold_tools, unknown_token=unknown_token)),
                "retrieved_tools": cands[: max(1, retriever_top_k)],
                "latency_ms": float(pred.latency_ms),
                "cost_usd": float(cost_per_request),
                "raw_model_response": pred.raw_response,
                "asr_text": pred.asr_text,
            }
        )

        if idx == 1 or idx % 20 == 0:
            logger.info("processed=%d/%d", idx, len(rows))

    metrics = compute_prediction_metrics(predictions, unknown_token=unknown_token)

    write_jsonl(pred_file, predictions)
    write_json(
        report_file,
        {
            "exp_id": exp_id,
            "config_path": str(cfg_path),
            "config_hash": cfg_hash,
            "split_name": split_name,
            "split_file": str(split_path),
            "num_rows": len(predictions),
            "unknown_token": unknown_token,
            "metrics": metrics,
            "artifacts": {
                "log_file": str(log_file),
                "prediction_file": str(pred_file),
                "report_file": str(report_file),
            },
        },
    )

    logger.info("metrics=%s", json.dumps(metrics, sort_keys=True))
    logger.info("wrote_predictions=%s rows=%d", pred_file, len(predictions))
    logger.info("wrote_report=%s", report_file)

    print(
        json.dumps(
            {
                "exp_id": exp_id,
                "prediction_file": str(pred_file),
                "report_file": str(report_file),
                "metrics": metrics,
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
