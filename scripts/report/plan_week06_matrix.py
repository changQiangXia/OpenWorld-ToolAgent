#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml


def _resolve(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else (root / path)


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    if not isinstance(obj, dict):
        raise ValueError(f"Matrix config must be mapping: {path}")
    return obj


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)
        f.write("\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate week06 experiment command plan from matrix yaml.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--matrix-config", type=Path, default=Path("configs/eval/week06_matrix.yaml"))
    parser.add_argument("--out-json", type=Path, default=Path("outputs/reports/week06_experiment_plan.json"))
    parser.add_argument("--out-commands", type=Path, default=Path("outputs/reports/week06_experiment_commands.txt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()

    matrix_path = _resolve(args.matrix_config, root)
    out_json = _resolve(args.out_json, root)
    out_commands = _resolve(args.out_commands, root)

    matrix = _load_yaml(matrix_path)
    models = matrix.get("models", [])
    seeds = [int(x) for x in matrix.get("seeds", [])]
    splits = [str(x) for x in matrix.get("eval_splits", [])]

    plan_runs: List[Dict[str, Any]] = []
    command_lines: List[str] = []

    for model in models:
        name = str(model.get("name", "model"))
        train_config = str(model.get("train_config", "configs/train/main_v1.yaml"))
        for seed in seeds:
            run_id = f"{name}_seed{seed}"
            train_cmd = (
                f"bash scripts/train/run_main_v1.sh "
                f"--train-config /root/autodl-tmp/project/{train_config} --seed {seed}"
            )
            eval_cmds = [
                f"bash scripts/eval/eval_main_v1.sh --train-config /root/autodl-tmp/project/{train_config} --split-name {split_name}"
                for split_name in splits
            ]

            plan_runs.append(
                {
                    "run_id": run_id,
                    "model": name,
                    "train_config": train_config,
                    "seed": seed,
                    "train_command": train_cmd,
                    "eval_commands": eval_cmds,
                }
            )

            command_lines.append(f"# {run_id}")
            command_lines.append(train_cmd)
            command_lines.extend(eval_cmds)
            command_lines.append("")

    payload = {
        "week": matrix.get("week", 6),
        "objective": matrix.get("objective", ""),
        "matrix_config": str(matrix_path),
        "num_runs": len(plan_runs),
        "runs": plan_runs,
        "notes": matrix.get("notes", []),
    }

    _write_json(out_json, payload)
    _write_text(out_commands, "\n".join(command_lines).rstrip() + "\n")

    print(
        json.dumps(
            {
                "out_json": str(out_json),
                "out_commands": str(out_commands),
                "num_runs": len(plan_runs),
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
