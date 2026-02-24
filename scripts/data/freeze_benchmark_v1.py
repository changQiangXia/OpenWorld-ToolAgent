#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _resolve(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else (root / path)


def _ensure_under_root(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Path escapes project root: {path}") from exc


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"JSON object required: {path}")
    return obj


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze benchmark v1 metadata and report.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/splits/benchmark_v1_manifest.json"))
    parser.add_argument("--quality-report", type=Path, default=Path("outputs/reports/benchmark_v1_quality_report.json"))
    parser.add_argument("--report-md", type=Path, default=Path("docs/benchmark_v1_report.md"))
    parser.add_argument("--changelog-md", type=Path, default=Path("data/splits/benchmark_v1_changelog.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()

    manifest_path = _resolve(args.manifest, root)
    quality_path = _resolve(args.quality_report, root)
    report_md_path = _resolve(args.report_md, root)
    changelog_path = _resolve(args.changelog_md, root)

    for p in [manifest_path, quality_path, report_md_path, changelog_path]:
        _ensure_under_root(p, root)

    manifest = _read_json(manifest_path)
    quality = _read_json(quality_path)

    version = str(manifest.get("benchmark_version", "benchmark_v1"))
    ann_ver = str(manifest.get("annotation_version", "ann_v1"))
    sizes = manifest.get("sizes", {})
    mapping = manifest.get("mapping_distribution", {})
    subsets = manifest.get("subset_files", {})

    quality_status = str(quality.get("status", "UNKNOWN"))
    total_errors = int(quality.get("total_error_count", -1))

    report_md = f"""# Benchmark v1 Report

## Version
- benchmark_version: `{version}`
- annotation_version: `{ann_ver}`
- generated_at_utc: `{manifest.get('generated_at_utc', 'unknown')}`

## Split Sizes
- train: {sizes.get('train', 0)}
- dev: {sizes.get('dev', 0)}
- test: {sizes.get('test', 0)}
- total: {sizes.get('total', 0)}

## Mapping Distribution
- train: {mapping.get('train', {})}
- dev: {mapping.get('dev', {})}
- test: {mapping.get('test', {})}

## Subset Indices
- number_of_files: {subsets.get('num_files', 0)}
- subsets_dir: `{manifest.get('outputs', {}).get('subsets_dir', 'unknown')}`

## Quality Gates
- status: `{quality_status}`
- total_error_count: `{total_errors}`
- report_path: `{quality_path}`

## Traceability
- manifest: `{manifest_path}`
- train: `{manifest.get('outputs', {}).get('train', 'unknown')}`
- dev: `{manifest.get('outputs', {}).get('dev', 'unknown')}`
- test: `{manifest.get('outputs', {}).get('test', 'unknown')}`
"""

    changelog_md = f"""# Benchmark v1 Changelog

## {datetime.now(timezone.utc).isoformat()}
- Freeze benchmark metadata for `{version}`.
- Confirm split outputs and subset indices are generated.
- Record quality gate result: `{quality_status}` (errors={total_errors}).
- Store report in `{report_md_path}`.
"""

    _write_text(report_md_path, report_md)
    _write_text(changelog_path, changelog_md)

    print(
        json.dumps(
            {
                "report_md": str(report_md_path),
                "changelog_md": str(changelog_path),
                "quality_status": quality_status,
                "total_error_count": total_errors,
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
