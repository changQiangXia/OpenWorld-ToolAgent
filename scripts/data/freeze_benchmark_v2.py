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
    parser = argparse.ArgumentParser(description="Freeze benchmark v2 metadata and report.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/splits/benchmark_v2_manifest.json"))
    parser.add_argument("--build-report", type=Path, default=Path("outputs/reports/benchmark_v2_build_report.json"))
    parser.add_argument("--quality-report", type=Path, default=Path("outputs/reports/benchmark_v2_quality_report_v2splits.json"))
    parser.add_argument("--report-md", type=Path, default=Path("docs/benchmark_v2_report.md"))
    parser.add_argument("--changelog-md", type=Path, default=Path("data/splits/benchmark_v2_changelog.md"))
    return parser.parse_args()


def _fmt_split_stats(split_stats: Dict[str, Any], split_name: str) -> str:
    row = split_stats.get(split_name, {}) if isinstance(split_stats, dict) else {}
    return (
        f"- {split_name}: rows={row.get('num_rows', 0)}, "
        f"unknown_count={row.get('unknown_count', 0)}, "
        f"unknown_ratio={row.get('unknown_ratio', 0.0):.4f}"
    )


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()

    manifest_path = _resolve(args.manifest, root)
    build_report_path = _resolve(args.build_report, root)
    quality_path = _resolve(args.quality_report, root)
    report_md_path = _resolve(args.report_md, root)
    changelog_path = _resolve(args.changelog_md, root)

    for p in [manifest_path, build_report_path, quality_path, report_md_path, changelog_path]:
        _ensure_under_root(p, root)

    manifest = _read_json(manifest_path)
    build_report = _read_json(build_report_path)
    quality = _read_json(quality_path)

    version = str(manifest.get("benchmark_version", "benchmark_v2"))
    generated_at = str(manifest.get("generated_at_utc", "unknown"))
    split_stats = manifest.get("split_stats", {})
    outputs = manifest.get("outputs", {})
    subset_files = manifest.get("subset_files", {})

    quality_status = str(quality.get("status", "UNKNOWN"))
    total_errors = int(quality.get("total_error_count", -1))

    build_options = build_report.get("build_options", {}) if isinstance(build_report.get("build_options"), dict) else {}
    counts = build_report.get("counts", {}) if isinstance(build_report.get("counts"), dict) else {}
    template_query_count = int(build_report.get("template_query_count", 0))
    dropped_template_query_count = int(build_report.get("dropped_template_query_count", 0))

    warn = quality.get("warnings", {}) if isinstance(quality.get("warnings"), dict) else {}
    template_warn = warn.get("template_query_rows", {}) if isinstance(warn.get("template_query_rows"), dict) else {}
    template_warn_count = int(template_warn.get("count", 0))

    report_md = f"""# Benchmark v2 Report

## Version
- benchmark_version: `{version}`
- generated_at_utc: `{generated_at}`

## Split Sizes
- train: {counts.get('train', 0)}
- dev: {counts.get('dev', 0)}
- test: {counts.get('test', 0)}

## Unknown Stats
{_fmt_split_stats(split_stats, "train")}
{_fmt_split_stats(split_stats, "dev")}
{_fmt_split_stats(split_stats, "test")}

## Build Options
- real_data_first: `{build_options.get('real_data_first', False)}`
- query_strategy: `{build_options.get('query_strategy', 'auto')}`
- strip_split_tags: `{build_options.get('strip_split_tags', True)}`
- drop_template_queries: `{build_options.get('drop_template_queries', False)}`
- force_include_gold_in_candidates: `{build_options.get('force_include_gold_in_candidates', True)}`

## Template Query Audit
- template_query_count_after_build: `{template_query_count}`
- dropped_template_query_count: `{dropped_template_query_count}`
- template_query_warning_count_from_quality: `{template_warn_count}`

## Subset Indices
- number_of_files: `{subset_files.get('num_files', 0)}`
- subsets_dir: `{outputs.get('subsets_dir', 'unknown')}`

## Quality Gates (v2)
- status: `{quality_status}`
- total_error_count: `{total_errors}`
- report_path: `{quality_path}`

## Traceability
- manifest: `{manifest_path}`
- build_report: `{build_report_path}`
- quality_report: `{quality_path}`
- train: `{outputs.get('train', 'unknown')}`
- dev: `{outputs.get('dev', 'unknown')}`
- test: `{outputs.get('test', 'unknown')}`
"""

    changelog_md = f"""# Benchmark v2 Changelog

## {datetime.now(timezone.utc).isoformat()}
- Freeze benchmark metadata for `{version}`.
- Record split outputs and subset indices from manifest.
- Record build report and quality gate result.
- Quality status: `{quality_status}` (errors={total_errors}).
- Report written to `{report_md_path}`.
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

