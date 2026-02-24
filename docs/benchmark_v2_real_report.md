# Benchmark v2 Report

## Version
- benchmark_version: `benchmark_v2_real`
- generated_at_utc: `2026-02-24T13:21:27.392364+00:00`

## Split Sizes
- train: 200
- dev: 200
- test: 200

## Unknown Stats
- train: rows=200, unknown_count=0, unknown_ratio=0.0000
- dev: rows=200, unknown_count=34, unknown_ratio=0.1700
- test: rows=200, unknown_count=49, unknown_ratio=0.2450

## Build Options
- real_data_first: `True`
- query_strategy: `auto`
- strip_split_tags: `True`
- drop_template_queries: `True`
- force_include_gold_in_candidates: `True`

## Template Query Audit
- template_query_count_after_build: `0`
- dropped_template_query_count: `0`
- template_query_warning_count_from_quality: `0`

## Subset Indices
- number_of_files: `17`
- subsets_dir: `/root/autodl-tmp/project/data/splits/subsets_v2_real`

## Quality Gates (v2)
- status: `PASS`
- total_error_count: `0`
- report_path: `/root/autodl-tmp/project/outputs/reports/benchmark_v2_real_quality_report.json`

## Traceability
- manifest: `/root/autodl-tmp/project/data/splits/benchmark_v2_real_manifest.json`
- build_report: `/root/autodl-tmp/project/outputs/reports/benchmark_v2_real_build_report.json`
- quality_report: `/root/autodl-tmp/project/outputs/reports/benchmark_v2_real_quality_report.json`
- train: `/root/autodl-tmp/project/data/splits/train_v2_real.jsonl`
- dev: `/root/autodl-tmp/project/data/splits/dev_v2_real.jsonl`
- test: `/root/autodl-tmp/project/data/splits/test_v2_real.jsonl`
