# Real Data Ingestion (Image + Audio + Video)

## Goal
从公开真实多模态数据构建本项目可训练样本，并落到
`baseline1_train/dev/test -> benchmark_v1 + benchmark_v2` 流程。

## Supported Sources (this repo script)
1. COCO Captions (image)
2. LibriSpeech (audio)
3. UCF101 (video)

脚本：
- `scripts/data/build_real_mm_pool.py`
- `scripts/data/split_real_mm_pool.py`
- `scripts/data/run_real_data_pipeline.sh`

配置：
- `configs/data/real_public_sources.yaml`
- `configs/data/benchmark_v2_real.yaml`

## Directory Convention
```text
data/raw/public/
  coco/
  librispeech/
  ucf101/
```

## Download Notes
1. 所有下载和解压必须在 `/root/autodl-tmp/project` 下完成（数据盘）。
2. 先确认许可证和使用条款，再用于训练。
3. 下载后不要改动默认目录层级，脚本按固定结构读取。

## Build Commands
```bash
# Run full pipeline: real pool -> baseline splits -> benchmark_v1 -> benchmark_v2
bash scripts/data/run_real_data_pipeline.sh configs/data/real_public_sources.yaml

# (Optional) explicitly pass benchmark_v2 config as 2nd argument
bash scripts/data/run_real_data_pipeline.sh \
  configs/data/real_public_sources.yaml \
  configs/data/benchmark_v2_real.yaml
```

## Outputs
1. `data/raw/real_mm_pool.jsonl`
2. `outputs/reports/real_mm_pool_stats.json`
3. `data/splits/baseline1_train.jsonl`
4. `data/splits/baseline1_dev.jsonl`
5. `data/splits/baseline1_test.jsonl`
6. `outputs/reports/real_mm_split_audit.json`
7. `data/splits/train.jsonl`
8. `data/splits/dev.jsonl`
9. `data/splits/test.jsonl`
10. `outputs/reports/benchmark_v1_quality_report.json`
11. `data/splits/train_v2_real.jsonl`
12. `data/splits/dev_v2_real.jsonl`
13. `data/splits/test_v2_real.jsonl`
14. `outputs/reports/benchmark_v2_real_build_report.json`
15. `outputs/reports/benchmark_v2_real_quality_report.json`
16. `docs/benchmark_v2_real_report.md`
17. `data/splits/benchmark_v2_real_changelog.md`
