# Week 04 Log

## 周目标
完成 Benchmark v1 构建、one-to-many 映射、子集索引与质量门禁，并冻结版本报告。

## D1-D7 任务清单
- D1: [x] 生成 train/dev/test 首版切分（`data/splits/train.jsonl` 等）
- D2: [x] 加入 one-to-many 标注映射并校验一致性（`mapping_type` + `gold_tools`）
- D3: [x] 生成子测试集索引（模态/难度/映射/工具状态）
- D4: [x] 跑质量门禁（重复样本/标签冲突/路径失效）
- D5: [x] 修复门禁失败项并重生成（修复跨 split query 泄漏）
- D6: [x] 冻结 benchmark v1（manifest + report + changelog）
- D7: [x] 输出周报和 Week5 输入清单

## 本周新增文件
- `scripts/data/build_benchmark_v1.py`
- `scripts/data/run_quality_gates.py`
- `scripts/data/freeze_benchmark_v1.py`
- `data/splits/train.jsonl`
- `data/splits/dev.jsonl`
- `data/splits/test.jsonl`
- `data/splits/subsets/*.json`
- `data/splits/benchmark_v1_manifest.json`
- `data/splits/benchmark_v1_changelog.md`
- `outputs/reports/benchmark_v1_quality_report.json`
- `docs/benchmark_v1_report.md`

## 验证结果
- 质量门禁: PASS（error_count=0）
- schema 校验:
  - train: OK (120)
  - dev: OK (60)
  - test: OK (60)

## Week5 输入清单
1. `data/splits/train.jsonl`
2. `data/splits/dev.jsonl`
3. `data/splits/test.jsonl`
4. `data/splits/subsets/*.json`
5. `docs/benchmark_v1_report.md`
6. `outputs/reports/benchmark_v1_quality_report.json`

## 风险与补救记录
- 风险: 首轮门禁发现跨 split query 文本重复导致冲突。
- 补救: 在构建脚本中保留 `query_raw`，并在 `query` 注入 `split/id` 标签，重建后门禁通过。
