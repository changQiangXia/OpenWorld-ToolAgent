# Week 03 Log

## 周目标
完成工具语料与动态工具库流水线脚本，支持版本快照、unknown 样本构造与 schema 校验。

## D1-D7 任务清单
- D1: [x] 定义工具元数据 schema（name/version/task/modality/status）
- D2: [x] 实现工具语料构建与清洗脚本（默认 synthetic，可接 input jsonl）
- D3: [x] 实现版本快照与差异报告（v_t/v_t+1）
- D4: [x] 实现 unknown 样本构造脚本（ratio/seed/strategy 可配置）
- D5: [x] 实现数据统计脚本（类别/模态/长尾）
- D6: [x] 实现 schema 校验器
- D7: [x] 输出周报和 Week4 构建参数

## 今日新增文件
- `scripts/data/build_tool_corpus.py`
- `scripts/data/make_unknown_split.py`
- `scripts/data/validate_schema.py`
- `scripts/data/summarize_dataset_stats.py`
- `docs/data_schema.md`

## 备注
- 所有脚本均限制输出路径在 `project_root` 内。
- unknown 构造会写审计 JSON，便于复现与追踪。
