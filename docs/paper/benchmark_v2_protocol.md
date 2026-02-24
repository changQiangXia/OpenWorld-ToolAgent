# Benchmark v2 Protocol（正式版草案）

- 版本: v1.0-draft
- 日期: 2026-02-24
- 适用项目: Open-World Multimodal Tool Agent
- 关联文件:
  - `project/scripts/data/build_benchmark_v2.py`
  - `project/scripts/data/run_quality_gates_v2.py`
  - `project/docs/paper/benchmark_v2_row_schema_v1.json`
  - `project/docs/paper/benchmark_v2_schema_and_annotation_playbook_20260224.md`

---

## 1. 协议目标

Benchmark v2 面向“可投稿级”开放世界工具调用评测，核心目标:

1. 提供统一、可审计、可复现的数据协议
2. 支持 known/unknown 与鲁棒性场景的公平比较
3. 让模型、策略、恢复模块可在同一标准下对比

---

## 2. 任务定义

给定输入样本 `x`，系统输出:

1. 工具选择（或 unknown）
2. 置信度与不确定性
3. 必要时的拒答/澄清动作

目标是在以下维度取得平衡:

1. known 任务成功率
2. unknown 检测质量
3. 幻觉控制
4. 端到端稳定性与鲁棒性

---

## 3. 数据协议

### 3.1 行级 schema

所有样本必须满足:

1. 通过 `benchmark_v2_row_schema_v1.json` 校验
2. 核心字段完整:
   - `id/split/query/modality/media`
   - `candidates/gold_tools/is_unknown_gold/unknown_reason_type`
   - `mapping_type/ambiguity_type/tool_status`
   - `annotation/trace`

### 3.2 known / unknown 约束

1. known 样本:
   - `is_unknown_gold=false`
   - `unknown_reason_type=none`
2. unknown 样本:
   - `is_unknown_gold=true`
   - `gold_tools=["__unknown__"]`
   - `unknown_reason_type != none`

### 3.3 one-to-many 约束

1. `mapping_type=one_to_many` 时 `len(gold_tools)>=2`
2. `mapping_type=one_to_one` 时 `len(gold_tools)=1`

---

## 4. split 与分布要求

### 4.1 split 定义

1. `train`: 训练与调参（不可用于最终报告）
2. `dev`: 阈值标定、策略选择、错误分析
3. `test`: 最终主结果与论文报告

### 4.2 分布检查（建议门槛）

在 `dev/test` 建议满足:

1. 各模态样本占比 >= 10%
2. unknown 占比位于目标区间（建议 15%~30%）
3. `tool_status` 与 `ambiguity_type` 不出现极端稀疏

---

## 5. 评测指标协议

### 5.1 主指标（论文主表）

1. `known_tool_success_rate`
2. `unknown_detection_f1`
3. `end_to_end_success_rate`

### 5.2 安全与可靠性指标

1. `hallucination_rate`
2. `false_reject_rate`
3. `unknown_miss_rate`
4. `calibration`（ECE/Brier）
5. `recover_success_rate`
6. `latency_p95/p99` 与成本

### 5.3 切片协议

主结果必须按以下切片同时报告:

1. modality
2. mapping_type
3. tool_status
4. ambiguity_type
5. known vs unknown

---

## 6. 公平比较协议（Baselines）

所有模型对比需满足:

1. 同一 split 与同一数据版本
2. 同一候选集合规则
3. 同一 unknown 定义
4. 同一评测脚本与统计口径
5. 同一延迟/成本记录方式

若不满足，必须在主文标注“不可直接比较”。

---

## 7. 统计协议

### 7.1 随机性控制

1. 核心实验建议 >= 5 seeds
2. 报告 mean/std/95% CI
3. 保留 seed-level 异常样本清单

### 7.2 显著性检验

核心对比建议使用:

1. paired bootstrap test（主指标）
2. McNemar（分类一致性）

---

## 8. 质量门禁协议

发布前必须执行:

```bash
python scripts/data/run_quality_gates_v2.py \
  --project-root /root/autodl-tmp/project \
  --schema-file docs/paper/benchmark_v2_row_schema_v1.json \
  --train-file data/splits/train_v2.jsonl \
  --dev-file data/splits/dev_v2.jsonl \
  --test-file data/splits/test_v2.jsonl \
  --report-out outputs/reports/benchmark_v2_quality_report.json \
  --fail-on-error
```

最小通过条件:

1. `total_error_count == 0`
2. 报告文件与 manifest 可追溯

---

## 9. 构建与发布流程（推荐）

### 9.1 构建 v2 样本

```bash
python scripts/data/build_benchmark_v2.py \
  --project-root /root/autodl-tmp/project \
  --train-in data/splits/train.jsonl \
  --dev-in data/splits/dev.jsonl \
  --test-in data/splits/test.jsonl \
  --train-out data/splits/train_v2.jsonl \
  --dev-out data/splits/dev_v2.jsonl \
  --test-out data/splits/test_v2.jsonl \
  --manifest-out data/splits/benchmark_v2_manifest.json \
  --report-out outputs/reports/benchmark_v2_build_report.json
```

### 9.1.b real-data-first 构建（推荐）

当具备 `baseline1_*` 大规模真实样本时，建议优先使用 real-data-first 配置:

```bash
python scripts/data/build_benchmark_v2.py \
  --project-root /root/autodl-tmp/project \
  --config configs/data/benchmark_v2_real.yaml
```

该模式特性:

1. 默认输入切到 `baseline1_train/dev/test.jsonl`
2. `query_strategy=auto`，优先使用非模板 query
3. 支持 `strip_split_tags` 清理 `[split=... id=...]` 尾标
4. 支持 `drop_template_queries` 丢弃模板样本

### 9.2 质量门禁

执行第 8 节命令。

### 9.3 冻结发布

发布时至少保留:

1. `train_v2/dev_v2/test_v2.jsonl`
2. `benchmark_v2_manifest.json`
3. `benchmark_v2_build_report.json`
4. `benchmark_v2_quality_report.json`

---

## 10. 失败处理协议

若质量门禁失败:

1. 先修 schema 错误（字段级）
2. 再修语义错误（known/unknown/mapping 一致性）
3. 最后处理分布与模板化 warning

禁止带 `FAIL` 状态进入主实验。

---

## 11. 版本管理协议

### 11.1 版本号建议

1. `benchmark_v2.0`: 首个可投稿版本
2. `benchmark_v2.1`: 标签修订不改 split
3. `benchmark_v3.0`: split 或定义重大变更

### 11.2 变更记录

每次升级必须记录:

1. 字段变化
2. 标签变化
3. 分布变化
4. 对历史结果的影响评估

---

## 12. 与论文资产联动

建议同步维护:

1. `docs/paper/claim_evidence_matrix_template_20260224.md`
2. `docs/paper/top_tier_12week_execution_board_20260224.md`
3. 主结果表与显著性报告路径

目标是做到:

1. 每条 claim 可追溯到具体数据版本与评测报告
2. rebuttal 时可快速定位证据文件
