# Error Taxonomy (Week06)

## Scope
用于 `main_v1` 预测错误的统一分类，覆盖以下最小集合：
- 错选（wrong selection）
- 幻觉（hallucination）
- 拒答失败（reject failure）
- 格式失败（format failure）

## Decision Rules
给定单条预测记录，按以下顺序判定：

1. `format_failure`
- 条件：缺少关键字段或类型非法（`gold_tools/pred_tools/unknown_prob/confidence/...`）。

2. `reject_failure_unknown_miss`
- 条件：`is_unknown_gold=true` 且 `is_unknown_pred=false`。
- 含义：本应拒答/澄清，但系统错误执行具体工具。

3. `over_reject_false_unknown`
- 条件：`is_unknown_gold=false` 且 `is_unknown_pred=true`。
- 含义：已知工具场景被过度拒答。

4. `one_to_many_miss`
- 条件：`mapping_type=one_to_many` 且 `pred_tool` 不在 `gold_tools`。
- 含义：多标签场景未命中任一正确工具。

5. `hallucination`
- 条件：已知工具场景下预测错误，且 `pred_tool` 不在 `retrieved_tools`。
- 含义：选择头输出超出候选约束，表现为工具幻觉。

6. `wrong_tool_selection`
- 条件：已知工具场景下预测错误，但 `pred_tool` 在 `retrieved_tools`。
- 含义：候选中存在可选项，但排序/打分错误。

## Root Cause Tags
- `unknown_calibration_failure`: unknown 分支校准不足，unknown 样本被执行。
- `selector_confusion`: 候选集中存在正确工具但选择失败。
- `retrieval_miss`: 检索候选未覆盖正确工具，引发错选/幻觉。
- `one_to_many_alignment_gap`: one-to-many 目标与解码对齐不足。
- `format_or_schema_issue`: 输出字段缺失或格式违规。

## Priority Mapping
- P0: `format_failure`
- P1: `reject_failure_unknown_miss`
- P2: `hallucination`, `wrong_tool_selection`
- P3: `one_to_many_miss`, `over_reject_false_unknown`
