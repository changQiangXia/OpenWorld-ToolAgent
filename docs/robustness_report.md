# Robustness Report (Week09)

## Goal
评估三类工具扰动下的端到端鲁棒性：下线（offline）/替换（replaced）/新增（newly_added）。

## Setup
- 配置：`configs/eval/week09_robustness.yaml`
- 命令：`bash scripts/eval/run_robustness.sh`
- 数据：`data/splits/test.jsonl` + `*_test_policy_balanced.jsonl`
- 样本：复杂样本 `52` 条（满足 `>=50`）

## Main Results (Seed42)
汇总文件：
- `outputs/reports/20260223_week09_robustness_pser_mock_42_robustness_report.json`
- `outputs/reports/20260223_week09_robustness_pser_mock_42_robustness_table.csv`

相对 baseline 的变化：
- `tool_offline`:
  - E2E `-0.3077`
  - Recover Success `-0.0962`
  - 失败数 `+18`
- `tool_replaced`:
  - E2E `+0.1154`
  - Recover Success `+0.0769`
- `tool_newly_added`:
  - E2E `+0.1346`
  - Recover Success `+0.1731`

## Interpretation
1. `offline` 扰动显著降低系统可用性，符合预期，说明离线工具场景是当前主要脆弱点。
2. `replaced/newly_added` 出现“指标提升”现象，不能直接解读为真实鲁棒性增强。
3. 当前结论可稳定支持：`offline` 风险显著，需要优先优化。

## Anomaly Check (D6)
异常：`replaced/newly_added` 在 mock 下优于 baseline。  
可能原因：
1. mock 机制下 `unknown/reject/clarify` 的得分规则在部分样本更容易命中成功判据。
2. `newly_added` 场景中 known->unknown 转换使部分样本由“执行正确”转为“拒答即可正确”。
3. 本周为快速鲁棒性框架阶段，尚未接入真实执行器反馈噪声。

## Week10 Action Items
1. 将 robustness 报告拆分为 known-only 与 unknown-only 两条主表。
2. 对 `replaced/newly_added` 引入更贴近真实的失败注入（版本参数不兼容、候选别名冲突）。
3. 使用 3 seeds 做均值/方差报告，避免单次扰动偶然性。
