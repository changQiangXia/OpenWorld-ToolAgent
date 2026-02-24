# Stability Report (Week10)

## Goal
对主结论进行 3-seed 稳定性复核，输出可追溯主表与异常样本清单。

## Inputs
- 聚合脚本：`scripts/report/aggregate_seeds.py`
- seeds：`42, 43, 44`
- 汇总输出：
  - `outputs/reports/week10_seed_aggregate_summary.json`
  - `outputs/reports/final_main_table.csv`
  - `outputs/reports/week10_seed_anomalies.jsonl`

## Main Table Draft
主表文件：`outputs/reports/final_main_table.csv`  
覆盖设置：
1. Week06 base test（policy 前）
2. Week07 balanced policy（before/after）
3. Week08 E2E PSER（balanced）
4. Week09 ablation（full / w_o_recover）
5. Week09 robustness（baseline / tool_offline）

## 3-Seed Claim Consistency
来自 `outputs/reports/week10_seed_aggregate_summary.json` 的 claim 复核结果：

1. C1：Week07 balanced 提升 Unknown F1（test）
- per-seed delta: `+0.3125`, `+0.2941`, `+0.2927`
- 方向一致：`True`

2. C2：Week07 balanced 降低 Hallucination（test）
- per-seed delta: `-0.2333`, `-0.1500`, `-0.2500`
- 方向一致：`True`

3. C3：Week09 消融显示 recover 提升 E2E
- per-seed delta (full - w_o_recover): `+0.2115`, `+0.1346`, `+0.2308`
- 方向一致：`True`

4. C4：Week09 offline 扰动显著降低 E2E
- per-seed delta (tool_offline - baseline): `-0.3077`, `-0.2500`, `-0.3077`
- 方向一致：`True`

5. C5：Week08 失败码覆盖率稳定为 100%
- per-seed delta (`failure_with_code_rate - 1.0`): `0.0`, `0.0`, `0.0`
- 方向一致：`True`

## Seed-Level Anomaly Review
异常样本清单：`outputs/reports/week10_seed_anomalies.jsonl`（40 条）  
来源：
1. Week07 test policy balanced 的跨 seed 预测分歧（top 20）
2. Week08 E2E trace 的跨 seed 结果分歧（top 20）

高频异常特征：
1. `multimodal_conflict` / `underspecified_tool_goal` / `version_drift` 样本更易发生 seed 分歧。
2. 主要分歧形态为 `success <-> reject/clarify` 切换，而非格式错误。
3. 一部分样本是同一 `failure_code` 下 success 判定差异，说明 gold/pred 关系受 seed 影响较大。

## Week10 DoD Check
1. 主结论在 3 seeds 下方向一致：已完成（5/5 claim 一致）。
2. 主表数字可追溯到原始结果文件：已完成（`final_main_table.csv` 每行包含 source_files 列）。
