# Ablation Report (Week09)

## Goal
评估 PSER 流水线关键模块贡献：多模态、检索、unknown 检测、recover、校准。

## Setup
- 配置：`configs/eval/week09_ablation.yaml`
- 命令：`bash scripts/eval/run_ablations.sh`
- 数据：`data/splits/test.jsonl` + `*_test_policy_balanced.jsonl`
- 样本：复杂样本 `52` 条（满足 `>=50`）

## Main Results (Seed42)
汇总文件：
- `outputs/reports/20260223_week09_ablation_pser_mock_42_ablation_report.json`
- `outputs/reports/20260223_week09_ablation_pser_mock_42_ablation_table.csv`

关键指标（对 `full` 的变化）：
- `w_o_recover`: E2E `-0.2115`，Recover Success `-0.2692`，失败数 `+10`
- `w_o_multimodal`: E2E `-0.0192`
- `w_o_unknown_detection`: E2E `0.0000`（动作分布变化但成功率未变）
- `w_o_calibration`: E2E `0.0000`（动作分布变化但成功率未变）
- `w_o_retrieval`: E2E `+0.1538`（异常反向提升）

## Interpretation
1. recover 是当前链路中贡献最明确的模块；去掉后成功率显著下滑。
2. 多模态特征对复杂样本有轻度正贡献。
3. unknown 检测与校准在该批样本上主要改变 `reject/clarify` 比例，未改变最终 E2E。
4. `w_o_retrieval` 反向提升属于异常现象，当前不作为“检索无用”的结论。

## Anomaly Check (D6)
异常：`w_o_retrieval` 指标提升。  
可能原因：
1. 当前 `MockExecutor` + 合成数据下，单候选路径减少了“错误重试链”。
2. 复杂样本中 one-to-many 与 unknown 混合分布使“少尝试”被动受益。
3. 检索消融实现仍保留了 `pred_tool`，未完全构造“检索缺失导致候选错漏”。

## Week10 Action Items
1. 在 ablation 中增加更强的检索退化扰动（删除正确候选、控制覆盖率）。
2. 将 unknown/calibration 消融评估拆到 known/unknown 子集分别统计。
3. 对关键消融做 3-seed 聚合，避免单 seed 偶然性。
