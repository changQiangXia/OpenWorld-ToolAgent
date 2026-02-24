# Week 09 Log

## 周目标
完成消融与鲁棒性压力测试框架，产出可复现报告与异常点核查记录。

## D1-D7 任务清单
- D1: [x] 冻结消融矩阵（w/o multimodal / retrieval / unknown / recover / calibration）
- D2: [x] 实现工具下线扰动集
- D3: [x] 实现工具替换扰动集
- D4: [x] 实现工具新增扰动集
- D5: [x] 输出消融与鲁棒性对比表
- D6: [x] 核查异常点并记录原因（w/o retrieval、replaced/newly_added 反向提升）
- D7: [x] 输出周报与 Week10 输入清单

## 本周新增文件
- `src/execution/week09_utils.py`
- `scripts/eval/run_ablations.py`
- `scripts/eval/run_robustness.py`
- `scripts/eval/run_ablations.sh`
- `scripts/eval/run_robustness.sh`
- `configs/eval/week09_ablation.yaml`
- `configs/eval/week09_robustness.yaml`
- `docs/ablation_report.md`
- `docs/robustness_report.md`
- `docs/weekly/week09_log.md`

## 本周新增产物
- `outputs/reports/20260223_week09_ablation_pser_mock_42_ablation_report.json`
- `outputs/reports/20260223_week09_ablation_pser_mock_42_ablation_table.csv`
- `outputs/reports/20260223_week09_robustness_pser_mock_42_robustness_report.json`
- `outputs/reports/20260223_week09_robustness_pser_mock_42_robustness_table.csv`
- `outputs/predictions/20260223_week09_ablation_pser_mock_42_ablation_traces/*.jsonl`
- `outputs/predictions/20260223_week09_robustness_pser_mock_42_robustness_traces/*.jsonl`

## 验证结果（Seed42）
- 样本选择：复杂样本 `52` 条（满足 `>=50`）
- 消融：
  - `w_o_recover` E2E 明显下降（`-0.2115`）
  - `w_o_multimodal` 小幅下降（`-0.0192`）
  - 其余项出现“动作变化但 E2E 持平/反向提升”现象（已记录异常）
- 鲁棒性：
  - `offline` 扰动 E2E 显著下降（`-0.3077`）
  - `replaced/newly_added` 出现反向提升（已记录异常并给出 Week10 修正计划）

## DoD 对齐
1. 每个模块贡献有量化证据：已完成（消融主表 + 逐场景 trace）。
2. 覆盖下线/替换/新增三类扰动：已完成。

## Week10 输入清单
1. `scripts/eval/run_ablations.py`
2. `scripts/eval/run_robustness.py`
3. `outputs/reports/20260223_week09_ablation_pser_mock_42_ablation_report.json`
4. `outputs/reports/20260223_week09_robustness_pser_mock_42_robustness_report.json`
5. `docs/ablation_report.md`
6. `docs/robustness_report.md`
