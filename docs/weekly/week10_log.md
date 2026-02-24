# Week 10 Log

## 周目标
冻结主结论最小实验集，完成 3-seed 聚合、稳定性复核和主表草案输出。

## D1-D7 任务清单
- D1: [x] 冻结主结论最小实验集（Week06/07/08/09 核心设置）
- D2: [x] 实现 3-seed 聚合脚本 `scripts/report/aggregate_seeds.py`
- D3: [x] 计算均值/标准差/95% CI 并输出主表草案
- D4: [x] 检查 seed 间异常样本并输出清单
- D5: [x] 生成 `outputs/reports/final_main_table.csv`
- D6: [x] 运行回归检查（语法 + 单测 + 严格聚合执行）
- D7: [x] 输出周报与 Week11 论文素材输入

## 本周新增文件
- `scripts/report/aggregate_seeds.py`
- `scripts/report/run_week10_aggregate.sh`
- `docs/stability_report.md`
- `docs/weekly/week10_log.md`

## 本周新增产物
- `outputs/reports/week10_seed_aggregate_summary.json`
- `outputs/reports/final_main_table.csv`
- `outputs/reports/week10_seed_anomalies.jsonl`
- `outputs/reports/20260223_open_world_e2e_eval_plan_select_execute_recover_v1_43_e2e_eval.json`
- `outputs/reports/20260223_open_world_e2e_eval_plan_select_execute_recover_v1_44_e2e_eval.json`
- `outputs/reports/20260223_week09_ablation_pser_mock_43_ablation_report.json`
- `outputs/reports/20260223_week09_ablation_pser_mock_44_ablation_report.json`
- `outputs/reports/20260223_week09_robustness_pser_mock_43_robustness_report.json`
- `outputs/reports/20260223_week09_robustness_pser_mock_44_robustness_report.json`

## 验证结果（3 Seeds）
- Claim 一致性：`5/5` 方向一致
  - Unknown F1 提升：一致
  - Hallucination 降低：一致
  - Recover 贡献为正：一致
  - Offline 扰动降级：一致
  - E2E failure code 覆盖率=100%：一致
- 异常样本：40 条（Week07 top20 + Week08 top20）

## 回归检查（D6）
- 语法检查：
  - `python -m py_compile scripts/report/aggregate_seeds.py`
- 聚合执行（严格模式）：
  - `python scripts/report/aggregate_seeds.py --project-root /root/autodl-tmp/project --strict`
- 单测：
  - `python -m unittest tests.test_main_v1_smoke tests.test_e2e_smoke`
  - 结果：`Ran 4 tests ... OK`

## DoD 对齐
1. 主结论 3-seed 方向一致：已完成。
2. 主表数字可追溯：已完成（`final_main_table.csv` 含 `source_files`）。

## Week11 输入清单
1. `outputs/reports/final_main_table.csv`
2. `outputs/reports/week10_seed_aggregate_summary.json`
3. `outputs/reports/week10_seed_anomalies.jsonl`
4. `docs/stability_report.md`
5. `docs/ablation_report.md`
6. `docs/robustness_report.md`
