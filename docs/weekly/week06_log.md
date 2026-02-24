# Week 06 Log

## 周目标
产出主结果首轮分析与错误分类报告，形成 Week7 开放世界增强输入。

## D1-D7 任务清单
- D1: [x] 确定实验矩阵（模型规模/seed/评测split）
- D2: [x] 产出主结果首轮配置与执行命令（不执行长任务）
- D3: [x] 解析预测并抽样失败案例
- D4: [x] 建立错误分类体系（错选/幻觉/拒答失败/格式失败）
- D5: [x] 输出错误分析报告与改进优先级
- D6: [x] 修复明显工程问题（检索重复候选、脚本配置覆盖）
- D7: [x] 输出周报和 Week7 输入

## 本周新增文件
- `scripts/report/analyze_main_v1_errors.py`
- `scripts/report/plan_week06_matrix.py`
- `scripts/report/aggregate_week06_reports.py`
- `scripts/report/run_week06_matrix.sh`
- `configs/train/main_v1_small.yaml`
- `configs/train/main_v1_large.yaml`
- `configs/eval/week06_matrix.yaml`
- `docs/error_taxonomy.md`
- `docs/error_analysis_week06.md`
- `docs/weekly/week06_log.md`

## 本周新增产物
- `outputs/reports/main_v1_week06_dev_seed42.json`
- `outputs/reports/main_v1_week06_test_seed42.json`
- `outputs/reports/main_v1_week06_summary_seed42.json`
- `outputs/reports/main_v1_week06_main_results_seed42.json`
- `outputs/reports/main_v1_week06_dev_seed42_cases.jsonl`
- `outputs/reports/main_v1_week06_test_seed42_cases.jsonl`
- `outputs/reports/week06_experiment_plan.json`
- `outputs/reports/week06_experiment_commands.txt`
- `outputs/reports/main_v1_week06_matrix_summary.json`
- `outputs/reports/main_v1_week06_matrix_table.csv`

## 验证结果
- 失败案例核查：40 条（dev 20 + test 20）
- Top-3 失败原因：unknown 校准失败、选择头混淆、检索覆盖不足
- 分类体系与报告：已完成并可追溯到具体样本 ID
- Week06 矩阵：9 组 run 全部完成（3 model x 3 seeds）

## Week07 输入清单
1. `docs/error_taxonomy.md`
2. `docs/error_analysis_week06.md`
3. `outputs/reports/main_v1_week06_main_results_seed42.json`
4. `outputs/reports/week06_experiment_plan.json`
5. `outputs/reports/week06_experiment_commands.txt`
