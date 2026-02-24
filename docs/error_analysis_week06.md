# Error Analysis Week06

## Input Artifacts
- main result snapshot: `outputs/reports/main_v1_week06_main_results_seed42.json`
- dev error summary: `outputs/reports/main_v1_week06_dev_seed42.json`
- test error summary: `outputs/reports/main_v1_week06_test_seed42.json`
- merged summary: `outputs/reports/main_v1_week06_summary_seed42.json`
- manual cases (dev): `outputs/reports/main_v1_week06_dev_seed42_cases.jsonl`
- manual cases (test): `outputs/reports/main_v1_week06_test_seed42_cases.jsonl`

## Main Metrics (Seed=42)
- dev: `tool_selection_accuracy=0.3415`, `hallucination_rate=0.45`, `unknown_detection_f1=0.0`, `end_to_end_success_rate=0.2333`
- test: `tool_selection_accuracy=0.3333`, `hallucination_rate=0.5333`, `unknown_detection_f1=0.0`, `end_to_end_success_rate=0.2667`

## Matrix Results (3 Models x 3 Seeds)
来源：`outputs/reports/main_v1_week06_matrix_summary.json`、`outputs/reports/main_v1_week06_matrix_table.csv`

dev (mean +- std):
- base: tool_acc `0.3496 +- 0.0230`, hall `0.4444 +- 0.0157`, e2e `0.2389 +- 0.0157`
- large: tool_acc `0.3171 +- 0.0000`, hall `0.4667 +- 0.0000`, e2e `0.2167 +- 0.0000`
- small: tool_acc `0.2764 +- 0.0304`, hall `0.4944 +- 0.0208`, e2e `0.1889 +- 0.0208`

test (mean +- std):
- base: tool_acc `0.3125 +- 0.0170`, hall `0.5500 +- 0.0136`, e2e `0.2500 +- 0.0136`
- large: tool_acc `0.2292 +- 0.0170`, hall `0.6167 +- 0.0136`, e2e `0.1833 +- 0.0136`
- small: tool_acc `0.2847 +- 0.0393`, hall `0.5722 +- 0.0314`, e2e `0.2278 +- 0.0314`

结论（Week6）：当前设置下 `base` 为综合最优，且三种规模均未解决 unknown 检测（Unknown F1 全部为 0.0）。

## Error Distribution
- combined samples: 120
- combined failures: 90
- error rate: 0.75

Category counts (combined):
- `reject_failure_unknown_miss`: 31
- `wrong_tool_selection`: 29
- `hallucination`: 20
- `one_to_many_miss`: 10

Root-cause counts (combined):
- `unknown_calibration_failure`: 31
- `selector_confusion`: 29
- `retrieval_miss`: 24
- `one_to_many_alignment_gap`: 6

## Slice Highlights
- dev highest fail modality: `audio` (0.8667)
- test highest fail modality: `image` (0.8667)
- dev highest fail tool status: `replaced` (0.8667)
- test highest fail tool status: `offline` (0.8667)
- worst ambiguity on test: `underspecified_tool_goal` (0.9167)

## Manual Review (>=20 Cases)
已人工核查 40 条失败样本（dev 20 + test 20），覆盖四类核心失败（每类各 5 条）。

Reviewed IDs (dev, 20):
- `dev_00042`, `dev_00027`, `dev_00031`, `dev_00007`, `dev_00000`
- `dev_00057`, `dev_00038`, `dev_00022`, `dev_00029`, `dev_00002`
- `dev_00004`, `dev_00041`, `dev_00052`, `dev_00056`, `dev_00048`
- `dev_00049`, `dev_00040`, `dev_00023`, `dev_00003`, `dev_00047`

Reviewed IDs (test, 20):
- `test_00007`, `test_00003`, `test_00052`, `test_00033`, `test_00056`
- `test_00022`, `test_00045`, `test_00057`, `test_00049`, `test_00010`
- `test_00041`, `test_00025`, `test_00024`, `test_00011`, `test_00051`
- `test_00005`, `test_00017`, `test_00016`, `test_00037`, `test_00039`

## Top-3 Failure Reasons and Fixes
1. unknown 校准失败（31）
- 现象：unknown 样本几乎都被当作已知工具执行，`unknown_detection_f1=0.0`。
- 最小修复：做阈值扫描 + 温度标定；增加 unknown 正样本占比（训练阶段）。

2. 选择头混淆（29）
- 现象：候选中含正确工具但最终选错。
- 最小修复：增加排序损失/边际约束；对高置信错选做 hard-negative 训练。

3. 检索覆盖不足（24）
- 现象：正确工具未进入候选，诱发错选或幻觉。
- 最小修复：改检索打分与去重策略，提升 gold 覆盖；约束解码优先候选内选择。

## Week06 Obvious Bug Fixes (Applied)
- 修复检索重复候选偏置：`src/retriever/simple_retriever.py` 去重同名工具，仅保留最高分版本。
- 修复脚本参数覆盖问题：`scripts/train/run_main_v1.sh` 与 `scripts/eval/eval_main_v1.sh` 支持外部 `--train-config` 覆盖默认值。

## Week07 Input
- `docs/error_taxonomy.md`
- `docs/error_analysis_week06.md`
- `outputs/reports/main_v1_week06_main_results_seed42.json`
- `outputs/reports/week06_experiment_plan.json`
- `outputs/reports/week06_experiment_commands.txt`
