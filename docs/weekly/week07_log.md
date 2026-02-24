# Week 07 Log

## 周目标
实现开放世界增强（unknown 标定 + 拒答/澄清策略），并完成策略对比。

## D1-D7 任务清单
- D1: [x] 实现 unknown 阈值自动标定流程
- D2: [x] 实现拒答策略（无合适工具返回 unknown）
- D3: [x] 实现澄清策略模板（请求补充信息）
- D4: [x] 实现策略切换配置（strict/balanced/recall-first）
- D5: [x] 完成增强前后对比脚本
- D6: [x] 检查误拒答与漏检案例
- D7: [x] 输出周报和 Week8 集成清单

## 本周新增文件
- `src/uncertainty/calibration.py`
- `src/agent/policy.py`
- `configs/eval/open_world_policy.yaml`
- `scripts/eval/compare_open_world.py`
- `docs/open_world_policy.md`
- `docs/weekly/week07_log.md`

## 本周新增产物
- `outputs/reports/open_world_compare_strict_seed42.json`
- `outputs/reports/open_world_compare_balanced_seed42.json`
- `outputs/reports/open_world_compare_recall_first_seed42.json`
- `outputs/reports/open_world_compare_summary_seed42.json`
- `outputs/reports/open_world_compare_balanced_seed43.json`
- `outputs/reports/open_world_compare_balanced_seed44.json`
- `outputs/reports/open_world_compare_balanced_3seeds_summary.json`
- `outputs/predictions/*_policy_strict.jsonl`
- `outputs/predictions/*_policy_balanced.jsonl`
- `outputs/predictions/*_policy_recall_first.jsonl`

## 验证结果（Seed42）
- `balanced`:
  - dev Unknown F1: 0.0000 -> 0.2105
  - dev Hallucination: 0.4333 -> 0.2667
  - test Unknown F1: 0.0000 -> 0.3125
  - test Hallucination: 0.5333 -> 0.3000
  - test E2E: 0.2667 -> 0.3333

- `strict`:
  - Hallucination 接近清零，但误拒答明显偏高（不作为默认）

## 3-Seed 稳定性（base + balanced）
- dev: Unknown F1 平均 `+0.2813`，Hallucination 平均 `-0.1667`，E2E 平均 `+0.0056`
- test: Unknown F1 平均 `+0.2998`，Hallucination 平均 `-0.2111`，E2E 平均 `-0.0056`（近似持平）

## DoD 对齐
1. Unknown F1 相对 Week6 有可测提升：已完成（balanced/test +0.3125）
2. Hallucination 明显下降：已完成（balanced/test -0.2333）

## Week08 输入清单
1. `src/agent/policy.py`
2. `src/uncertainty/calibration.py`
3. `scripts/eval/compare_open_world.py`
4. `docs/open_world_policy.md`
5. `outputs/reports/open_world_compare_balanced_seed42.json`
6. `outputs/reports/open_world_compare_balanced_3seeds_summary.json`
