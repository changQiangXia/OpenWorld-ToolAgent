# Week 08 Log

## 周目标
完成端到端 `Plan-Select-Execute-Recover` 集成，并在复杂样本上跑通 smoke test。

## D1-D7 任务清单
- D1: [x] 定义端到端状态机与失败码
- D2: [x] 实现执行器接口（mock 已完成，真执行保留扩展点）
- D3: [x] 实现 recover 机制（重试/拒答/澄清/终止）
- D4: [x] 接入 E2E 指标计算脚本
- D5: [x] 跑通复杂样本 smoke test
- D6: [x] 修复状态字段缺失问题（补充 `ambiguity_type` 到 trace）
- D7: [x] 输出周报与 Week9 输入清单

## 本周新增文件
- `src/execution/pipeline.py`
- `src/execution/recover.py`
- `src/execution/__init__.py`
- `scripts/eval/eval_e2e.py`
- `scripts/eval/eval_e2e.sh`
- `configs/eval/e2e_v1.yaml`
- `docs/e2e_protocol.md`
- `docs/weekly/week08_log.md`

## 本周新增产物
- `outputs/logs/20260223_open_world_e2e_eval_plan_select_execute_recover_v1_42_e2e.log`
- `outputs/predictions/20260223_open_world_e2e_eval_plan_select_execute_recover_v1_42_e2e_traces.jsonl`
- `outputs/reports/20260223_open_world_e2e_eval_plan_select_execute_recover_v1_42_e2e_eval.json`

## 验证结果（Seed42, balanced policy 输入）
- 复杂样本数：`52`（满足 DoD `>=50`）
- E2E Success Rate：`0.4808`
- Recover Success Rate：`0.2692`
- Avg Attempts：`2.4423`
- Avg Latency：`667.17 ms`
- 失败码覆盖率：`failure_with_code_rate = 1.0`
- 恢复路径覆盖率：`failure_with_recover_path_rate = 1.0`
- 失败码分布：`E_POLICY_CLARIFY=22, E_POLICY_REJECT=9, OK=21`

## DoD 对齐
1. 至少 50 条复杂样本可自动完成：已完成（52 条）。
2. 每次失败都有明确失败码与恢复路径：已完成（覆盖率均为 100%）。

## Week09 输入清单
1. `scripts/eval/eval_e2e.py`
2. `configs/eval/e2e_v1.yaml`
3. `outputs/reports/20260223_open_world_e2e_eval_plan_select_execute_recover_v1_42_e2e_eval.json`
4. `outputs/predictions/20260223_open_world_e2e_eval_plan_select_execute_recover_v1_42_e2e_traces.jsonl`
5. `docs/e2e_protocol.md`
