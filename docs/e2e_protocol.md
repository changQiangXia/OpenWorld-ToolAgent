# E2E Protocol (Week08)

## Goal
将 Week07 的开放世界策略接入端到端状态机，形成 `Plan -> Select -> Execute -> Recover`（PSER）评测闭环，并输出可审计 trace。

## State Machine
1. `PLAN_OK / PLAN_FAIL`
- 输入 query 与模态信息。
- 空 query 直接失败，失败码：`E_NO_PLAN`。

2. `SELECT_OK / SELECT_EMPTY`
- 候选来源优先级：`pred_tools` -> `pred_tool` -> `candidates`。
- 若候选为空，按 unknown 预测执行 `reject/clarify` 回退，失败码：`E_NO_CANDIDATE`。

3. `EXECUTE_OK / EXECUTE_FAIL`
- Week08 默认执行器：`MockExecutor`。
- 可能失败码：`E_EXEC_TOOL_OFFLINE`、`E_EXEC_TIMEOUT`、`E_EXEC_RUNTIME`。

4. `RECOVER`
- 使用 `RecoverManager` 决策动作：`retry / reject / clarify / halt`。
- `retry` 时切换下一个候选工具并重试；超过重试上限则进入策略回退。

## Failure Codes
- `OK`: 成功执行。
- `E_NO_PLAN`: 规划阶段失败。
- `E_NO_CANDIDATE`: 选择阶段无可用候选。
- `E_EXEC_TOOL_OFFLINE`: 执行器检测工具不可用。
- `E_EXEC_TIMEOUT`: 执行超时。
- `E_EXEC_RUNTIME`: 执行时错误。
- `E_RECOVER_EXHAUSTED`: 兜底失败（保留码）。
- `E_POLICY_REJECT`: 恢复策略选择拒答。
- `E_POLICY_CLARIFY`: 恢复策略选择澄清。
- `E_HALT`: 恢复策略选择终止。

## Trace Schema (核心字段)
- `id`, `split`, `query`, `modality`, `ambiguity_type`, `mapping_type`, `tool_status`
- `state_path`: 状态迁移路径
- `attempts`: 每次执行尝试（tool / success / failure_code / latency）
- `recover_path`: 每次失败后的恢复决策路径
- `final_action`, `final_tool`, `failure_code`, `e2e_success`

## Week08 Default Config
- 配置文件：`configs/eval/e2e_v1.yaml`
- 输入 split：`data/splits/test.jsonl`
- 输入预测：Week07 `balanced` policy test 预测
- 复杂样本规则：
  - `mapping_type == one_to_many` 或
  - `tool_status in {offline, replaced, newly_added}` 或
  - `is_unknown_gold == true`
- 约束：`min_complex_samples = 50`

## Run Command
```bash
bash scripts/eval/eval_e2e.sh
```

## Output Artifacts
- trace: `outputs/predictions/<exp_id>_e2e_traces.jsonl`
- report: `outputs/reports/<exp_id>_e2e_eval.json`
- log: `outputs/logs/<exp_id>_e2e.log`
