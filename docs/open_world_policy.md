# Open World Policy (Week07)

## Goal
在 unknown 场景下降低误执行与幻觉，同时控制误拒答，支持策略切换：`strict / balanced / recall-first`。

## Components
1. unknown 阈值自动标定
- 文件：`src/uncertainty/calibration.py`
- 基于 dev 预测的 `unknown_prob` 与 `is_unknown_gold`，网格搜索阈值。
- Week07 使用约束标定：`objective=unknown_f1_constrained` + `max_false_reject_rate=0.20`。

2. 拒答/澄清策略
- 文件：`src/agent/policy.py`
- 核心动作：`execute / reject / clarify`
- 输入信号：`unknown_prob`、`confidence`、`retrieved_tools`、策略参数。

3. 策略切换配置
- 文件：`configs/eval/open_world_policy.yaml`
- `strict`: 更保守，降低阈值并启用“候选外拒答”。
- `balanced`: 标定阈值，不额外放宽或收紧。
- `recall-first`: 提高阈值，减少拒答。

4. 对比评测脚本
- 文件：`scripts/eval/compare_open_world.py`
- 输出 before/after 指标、误拒答/漏检统计、策略动作分布。

## Week07 Seed42 Results
输入预测：
- dev: `outputs/predictions/20260223_open_world_tool_agent_main_v1_mm_openworld_42_dev.jsonl`
- test: `outputs/predictions/20260223_open_world_tool_agent_main_v1_mm_openworld_42_test_eval.jsonl`

标定阈值（dev）：`0.15`

### strict
- dev: Unknown F1 `0.0000 -> 0.4810`, Hall `0.4333 -> 0.0000`, E2E `0.2500 -> 0.3167`, false_reject=41
- test: Unknown F1 `0.0000 -> 0.3333`, Hall `0.5333 -> 0.0000`, E2E `0.2667 -> 0.2000`, false_reject=48

### balanced
- dev: Unknown F1 `0.0000 -> 0.2105`, Hall `0.4333 -> 0.2667`, E2E `0.2500 -> 0.2333`
- test: Unknown F1 `0.0000 -> 0.3125`, Hall `0.5333 -> 0.3000`, E2E `0.2667 -> 0.3333`

### recall-first
- dev: Unknown F1 `0.0000 -> 0.0952`, Hall `0.4333 -> 0.4167`, E2E `0.2500 -> 0.2667`
- test: Unknown F1 `0.0000 -> 0.0000`, Hall `0.5333 -> 0.5167`, E2E `0.2667 -> 0.2667`

## Recommendation
- 默认策略：`balanced`
- 适用场景：综合权衡 unknown 检测、幻觉下降、E2E 成功率。
- 备选：`strict` 用于高安全约束场景（代价是高误拒答）。

## 3-Seed Check (base model)
汇总文件：`outputs/reports/open_world_compare_balanced_3seeds_summary.json`

- dev 平均变化：
  - Unknown F1: `0.0000 -> 0.2813`（+0.2813）
  - Hallucination: `0.4444 -> 0.2778`（-0.1667）
  - E2E: `0.2389 -> 0.2444`（+0.0056）

- test 平均变化：
  - Unknown F1: `0.0000 -> 0.2998`（+0.2998）
  - Hallucination: `0.5500 -> 0.3389`（-0.2111）
  - E2E: `0.2500 -> 0.2444`（-0.0056，近似持平）

## Run Commands
```bash
python scripts/eval/compare_open_world.py \
  --project-root /root/autodl-tmp/project \
  --dev-prediction outputs/predictions/20260223_open_world_tool_agent_main_v1_mm_openworld_42_dev.jsonl \
  --test-prediction outputs/predictions/20260223_open_world_tool_agent_main_v1_mm_openworld_42_test_eval.jsonl \
  --strategy balanced \
  --output-report outputs/reports/open_world_compare_balanced_seed42.json \
  --output-dev-policy-jsonl outputs/predictions/20260223_open_world_tool_agent_main_v1_mm_openworld_42_dev_policy_balanced.jsonl \
  --output-test-policy-jsonl outputs/predictions/20260223_open_world_tool_agent_main_v1_mm_openworld_42_test_policy_balanced.jsonl
```
