# Qwen2.5-VL + Whisper 交接文档（2026-02-24）

## 1. 文档目的

这份文档用于在新对话中快速接续当前工作，覆盖：

- 已完成工作（代码、环境、模型、评测）
- 已定位并修复的 bug（症状、排查方法、修复动作）
- 当前结果质量与风险点
- 未来目标与建议执行顺序
- 一键复现/继续推进的命令清单

---

## 2. 当前状态快照（必须先看）

### 2.1 工作目录与关键路径

- 项目根目录：`/root/autodl-tmp/project`
- 评测配置：`/root/autodl-tmp/project/configs/eval/qwen25_vl_whisper.yaml`
- 评测脚本：
  - `scripts/eval/eval_qwen25_vl_whisper.sh`
  - `scripts/eval/eval_qwen25_vl_whisper.py`
- 模型实现文件（本轮主要改动）：`src/agent/qwen_vl_whisper.py`

### 2.2 模型文件路径（当前有效）

- `models/Qwen2.5-VL-7B-Instruct` -> `/root/autodl-tmp/project/.cache/modelscope_clean/Qwen/Qwen2___5-VL-7B-Instruct`
- `models/whisper-large-v3` -> `/root/autodl-tmp/project/.cache/modelscope_clean/AI-ModelScope/whisper-large-v3`

说明：当前模型软链接已明确指向 `modelscope_clean`，不是旧缓存。

### 2.3 环境版本（当前有效）

- `torch 2.5.1+cu121`
- `transformers 5.3.0.dev0`
- `accelerate 1.12.0`
- `qwen_vl_utils` 已安装

已观测环境告警：

- `libgomp: Invalid value for environment variable OMP_NUM_THREADS`
- 当前值：`OMP_NUM_THREADS=0`（建议改为正整数，如 8）

---

## 3. 本轮已经完成的工作

### 3.1 工程与资产核验

- 确认路径和目录结构正常（`check_paths.py`）
- 确认模型资产完整（`check_qwen_whisper_assets.py` 返回 `PASS`）
- 确认 `dev/test` split 路径正确：
  - `data/splits/dev.jsonl`
  - `data/splits/test.jsonl`

### 3.2 代码改造（核心）

已在 `src/agent/qwen_vl_whisper.py` 做关键改造，解决运行正确性问题并提升实用性：

1. 对齐 Qwen 官方推理流程
   - 新增 `qwen_vl_utils.process_vision_info` 路径
   - 统一生成与 decode 逻辑（按输入长度裁剪生成 token）
   - 关键入口：`src/agent/qwen_vl_whisper.py:450`

2. 修复多卡自动切分导致乱码
   - 将 `device_map="auto"` 改为基于配置设备强制单卡
   - 代码位置：`src/agent/qwen_vl_whisper.py:272`、`src/agent/qwen_vl_whisper.py:275`

3. 强化 prompt 约束
   - 强调“有候选时优先选候选，只有不适用才 unknown”
   - 代码位置：`src/agent/qwen_vl_whisper.py:390`

4. 增加退化输出回退策略
   - 当模型输出 `pred_tool=__unknown__` 且信号退化（如 `confidence` 极低）时，回退到候选工具
   - 回退函数：`src/agent/qwen_vl_whisper.py:538`
   - 触发逻辑：`src/agent/qwen_vl_whisper.py:630`

5. 视频/图片输入路径整理
   - 视频首帧抽取为临时文件，再转 `file://` URI 供官方流程消费
   - 增加临时文件清理，避免泄露

### 3.3 评测执行结果（最新全量）

#### Dev 全量（60条）

- 报告：`outputs/reports/20260223_open_world_tool_agent_qwen25_vl_7b_whisper_42_dev_qwen_eval.json`
- 预测：`outputs/predictions/20260223_open_world_tool_agent_qwen25_vl_7b_whisper_42_dev_qwen_eval.jsonl`
- 日志：`outputs/logs/20260223_open_world_tool_agent_qwen25_vl_7b_whisper_42_dev_qwen_eval.log`

指标：

- `tool_selection_accuracy`: `0.9512195121951219`
- `unknown_detection_f1`: `0.1`
- `hallucination_rate`: `0.03333333333333333`
- `end_to_end_success_rate`: `0.6666666666666666`
- `avg_latency_ms`: `728.4116258844733`

预测分布（dev）：

- `search_web`: 18
- `summarize_text`: 10
- `transcribe_audio`: 12
- `qa_text`: 12
- `ocr_image`: 7
- `__unknown__`: 1

#### Test 全量（60条）

- 报告：`outputs/reports/20260223_open_world_tool_agent_qwen25_vl_7b_whisper_42_test_qwen_eval.json`
- 预测：`outputs/predictions/20260223_open_world_tool_agent_qwen25_vl_7b_whisper_42_test_qwen_eval.jsonl`
- 日志：`outputs/logs/20260223_open_world_tool_agent_qwen25_vl_7b_whisper_42_test_qwen_eval.log`

指标：

- `tool_selection_accuracy`: `0.9583333333333334`
- `unknown_detection_f1`: `0.0`
- `hallucination_rate`: `0.03333333333333333`
- `end_to_end_success_rate`: `0.7666666666666667`
- `avg_latency_ms`: `696.4518600453933`

预测分布（test）：

- `search_web`: 22
- `qa_text`: 14
- `ocr_image`: 9
- `transcribe_audio`: 9
- `summarize_text`: 6
- `__unknown__`: 0

---

## 4. 已解决 bug 详单（含排查路径）

## Bug-1：依赖缺失导致评测“看似成功，实则全回退”

### 现象

- 早期预测几乎全是 `__unknown__`
- `raw_model_response` 中出现 `No module named 'transformers'`
- 延迟异常低（毫秒级）

### 排查动作

- 检查预测样本 `raw_model_response`
- 直接执行 Python import 检查 `torch/transformers/accelerate/datasets`

### 根因

- 运行环境缺失 `transformers` 等关键依赖，进入 fallback 路径

### 修复

- 安装并对齐推理依赖（后续还经历了版本迭代，见 Bug-2）

---

## Bug-2：版本不兼容（transformers 与 torch）

### 现象

- 出现提示：`PyTorch >= 2.4 is required but found 2.1.2`

### 排查动作

- 直接读取版本输出，复现错误提示

### 根因

- `transformers` 新版本与旧 `torch` 组合不兼容

### 修复

- 最终对齐到：
  - `torch 2.5.1+cu121`
  - `transformers 5.3.0.dev0`
  - `qwen_vl_utils` 已安装

---

## Bug-3：官方流程未启用（qwen_vl_utils 缺失）

### 现象

- 代码虽改了官方路径，但环境里无 `qwen_vl_utils`

### 排查动作

- `python -c "import qwen_vl_utils"` 检查失败

### 根因

- 缺少 `qwen-vl-utils` 依赖

### 修复

- 安装 `qwen-vl-utils`，官方流程可被调用

---

## Bug-4：ModelScope 重新下载后仍乱码输出

### 现象

- 已重下模型，但生成仍出现乱码（符号/杂字符）
- 无法稳定产出 JSON

### 排查动作

1. 验证软链接已指向新缓存：
   - `models/Qwen2.5-VL-7B-Instruct -> .cache/modelscope_clean/...`
2. 使用“脱离项目代码”的最小脚本直接调用 Qwen 生成，仍乱码
3. 对照实验：
   - `device_map="auto"`：乱码
   - `device_map={'':0}`：可输出正常 JSON

### 根因

- 多 GPU 自动切分（`device_map="auto"`）在当前环境下导致生成异常

### 修复

- 在代码中强制单卡映射（按 `runtime.device`）
- 改动位置：`src/agent/qwen_vl_whisper.py:272-275`

---

## Bug-5：不乱码后仍“全 unknown”

### 现象

- 模型开始输出结构化 JSON，但常见：
  - `pred_tool="__unknown__"`
  - `unknown_prob≈0.9`
  - `confidence=0.0`

### 排查动作

- 抽样读取预测文件逐条检查 `raw_model_response`

### 根因

- 模型策略偏保守，且阈值机制将结果持续压回 unknown

### 修复

1. Prompt 加强“优先从候选中选工具”
2. 增加候选回退策略（回退到候选工具）
3. 对退化 unknown 输出做校正，防止阈值再次压回 unknown

结果：工具选择准确率显著提升到 0.95+（dev/test）

---

## 5. 当前需要改进的点（未完成）

### 5.1 Unknown 检测性能偏低

- 当前策略偏“工具优先”，导致 `unknown_detection_f1` 偏低（dev=0.1，test=0.0）
- 需要在“工具选择准确率”与“unknown 检测”之间做策略平衡

建议：

- 增加配置开关（例如 `strict_unknown`）切换策略
- 对 `unknown_threshold` 做校准扫描（如 0.45/0.5/0.55/0.6）
- 在 dev 上选 Pareto 点再固定到 test

### 5.2 线程环境变量异常

- `OMP_NUM_THREADS=0` 触发 `libgomp` 告警
- 建议固定为合理正整数（如 `8`）

### 5.3 运行效率优化

- 当前平均延迟 ~700ms/样本（全量场景）
- 可评估：
  - `use_flash_attention_2`（与环境兼容后再开）
  - batch 化策略
  - tokenizer/processor fast 配置一致性

---

## 6. 下一阶段目标（建议优先级）

## P0（下一次对话先做）

1. 增加 `strict_unknown` / `tool_first` 策略开关（配置化）
2. 在 `dev` 上跑阈值扫描，产出：
   - `tool_selection_accuracy`
   - `unknown_detection_f1`
   - `hallucination_rate`
3. 选定策略后在 `test` 固化评测

## P1（短期）

1. 对 fallback 规则做更细粒度（可按 modality + ambiguity_type）
2. 输出策略对比报告（建议新增 `outputs/reports/qwen_policy_compare_*.json`）

## P2（中期）

1. 加入回归测试（至少 smoke 测试）防止再次出现：
   - 乱码输出回归
   - 全 unknown 回归

---

## 7. 快速启动（新对话直接复制）

工作目录：

```bash
cd /root/autodl-tmp/project
```

### 7.1 环境快速自检

```bash
python - <<'PY'
import torch, transformers, accelerate
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("accelerate", accelerate.__version__)
try:
    import qwen_vl_utils
    print("qwen_vl_utils", getattr(qwen_vl_utils, "__version__", "installed"))
except Exception as e:
    print("qwen_vl_utils", repr(e))
PY
```

### 7.2 跑全量评测（当前可用）

```bash
bash scripts/eval/eval_qwen25_vl_whisper.sh configs/eval/qwen25_vl_whisper.yaml --split-name dev
bash scripts/eval/eval_qwen25_vl_whisper.sh configs/eval/qwen25_vl_whisper.yaml --split-name test
```

### 7.3 看最终指标

```bash
cat outputs/reports/20260223_open_world_tool_agent_qwen25_vl_7b_whisper_42_dev_qwen_eval.json
cat outputs/reports/20260223_open_world_tool_agent_qwen25_vl_7b_whisper_42_test_qwen_eval.json
```

### 7.4 看预测分布

```bash
python - <<'PY'
import json
from collections import Counter
for split in ["dev", "test"]:
    p=f"outputs/predictions/20260223_open_world_tool_agent_qwen25_vl_7b_whisper_42_{split}_qwen_eval.jsonl"
    c=Counter()
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c[json.loads(line)["pred_tool"]] += 1
    print(split, dict(c))
PY
```

---

## 8. 本轮关键文件索引

- 交接文档（本文件）：`docs/qwen25_vl_whisper_handover.md`
- 核心代码：`src/agent/qwen_vl_whisper.py`
- 评测脚本：`scripts/eval/eval_qwen25_vl_whisper.py`
- Shell 入口：`scripts/eval/eval_qwen25_vl_whisper.sh`
- 配置：`configs/eval/qwen25_vl_whisper.yaml`
- 最新 dev 报告：`outputs/reports/20260223_open_world_tool_agent_qwen25_vl_7b_whisper_42_dev_qwen_eval.json`
- 最新 test 报告：`outputs/reports/20260223_open_world_tool_agent_qwen25_vl_7b_whisper_42_test_qwen_eval.json`

---

## 9. 额外说明

- 当前版本为了避免“全 unknown”，引入了较强的候选回退策略，所以 unknown 相关指标会被压低。
- 若后续任务更关注 open-world reject 能力，应优先做 unknown 策略回调与阈值校准。
- 若后续任务更关注工具选择准确率，当前策略已可直接使用并复现。

