# Week 05 Log

## 周目标
完成主方法 v1 编码，实现多模态编码、工具选择、unknown 检测、检索增强、前向与loss、推理输出，并跑通小样本 train/eval。

## D1-D7 任务清单
- D1: [x] 实现多模态编码与特征投影模块接口
- D2: [x] 实现工具选择头（支持 one-to-many 目标）
- D3: [x] 实现 unknown 检测头（logit + energy 融合）
- D4: [x] 实现检索增强接口（top-k 候选拼接）
- D5: [x] 打通前向、loss、推理与 JSON 格式化输出
- D6: [x] 写最小集成测试（shape、数值稳定、输出格式）
- D7: [x] 输出周报和 Week6 输入清单

## 本周新增文件
- `configs/train/main_v1.yaml`
- `scripts/train/run_main_v1.py`
- `scripts/train/run_main_v1.sh`
- `scripts/eval/eval_main_v1.py`
- `scripts/eval/eval_main_v1.sh`
- `src/agent/text_features.py`
- `src/agent/main_v1_model.py`
- `src/agent/main_v1_data.py`
- `src/agent/main_v1_eval.py`
- `src/retriever/base.py`
- `src/retriever/simple_retriever.py`
- `src/retriever/__init__.py`
- `src/uncertainty/scoring.py`
- `src/uncertainty/__init__.py`
- `tests/test_main_v1_smoke.py`

## 验证结果
- 单元 smoke test: `python -m unittest tests/test_main_v1_smoke.py` -> `OK (3 tests)`
- 训练脚本: `bash scripts/train/run_main_v1.sh` -> 成功生成 checkpoint/predictions/report
- 评测脚本: `bash scripts/eval/eval_main_v1.sh --split-name test` -> 成功输出 test 评测报告

## DoD 对齐
1. 模型可在小样本上完整跑通 train/eval: 已完成
2. 输出 JSON 格式合法率 100%: 已完成（`json_valid_rate=1.0`）

## Week6 输入清单
1. `outputs/checkpoints/20260223_open_world_tool_agent_main_v1_mm_openworld_42.pt`
2. `outputs/reports/20260223_open_world_tool_agent_main_v1_mm_openworld_42.json`
3. `outputs/reports/20260223_open_world_tool_agent_main_v1_mm_openworld_42_test_eval.json`
4. `outputs/predictions/20260223_open_world_tool_agent_main_v1_mm_openworld_42_dev.jsonl`
5. `outputs/predictions/20260223_open_world_tool_agent_main_v1_mm_openworld_42_test_eval.jsonl`

## 风险与补救记录
- 风险: 当前 unknown F1 仍为 0.0，unknown 头需要 Week6 错误分析后再调阈值/损失权重。
- 补救: Week6 优先做 unknown 样本分布检查和阈值扫描，不改大结构。
