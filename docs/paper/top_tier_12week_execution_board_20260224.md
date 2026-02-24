# 冲顶 12 周执行看板（可直接执行版）

- 版本: v1.0
- 日期: 2026-02-24
- 适用场景: 后续设备升级后的连续推进
- 配套总报告: `project/docs/paper/top_tier_research_blueprint_20260224.md`

---

## 1. 使用说明

### 1.1 看板目的

把冲顶路线落成“按周可执行任务”，每周都有:

1. 本周目标
2. 关键任务
3. 产出物
4. DoD（完成定义）
5. 风险与回滚动作

### 1.2 执行规则

1. 每周只追 1-2 个核心里程碑
2. 每周必须冻结产出文件
3. 所有结论必须可追溯到 `outputs/reports/*`

---

## 2. 里程碑总览

1. M1（W1-W2）: benchmark_v2 协议与数据门禁冻结
2. M2（W3-W4）: 强基线公平对齐与重跑
3. M3（W5-W7）: main_v2 方法开发与消融
4. M4（W8-W9）: 真实执行器与鲁棒性闭环
5. M5（W10）: 5-seed + 显著性主结果冻结
6. M6（W11-W12）: 论文与投稿包冻结

---

## 3. 周计划看板

## Week 1: benchmark_v2 协议冻结

### 目标

冻结数据字段、标注规范、质量门禁标准。

### 任务

1. 固化 schema（JSON Schema + 文档）
2. 固化 unknown taxonomy 与 one-to-many 标注规则
3. 评审并冻结标注手册 v1

### 产出物

1. `docs/paper/benchmark_v2_schema_and_annotation_playbook_20260224.md`
2. `docs/paper/benchmark_v2_row_schema_v1.json`
3. `docs/paper/benchmark_v2_protocol.md`（建议新增）

### DoD

1. schema 可机器校验
2. 标注冲突场景有明确仲裁规则
3. 评测脚本字段与 schema 对齐

### 风险与回滚

1. 风险: 字段定义反复变动导致下游脚本返工
2. 回滚: 先冻结最小字段集合，扩展字段放 `metadata`

---

## Week 2: 数据构建与质量门禁脚本

### 目标

完成 v2 构建脚本与质量门禁脚本首版。

### 任务

1. 新建 `build_benchmark_v2.py`
2. 新建 `run_quality_gates_v2.py`
3. 增加泄漏检测、重复检测、分布检测

### 产出物

1. `scripts/data/build_benchmark_v2.py`
2. `scripts/data/run_quality_gates_v2.py`
3. `outputs/reports/benchmark_v2_quality_report.json`

### DoD

1. 可一键构建 train/dev/test
2. 质量门禁可自动 fail-on-error
3. 输出 manifest 与 split hash

### 风险与回滚

1. 风险: 数据分布不平衡
2. 回滚: 先保证 dev/test 分布平衡，再扩张 train

---

## Week 3: 强基线公平对齐（协议统一）

### 目标

让所有基线在同协议、同评测脚本下可比较。

### 任务

1. 统一候选集约束规则
2. 统一 unknown 判定与统计口径
3. 统一 latency/cost 记录口径

### 产出物

1. `docs/paper/baseline_fairness_protocol.md`
2. `outputs/reports/baseline_alignment_audit.json`

### DoD

1. 每个 baseline 均通过 fairness audit
2. 无“脚本口径差异导致的指标漂移”

### 风险与回滚

1. 风险: 历史脚本逻辑分叉过多
2. 回滚: 抽出统一 evaluator，老脚本仅做适配层

---

## Week 4: 强基线全量重跑

### 目标

输出“可投稿可比较”的 baseline 主表。

### 任务

1. 重跑 majority/text-only/mm baseline
2. 重跑 Qwen+Whisper 强基线
3. 输出 baseline summary table

### 产出物

1. `outputs/reports/baseline_main_table_v2.csv`
2. `outputs/reports/baseline_seed_summary_v2.json`

### DoD

1. 至少 3 seeds
2. 主切片（modality/status/known-unknown）齐全

### 风险与回滚

1. 风险: 强基线耗时过长
2. 回滚: 先固定 dev 小集合做 smoke，再上全量

---

## Week 5: main_v2 方法实现（第一阶段）

### 目标

实现候选约束选择 + 联合拒答训练。

### 任务

1. 新增 candidate-constrained selector
2. 新增 unknown 联合损失
3. 新增配置与训练入口

### 产出物

1. `src/agent/main_v2_model.py`
2. `scripts/train/run_main_v2.py`
3. `configs/train/main_v2.yaml`

### DoD

1. 训练与推理链路跑通
2. 与 main_v1 指标可直接对比

### 风险与回滚

1. 风险: 新损失不收敛
2. 回滚: 先做 warm-start + 分阶段解冻

---

## Week 6: main_v2 消融与阈值标定

### 目标

证明每个新增模块的独立贡献。

### 任务

1. 设计 main_v2 消融矩阵
2. known/unknown 分开评估
3. 标定策略稳定性评估

### 产出物

1. `outputs/reports/main_v2_ablation_report.json`
2. `outputs/reports/main_v2_ablation_table.csv`
3. `docs/paper/main_v2_ablation_notes.md`

### DoD

1. 至少 3 个关键消融项有可解释变化
2. unknown 改善不以 known 崩塌为代价

### 风险与回滚

1. 风险: 指标互相拉扯严重
2. 回滚: 用约束式目标优化 false_reject 上限

---

## Week 7: main_v2 与强基线首轮对决

### 目标

在统一协议下验证 main_v2 的竞争力。

### 任务

1. 与 strongest baseline 同设置对比
2. 输出 Pareto 曲线（known success vs unknown f1）
3. 做失败案例首轮分析

### 产出物

1. `outputs/reports/main_v2_vs_baselines_v2.json`
2. `outputs/figures/main_v2_pareto_curve.png`

### DoD

1. 至少在一个主目标上形成稳定优势
2. 优势在关键切片方向一致

### 风险与回滚

1. 风险: 绝对指标仍落后强基线
2. 回滚: 调整主 claim 为“安全-效能 Pareto 优势”

---

## Week 8: 真实执行器接入（RealExecutor v1）

### 目标

替换 mock 执行，拿到真实 E2E 反馈。

### 任务

1. 定义工具调用接口规范
2. 接入 real execution adapter
3. trace 增加真实错误上下文

### 产出物

1. `src/execution/real_executor.py`
2. `configs/eval/e2e_real_v1.yaml`
3. `outputs/reports/e2e_real_smoke_report.json`

### DoD

1. 可稳定执行并产出 trace
2. 错误码覆盖可审计

### 风险与回滚

1. 风险: 外部依赖不稳定
2. 回滚: 增加可控 sandbox 服务模拟真实接口

---

## Week 9: 鲁棒性场景重跑（Real + Inject）

### 目标

在真实执行链路下完成鲁棒性主实验。

### 任务

1. 重跑 offline/replaced/newly_added
2. 增加 alias collision / version incompatibility
3. 重做 recover 策略对比

### 产出物

1. `outputs/reports/main_v2_robustness_real_report.json`
2. `outputs/reports/main_v2_robustness_real_table.csv`

### DoD

1. 结论不依赖 mock 假设
2. 鲁棒性结论可复现

### 风险与回滚

1. 风险: 实验成本高
2. 回滚: 先跑 challenge subset，再扩全量

---

## Week 10: 5-seed 主实验与显著性

### 目标

冻结可投稿主结果。

### 任务

1. 跑 5 seeds 主实验
2. 计算 CI 与显著性检验
3. 冻结主表与附表

### 产出物

1. `outputs/reports/final_main_table_v2.csv`
2. `outputs/reports/final_significance_summary_v2.json`
3. `outputs/reports/final_seed_anomalies_v2.jsonl`

### DoD

1. 核心 claim 统计显著
2. 方向一致性满足投稿要求

### 风险与回滚

1. 风险: seed 波动大
2. 回滚: 加强数据与校准稳健性，必要时增 seed

---

## Week 11: 论文主文与图表定稿

### 目标

完成论文初稿与图表全套。

### 任务

1. 完成 Methods/Experiments/Analysis
2. 统一图表风格与编号
3. 完成附录与复现说明

### 产出物

1. `docs/paper/manuscript_v1.md`（或 tex）
2. `docs/paper/tables/*`
3. `docs/paper/figures/*`

### DoD

1. claim-evidence 一一对应
2. 所有图表可由脚本复现

### 风险与回滚

1. 风险: 叙事与实验不对齐
2. 回滚: 使用 claim matrix 逐条对账

---

## Week 12: 内审、rebuttal 预演、投稿包冻结

### 目标

形成可提交版本并完成风险兜底。

### 任务

1. 内部审稿与硬问题清单
2. rebuttal FAQ 预演
3. 冻结匿名投稿包

### 产出物

1. `docs/paper/rebuttal_faq_v1.md`
2. `docs/paper/submission_checklist_v1.md`
3. 匿名提交压缩包（本地）

### DoD

1. 投稿包完整、可复现、可答辩
2. 高风险问题均有预案

### 风险与回滚

1. 风险: 临近投稿发现主结论薄弱
2. 回滚: 缩主 claim，突出最稳健贡献

---

## 4. 每周固定例会模板（建议）

每周固定回答 5 个问题:

1. 本周最关键目标是否达成
2. 哪个假设被支持/被证伪
3. 哪个风险上升了
4. 下周必须冻结什么工件
5. 当前是否偏离投稿主线

---

## 5. 资源与预算模板（执行时填写）

每周记录:

1. GPU 小时
2. 训练成本
3. 失败重跑次数
4. 有效实验占比

目标是逐周提高“有效实验占比”。

---

## 6. 快速状态标记（建议）

每周在看板中标记:

1. `GREEN`: 按计划推进
2. `YELLOW`: 进度可追但风险升高
3. `RED`: 主线受阻需立刻裁剪

---

## 7. 下一步建议（你现在就能做）

1. 先把 Week1/Week2 的 DoD 具体化为脚本 TODO
2. 把本文件复制成 `execution_board_live.md` 每周更新
3. 每周冻结一次 artifacts，避免后续“结果不可追溯”

