# Open-World Multimodal Tool Agent 冲顶技术报告（长期研究版）

- 版本: v1.0
- 日期: 2026-02-24
- 面向对象: 当前项目负责人（后续设备升级后继续推进）
- 项目根目录: `/root/autodl-tmp/project`

---

## 0. 报告目的与结论先行

### 0.1 这份报告解决什么问题

本报告用于把当前项目从“工程完整的研究原型”推进到“可冲顶会/顶刊的研究项目”，并提供在未来算力充足时可直接执行的技术与实验路线。

### 0.2 当前状态一句话结论

当前项目具备完整 pipeline、可复现实验脚手架和多周实验资产，但距离顶会/顶刊仍有关键差距，主要集中在:

1. 数据真实性与规模（主结论仍建立在小规模合成模板上）
2. 方法创新深度（目前更偏工程集成）
3. 实验证据强度（真实执行器、强基线公平对比、统计显著性）
4. 论文资产完整性（正式论文叙事与实验主表尚未闭环）

### 0.3 冲顶总策略（核心）

1. 把研究主轴从“搭 pipeline”升级为“提出并验证新方法 + 新评测协议”
2. 把主实验从 `benchmark_v1(120/60/60)` 迁移到真实规模数据与真实执行反馈
3. 把“现象描述”升级为“可证伪的研究假设 + 严格统计验证”
4. 把“单点结果”升级为“强基线下、跨场景、跨seed、可复现的稳定优势”

---

## 1. 当前项目技术盘点（基于仓库资产）

### 1.1 已有资产（优势）

1. 端到端研究骨架完整:
   `data -> train -> eval -> policy -> e2e -> ablation -> robustness -> aggregation`
2. 实验可追溯:
   结果文件命名、日志、report、trace 基本齐全
3. 评测维度清晰:
   已覆盖 `tool_selection / hallucination / unknown_f1 / e2e / latency`
4. 3-seed 聚合流程已打通:
   有稳定性聚合脚本与主表输出
5. 强基线已接入:
   `Qwen2.5-VL + Whisper` 已完成运行和评测输出

### 1.2 当前主结果概况（风险）

1. 主 benchmark 规模仍小:
   `train=120, dev=60, test=60`
2. main_v1 在 test 上关键指标偏低:
   `tool_selection_accuracy≈0.33`, `unknown_f1=0`, `e2e≈0.27`
3. 策略后 unknown_f1 虽提升，但 tool accuracy 与 e2e 有 trade-off
4. 强基线（Qwen+Whisper）在 tool/e2e 上明显更高，当前“主方法优于强基线”的论证不成立
5. E2E 使用 `MockExecutor`，鲁棒性结论存在外部有效性风险

### 1.3 数据资产现状（关键矛盾）

1. 已构建真实多模态池（25万+规模），这是重大资产
2. 但主实验使用的是 `benchmark_v1` 小切分
3. 真实池的 `gold_tool` 目前依赖脚本规则分配，不是人工标注任务语义
4. 换言之:
   “有大数据”与“主结论依赖小规模规则化数据”并存，是当前最大瓶颈

---

## 2. 顶会/顶刊评审视角下的差距模型

本节给出严格审稿视角，用于后续每周自检。

### 2.1 新颖性差距（Novelty Gap）

当前方法核心由以下模块构成:

1. 哈希 BoW 特征 + 轻量融合 MLP
2. unknown head + 温度/能量加权
3. 策略层 reject/clarify
4. PSER 状态机与恢复策略

问题在于:

1. 每个模块单独看都偏经典
2. 主要贡献像“工程集成与流程化”而非“新方法”
3. 对顶会来说，方法学贡献点需要更明确、更可归因

### 2.2 证据强度差距（Evidence Gap）

1. 主要结论未在真实执行环境中验证
2. 数据规模与自然性不足以支撑强泛化结论
3. 与强基线比较中缺少“同设置公平对齐”的系统证明
4. 统计显著性分析还不充分（当前以均值/std/CI 为主）

### 2.3 可迁移性差距（External Validity Gap）

1. 当前 query 中模板痕迹明显，现实复杂语义覆盖不足
2. 扰动注入中存在“规则改写标签”的路径，可能高估或低估真实鲁棒性
3. 真实工具行为噪声（超时、参数不兼容、别名冲突、版本漂移）未充分建模

### 2.4 论文资产差距（Paper Asset Gap）

1. `docs/paper/` 尚未形成论文主文稿资产
2. 缺少可直接投稿的图表矩阵与主要可视化
3. 缺少“方法-假设-实验-结论”的闭环 narrative

---

## 3. 冲顶研究定位重构（必须先定“要证明什么”）

### 3.1 建议的论文主命题（建议）

> 在开放世界多模态工具调用中，提出一种“候选约束 + 校准拒答 + 恢复学习”联合框架，在保证已知任务成功率的同时显著降低 unknown 误执行与幻觉，并在真实扰动下维持鲁棒性。

### 3.2 可证伪研究假设（顶会友好）

1. H1:
   相比强基线，联合框架能在 unknown 指标提升的同时维持或提升 known 任务成功率
2. H2:
   候选约束解码可显著降低 hallucination，不显著伤害正确选择率
3. H3:
   联合校准目标（unknown_f1 + false_reject 约束）优于单纯阈值策略
4. H4:
   恢复策略学习在真实执行失败场景下优于静态规则恢复
5. H5:
   方法在跨模态、跨工具状态、跨 seed 下优势方向一致且统计显著

### 3.3 论文贡献应拆成三层

1. 方法贡献:
   提出新的联合优化/决策机制（不是只调阈值）
2. 评测贡献:
   提出更贴近开放世界的评测协议与失败分类
3. 实证贡献:
   在强基线下给出全面而稳定的改进证据

---

## 4. 数据与Benchmark升级方案（从 v1 到 v2）

本节是最关键落地项。没有高质量数据，就很难冲顶。

### 4.1 总目标

构建 `benchmark_v2`，满足:

1. 规模:
   建议至少 `train 50k+`, `dev 5k+`, `test 5k+`（可按算力分层）
2. 真实性:
   query 来源于真实数据语义，不使用模板 query 作为主评估
3. 标注可信:
   gold_tool 来自明确规则与人工抽检，而非纯随机加权分配
4. unknown 构造可解释:
   unknown 样本区分“真未知任务”与“工具不可用/版本失配”

### 4.2 数据构建分层（建议采用三层样本）

1. L1: 自动规则样本（大规模）
   用于预训练与粗评估
2. L2: 人工复核样本（中规模）
   用于 dev/ablation/阈值标定
3. L3: 专家高难样本（小规模）
   用于论文主表挑战集

### 4.3 标注协议（必须文档化）

为每条样本至少标注:

1. `gold_tools`（允许 one-to-many）
2. `is_unknown_gold`
3. `unknown_reason_type`
4. `tool_status`
5. `ambiguity_type`
6. `required_constraints`（是否缺关键信息）

并记录:

1. 标注员 ID（匿名即可）
2. 初标-复核差异
3. 仲裁规则
4. 互标一致性（Cohen’s kappa 或 Fleiss’ kappa）

### 4.4 数据质量门禁（Data Gates v2）

每次发布 split 前必须通过:

1. Schema 完整性门禁
2. 重复样本与近重复样本检查
3. train-dev-test 泄漏检查（文本、媒体、ID、多模态指纹）
4. 标签一致性与互标一致性阈值
5. known/unknown、模态、状态、歧义类型分布平衡检查

### 4.5 你现有仓库的最小改造建议

1. 新增 `configs/data/benchmark_v2.yaml`
2. 新增 `scripts/data/build_benchmark_v2.py`
3. 新增 `scripts/data/run_quality_gates_v2.py`
4. 在 `outputs/reports/` 输出 v2 的标注一致性报告

---

## 5. 方法升级路线（从 main_v1 到 main_v2/main_v3）

### 5.1 现有 main_v1 的定位

`main_v1` 应定位为“可复现起点模型”，不是论文最终模型。

### 5.2 main_v2 目标（可投稿级）

在现有框架上引入以下可发表改进:

1. 候选约束排序头（Candidate-Constrained Ranking）
   让模型显式学习 `P(tool | query, candidate_set)`，避免候选外幻觉
2. 联合拒答学习（Joint Abstention Learning）
   用多目标损失联合优化:
   `known正确率 + unknown召回 + false_reject约束`
3. 覆盖率感知检索（Coverage-Aware Retrieval）
   估计候选覆盖置信度，并将其注入决策层
4. 恢复策略学习（Learned Recovery Policy）
   用 trace 学习何时 retry/reject/clarify，而非纯规则阈值

### 5.3 main_v3 目标（冲顶增强版）

1. 多任务训练:
   选择、拒答、失败码预测、恢复动作联合训练
2. 不确定性建模升级:
   温度缩放 + conformal / selective prediction 机制
3. 因果或对抗式鲁棒训练:
   针对 offline/replaced/newly-added 的结构化扰动训练

### 5.4 建议损失函数（可直接工程化）

总损失:

`L = L_select + λ1 * L_unknown + λ2 * L_calibration + λ3 * L_recovery`

其中:

1. `L_select`: 候选约束下的排序/分类损失
2. `L_unknown`: unknown 二分类或 focal loss
3. `L_calibration`: ECE surrogate / Brier / temperature regularization
4. `L_recovery`: 恢复动作监督损失

---

## 6. 评测协议升级（顶会可辩护版本）

### 6.1 指标体系升级（分主指标与安全指标）

主指标:

1. Known Tool Success Rate
2. Unknown Detection F1
3. End-to-End Success Rate

安全与可靠性指标:

1. Hallucination Rate
2. False Reject Rate
3. Unknown Miss Rate
4. ECE / Brier
5. Recover Success Rate
6. Tail Latency（P95/P99）

### 6.2 切片报表必须固定

按以下维度必须出全表:

1. 模态: text/image/audio/video
2. 映射: one-to-one / one-to-many
3. 工具状态: stable/offline/replaced/newly_added
4. 歧义类型: lexical/missing_constraints/underspecified/multimodal_conflict/version_drift
5. known / unknown 两大子集

### 6.3 统计显著性要求（建议）

1. 主结论至少 `5 seeds`
2. 报告 mean/std/95% CI
3. 对主要比较做 paired bootstrap test 或 McNemar test
4. 在主表标注显著性符号与 p-value 区间

### 6.4 强基线公平对齐（必须）

至少保证:

1. 同一数据切分
2. 同一候选集合约束
3. 同一 unknown 定义
4. 同一评测脚本与同一统计方式
5. 同一成本/延迟记录方式

---

## 7. E2E与鲁棒性评测重构（从 Mock 到 Real）

### 7.1 当前问题

`MockExecutor` 的好处是快，但无法代表真实系统噪声与失败分布。

### 7.2 真实执行器引入路线

1. RealExecutor v1:
   接真实工具 API（可先接本地模拟服务）
2. Failure Injection v1:
   注入超时、参数不兼容、权限错误、版本变更、结果为空
3. Trace Schema v2:
   增加请求参数摘要、响应码、错误堆栈类别、重试上下文

### 7.3 鲁棒性场景定义升级

1. Offline:
   工具完全不可用
2. Replaced:
   工具接口变更，旧参数不兼容
3. Newly Added:
   新工具加入候选，要求系统识别并使用
4. Alias Collision:
   工具别名冲突导致检索混淆
5. Version Drift:
   同名工具不同版本行为差异

### 7.4 报告规范

每个场景必须报告:

1. 成功率变化
2. 恢复路径分布
3. 平均重试次数
4. 失败码覆盖率
5. 成本与延迟增量

---

## 8. 算力升级后的训练与实验计划

### 8.1 算力分层策略

#### Tier-A（中等算力）

1. 2x~4x 消费级/数据中心 GPU
2. 用于 main_v2 原型与中规模实验

#### Tier-B（高算力）

1. 8x GPU（如 H800/A100 级）
2. 用于 main_v3、大规模种子实验与超参扫描

### 8.2 训练预算建议

1. 先固定数据版本，再扫模型
2. 先固定模型，再扫策略阈值
3. 全流程预算分配:
   `60% 训练 + 25% 评测 + 15% 消融鲁棒性`

### 8.3 实验矩阵（建议）

阶段一（可行性）:

1. 3 个模型变体
2. 2 个数据规模
3. 3 seeds

阶段二（论文主表）:

1. 2 个最终模型
2. 全切片评测
3. 5 seeds
4. 显著性测试

阶段三（补充材料）:

1. 鲁棒性全场景
2. 失败案例剖析
3. 成本-性能曲线

---

## 9. 工程与MLOps要求（顶会级复现保障）

### 9.1 版本冻结

每次主结果必须冻结:

1. 数据 manifest
2. 代码 commit hash
3. 环境 lock 文件
4. 模型权重 checksum
5. 配置 hash

### 9.2 实验追踪

建议接入:

1. MLflow 或 Weights & Biases（选其一）
2. 统一记录 config、metrics、artifacts、stderr/stdout

### 9.3 自动化回归

新增测试层级:

1. Unit tests（现有 smoke 保留）
2. Metric consistency tests
3. Data leakage tests
4. End-to-end regression tests（固定小样本）

---

## 10. 12周冲顶执行路线图（设备升级后）

### Week 1-2: 数据与问题重定义

1. 冻结 `benchmark_v2` 协议
2. 完成标注规范与样本抽检流程
3. 输出数据质量报告 v2

里程碑:

1. v2 数据可复现构建完成
2. known/unknown 与切片分布合理

### Week 3-4: 强基线统一重跑

1. 统一评测协议下重跑所有基线
2. 完成公平对齐脚本
3. 产出 baseline 主表

里程碑:

1. 基线结果稳定可复现
2. 可确认主方法目标差距

### Week 5-7: main_v2 方法实现与消融

1. 实现候选约束排序 + 联合拒答学习
2. 做模块级消融
3. 进行阈值/校准策略搜索

里程碑:

1. 至少在 2 个主指标上稳定超过当前 main_v1
2. 不出现灾难性 false reject 增长

### Week 8-9: 真实执行器与鲁棒性实验

1. 接入 RealExecutor v1
2. 重做 offline/replaced/newly_added/alias/version_drift
3. 完成 recover 学习策略对比

里程碑:

1. 真实执行器场景下结论方向与 mock 不冲突
2. 异常样本可解释

### Week 10: 5-seed 主实验与显著性

1. 跑满 5 seeds
2. 计算 CI 与显著性
3. 形成最终主表与附表

里程碑:

1. 核心 claim 统计显著
2. 方向一致性满足投稿要求

### Week 11: 论文写作与图表定稿

1. 主文稿初版
2. 图表（主表、雷达图、可靠性图、失败案例图）
3. 补充材料与复现实验脚本

里程碑:

1. 全文自洽
2. 可独立复现

### Week 12: 内部审稿与投稿包

1. 模拟 rebuttal
2. 修复论证薄弱点
3. 准备匿名包与开源包

里程碑:

1. 投稿版本冻结
2. 风险清单闭环

---

## 11. 论文结构模板（可直接套用）

### 11.1 主文结构建议

1. Introduction
2. Problem Setup and Open-World Formalization
3. Method（main_v2/main_v3）
4. Benchmark v2 and Protocol
5. Main Results vs Strong Baselines
6. Ablation and Robustness
7. Analysis（Calibration, Failure Taxonomy, Cost-Latency）
8. Related Work
9. Limitations and Ethical Considerations
10. Conclusion

### 11.2 图表清单（最低配置）

1. 主结果总表（含显著性）
2. known/unknown 双轴权衡图
3. 校准曲线（Reliability Diagram）
4. 鲁棒性场景柱状图
5. 恢复路径 Sankey/状态转移图
6. 失败案例可视化（按歧义类型）

---

## 12. 风险清单与规避策略

### 12.1 高风险项

1. 强基线无法超越
2. unknown 提升导致 known 任务显著下降
3. 真实执行器结果与 mock 结论冲突
4. 数据标注一致性不足

### 12.2 应对策略

1. 将主 claim 从“全面超越”改为“安全-效能 Pareto 优势”
2. 主表强制加入 false_reject 与 unknown_miss 双约束
3. 提前在 Week8 做 mock-real 一致性对照
4. 建立标注仲裁与一致性门槛

---

## 13. 现阶段（低算力）你现在就能做的准备工作

以下工作不依赖大规模 GPU，可立刻推进，显著降低未来返工成本。

### 13.1 数据与协议准备

1. 完成 `benchmark_v2` schema 与标注手册
2. 增加数据门禁脚本（泄漏、分布、重复、一致性）
3. 预先构建抽样审查工具（导出人工复核任务）

### 13.2 代码结构准备

1. 将 `main_v1` 模块化拆成:
   `retrieval`, `selector`, `abstention`, `recovery`
2. 统一 metrics API，避免不同脚本重复实现
3. 把配置升级为“单主配置 + override”

### 13.3 实验工程准备

1. 建立 run registry（记录每次 run 的 hash 与来源）
2. 建立一键复现实验命令集
3. 建立论文图表自动导出脚本

### 13.4 写作资产准备

1. 先写 Methods 与 Protocol 两章草稿
2. 预置表格模板与图标题模板
3. 维护“claim-evidence 对照表”

---

## 14. 针对你当前仓库的具体实施建议（可执行）

### 14.1 建议新增目录

1. `docs/paper/figures/`
2. `docs/paper/tables/`
3. `configs/data/benchmark_v2.yaml`
4. `scripts/data/build_benchmark_v2.py`
5. `scripts/eval/eval_main_v2.py`
6. `scripts/report/export_paper_tables.py`

### 14.2 建议新增报告文件

1. `docs/paper/claim_matrix.md`
2. `docs/paper/benchmark_v2_protocol.md`
3. `docs/paper/reproducibility_checklist.md`
4. `docs/paper/limitations_and_ethics.md`

### 14.3 建议保留并升级现有资产

1. 保留 `week06-10` 资产作为“方法演进历史”
2. 把 `final_main_table.csv` 升级为 v2 主表模板
3. 把异常样本分析脚本升级为论文分析脚本

---

## 15. 冲顶就绪度评分卡（建议每两周打分）

总分 100，建议 85+ 才进入正式投稿阶段。

1. 问题与贡献清晰度: 15
2. 数据与标注可信度: 20
3. 方法创新强度: 20
4. 主实验与统计强度: 20
5. 鲁棒性与可解释分析: 10
6. 可复现性与工程质量: 10
7. 写作完成度: 5

你当前保守估计:

1. 优势项:
   工程质量、流程完整、可追溯性
2. 短板项:
   数据真实性、方法新颖性、强基线对比胜率

---

## 16. 最终执行建议（给未来“设备充足版本”的你）

### 16.1 第一优先级（必须）

1. 先做 `benchmark_v2`，再谈模型冲顶
2. 先统一强基线公平对比，再谈方法优势
3. 先接真实执行器，再写鲁棒性主结论

### 16.2 第二优先级（高收益）

1. 推出 main_v2（联合优化）
2. 做 5-seed + 显著性
3. 做 known/unknown Pareto 曲线

### 16.3 第三优先级（投稿加分）

1. 失败案例可解释性图谱
2. 成本-性能折中分析
3. 提供可复现实验与匿名开源包

---

## 附录A: 未来大算力阶段的建议执行顺序（命令导向）

> 以下是建议顺序，不是当前必须执行。

1. 数据构建与质检:
   - `bash scripts/data/run_real_data_pipeline.sh configs/data/real_public_sources.yaml`
   - `python scripts/data/run_quality_gates.py --project-root /root/autodl-tmp/project --fail-on-error`
2. 训练矩阵:
   - `bash scripts/train/run_real_train_matrix.sh configs/train/main_v1_h800_base.yaml`
3. 主评测:
   - `bash scripts/eval/run_real_eval_matrix.sh`
4. 策略与E2E:
   - `python scripts/eval/compare_open_world.py ...`
   - `bash scripts/eval/eval_e2e.sh`
5. 消融与鲁棒性:
   - `bash scripts/eval/run_ablations.sh`
   - `bash scripts/eval/run_robustness.sh`
6. 聚合出表:
   - `bash scripts/report/run_week10_aggregate.sh`
   - `python scripts/report/export_paper_tables.py ...`（建议新增）

---

## 附录B: 你后续继续研究时的启动清单（Checklist）

每次重启项目时，按以下顺序检查:

1. 数据版本是否冻结并记录 hash
2. 配置是否固定并记录 hash
3. 强基线是否与主方法同协议对齐
4. 是否至少 3 seeds（主实验建议 5 seeds）
5. 是否包含显著性分析
6. 是否包含 known/unknown 与关键切片结果
7. 是否可一键复现主表
8. 是否完成风险与限制说明

---

## 结束语

你的项目已经具备“工程与研究管理基础设施”这条非常难得的底座。  
下一阶段不要先追更复杂模型，而要先把“数据真实性 + 评测真实性 + 贡献可证伪性”三件事做实。  
只要这三件事完成，后续算力一到位，项目有现实机会进入顶会/顶刊竞争区间。

