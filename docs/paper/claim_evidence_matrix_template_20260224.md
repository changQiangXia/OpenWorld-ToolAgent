# Claim-Evidence Matrix 模板（论文与 Rebuttal 通用）

- 版本: v1.0-template
- 日期: 2026-02-24
- 配套文档:
  - `project/docs/paper/top_tier_research_blueprint_20260224.md`
  - `project/docs/paper/top_tier_12week_execution_board_20260224.md`

---

## 1. 使用目标

本模板用于确保“每条论文主张都有可追溯证据”，避免以下高频问题:

1. 结论先行但证据不足
2. 指标改善但统计不显著
3. 实验存在但无法定位来源文件
4. rebuttal 时无法快速回答“你这条 claim 由什么支持”

---

## 2. 使用规则

1. 一条 claim 只对应一个 `claim_id`
2. 每条 claim 至少有 1 个主证据和 1 个补充证据
3. 每条 claim 必须给出:
   - 指标
   - 对比对象
   - split
   - seeds
   - 显著性结论
   - 证据文件路径
4. 若证据不充分，状态必须标 `At Risk`

---

## 3. 状态标签

1. `Green`: 证据完整且显著，准备投稿
2. `Yellow`: 方向正确但证据不完整或显著性不足
3. `Red`: claim 未被支持或与结果冲突
4. `At Risk`: 有初步信号，但缺关键实验

---

## 4. 主矩阵（核心表）

> 建议每周更新一次，并在投稿前冻结。

| claim_id | claim_statement | split | primary_metric | compare_to | seeds | effect_size | significance | status | main_evidence_file | backup_evidence_file | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 | 在 unknown 检测上优于强基线且 known 成功率不降 | test | unknown_f1 + known_success | qwen_whisper_baseline | 5 | TBD | TBD | At Risk | outputs/reports/TBD.json | outputs/reports/TBD.csv | 需补 paired test |
| C2 | 候选约束显著降低 hallucination | test | hallucination_rate | w/o_candidate_constraint | 5 | TBD | TBD | At Risk | outputs/reports/TBD.json | outputs/reports/TBD_cases.jsonl | 需补分模态切片 |
| C3 | 联合校准在 false_reject 约束下提升 unknown F1 | dev+test | unknown_f1@fr<=x | plain_threshold | 5 | TBD | TBD | At Risk | outputs/reports/TBD.json | outputs/figures/TBD_curve.png | 需补阈值稳定性 |
| C4 | 恢复策略学习优于规则恢复 | e2e_test | recover_success_rate | rule_recover | 5 | TBD | TBD | At Risk | outputs/reports/TBD.json | outputs/predictions/TBD_traces.jsonl | 需真实执行器 |
| C5 | 离线/替换/新增场景具备鲁棒性优势 | robustness_test | e2e_success_rate | strongest_baseline | 5 | TBD | TBD | At Risk | outputs/reports/TBD.json | outputs/reports/TBD_table.csv | 需 known/unknown 分表 |

---

## 5. Claim 逐条证据卡片（模板）

## Claim ID: C1

### Claim 语句

`<一句话可证伪陈述>`

### 证据要求

1. 主指标:
   - `<metric_1>`
   - `<metric_2>`
2. 对比对象:
   - `<baseline_name>`
3. 数据范围:
   - `<split/subset>`
4. 统计要求:
   - `>=5 seeds`
   - `95% CI`
   - `<significance_test>`

### 当前证据

1. 主证据文件:
   - `outputs/reports/<...>.json`
2. 补充证据文件:
   - `outputs/reports/<...>.csv`
3. 可视化:
   - `outputs/figures/<...>.png`

### 评估结论

1. 是否支持 claim: `Yes/No/Partial`
2. 风险等级: `Low/Medium/High`
3. 下步行动:
   - `<action_1>`
   - `<action_2>`

---

## 6. Rebuttal 映射表（模板）

| reviewer_question | related_claim_id | direct_evidence | fallback_evidence | response_owner | status |
|---|---|---|---|---|---|
| 你的方法是否仅在小样本有效？ | C1/C5 | outputs/reports/TBD_large_eval.json | outputs/reports/TBD_slice_table.csv | TBD | At Risk |
| 改进是否来自阈值技巧而非方法创新？ | C2/C3 | outputs/reports/TBD_ablation.json | outputs/reports/TBD_threshold_sweep.json | TBD | At Risk |
| 与强基线是否公平比较？ | C1 | outputs/reports/TBD_fairness_audit.json | docs/paper/baseline_fairness_protocol.md | TBD | At Risk |
| 真实执行器下是否仍成立？ | C4/C5 | outputs/reports/TBD_e2e_real.json | outputs/predictions/TBD_real_traces.jsonl | TBD | At Risk |

---

## 7. 快速自检清单（投稿前）

每条 `Green` claim 必须满足:

1. 主指标明确且与论文文本一致
2. 对比对象和设定完全公平
3. seed 数满足标准（建议 5）
4. 给出显著性结论（不仅均值）
5. 有可追溯文件路径
6. 有失败案例分析，不只报平均值

---

## 8. 周更模板（建议复制后每周填写）

### Week `<N>`

1. 新增/更新 claim:
   - `<C1/C2/...>`
2. 状态变化:
   - `C1: At Risk -> Yellow`
3. 新增证据文件:
   - `outputs/reports/<...>`
4. 仍缺失证据:
   - `<missing_item>`
5. 下周必须补齐:
   - `<must_do_1>`

---

## 9. 常见失败模式与修复建议

1. 失败模式: claim 太大，证据碎片化
   - 修复: 拆成两个可证伪 claim

2. 失败模式: 指标改进但方差过大
   - 修复: 增 seed、加约束、做异常样本剖析

3. 失败模式: 强基线协议不一致
   - 修复: 增加 fairness audit 并强制通过

4. 失败模式: rebuttal 问题找不到证据
   - 修复: 先补 “question -> file path” 映射

