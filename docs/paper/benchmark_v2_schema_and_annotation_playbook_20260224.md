# benchmark_v2 数据 Schema 与标注手册（草稿）

- 版本: v0.1-draft
- 日期: 2026-02-24
- 适用项目: Open-World Multimodal Tool Agent
- 关联目录: `/root/autodl-tmp/project`
- 机器可读 schema: `project/docs/paper/benchmark_v2_row_schema_v1.json`

---

## 1. 目标与范围

本手册用于定义 `benchmark_v2` 的:

1. 样本级 schema（字段、类型、约束）
2. 标注流程（初标、复核、仲裁）
3. 质量门禁（发布前必须通过）

本手册服务于“可投稿级实验”，因此要求:

1. 标签可追溯
2. 规则可复现
3. 结论可审计

---

## 2. 样本定义

### 2.1 单条样本语义

一条样本表示“在给定多模态输入与上下文约束下，系统应执行哪个工具，或应拒答/澄清（unknown）”。

### 2.2 样本最小结构（强制字段）

1. `id`
2. `split`
3. `query`
4. `modality`
5. `candidates`
6. `gold_tools`
7. `is_unknown_gold`
8. `unknown_reason_type`
9. `mapping_type`
10. `ambiguity_type`
11. `tool_status`
12. `annotation`
13. `trace`

---

## 3. 字段规范（Schema 说明）

### 3.1 核心任务字段

1. `id: string`
   - 全局唯一
   - 推荐格式: `<split>_<6-digit>` 或 `<source>:<hash>`

2. `split: enum`
   - `train | dev | test`

3. `query: string`
   - 用户原始请求文本
   - 禁止模板占位文本作为主评估集（例如 `Synthetic query ...`）

4. `modality: enum`
   - `text | image | audio | video | multimodal`

5. `media: array<object>`
   - 可为空（text-only）
   - 每个元素包含:
     - `type: text|image|audio|video`
     - `path: string`（项目内相对路径）
     - `sha256: string`（可选）

6. `candidates: array<string>`
   - 候选工具集合
   - 推荐至少 1 个候选；unknown 场景可允许为空，但需在 `trace` 解释

7. `gold_tools: array<string>`
   - one-to-one: 长度=1
   - one-to-many: 长度>=2
   - unknown 样本: 应仅为 `["__unknown__"]`

8. `is_unknown_gold: boolean`
   - 与 `gold_tools` 一致

9. `unknown_reason_type: enum`
   - `none`
   - `missing_capability`
   - `insufficient_constraints`
   - `safety_policy`
   - `tool_unavailable`
   - `version_incompatible`
   - `ambiguous_intent`
   - unknown 样本禁止为 `none`
   - known 样本必须为 `none`

10. `mapping_type: enum`
    - `one_to_one | one_to_many`

11. `ambiguity_type: enum`
    - `none`
    - `lexical_ambiguity`
    - `missing_constraints`
    - `underspecified_tool_goal`
    - `multimodal_conflict`
    - `version_drift`
    - `long_tail_intent`

12. `tool_status: enum`
    - `stable | offline | replaced | newly_added`

### 3.2 标注与追溯字段

1. `annotation: object`
   - `guideline_version: string`
   - `annotator_id: string`
   - `reviewer_id: string`
   - `adjudicator_id: string`
   - `label_source: human|rule|hybrid`
   - `confidence: number[0,1]`
   - `review_status: draft|reviewed|adjudicated|final`
   - `created_at_utc: string`
   - `updated_at_utc: string`

2. `trace: object`
   - `source_dataset: string`
   - `source_id: string`
   - `source_path: string`
   - `generation_pipeline: string`
   - `notes: string`

---

## 4. 标注流程（SOP）

### 4.1 阶段拆分

1. 阶段A 初标
2. 阶段B 复核
3. 阶段C 仲裁
4. 阶段D 质检与冻结

### 4.2 初标规范

初标员必须完成:

1. 判断任务是否可由现有工具完成
2. 给出 `gold_tools`
3. 给出 `is_unknown_gold`
4. 若 unknown，必须给 `unknown_reason_type != none`
5. 标记 `ambiguity_type` 与 `mapping_type`

### 4.3 复核规范

复核员必须独立判断，不参考初标意见。

若初标与复核冲突，冲突类型记录为:

1. `tool_mismatch`
2. `unknown_mismatch`
3. `mapping_mismatch`
4. `ambiguity_mismatch`

### 4.4 仲裁规范

仲裁员给最终标签，并写明 `adjudication_note`:

1. 冲突点
2. 决策理由
3. 引用规则条目

---

## 5. 质量门禁（发布前必须通过）

### 5.1 Schema 门禁

1. 所有样本通过 JSON Schema 校验
2. 必填字段缺失率为 0
3. 枚举值越界率为 0

### 5.2 一致性门禁

1. `is_unknown_gold` 与 `gold_tools` 一致率 100%
2. known 样本 `unknown_reason_type=none`
3. unknown 样本 `unknown_reason_type!=none`
4. one_to_many 样本 `len(gold_tools)>=2`

### 5.3 泄漏与重复门禁

1. train-dev-test 无 ID 重叠
2. 文本近重复低于阈值（建议 <1%）
3. 媒体 hash 不跨 split 重复（可容忍白名单项）

### 5.4 分布门禁

主评测 split（dev/test）应满足:

1. 每模态占比不低于最小阈值（建议 >=10%）
2. each `tool_status` 样本数满足最低可统计规模
3. unknown 占比在目标区间（建议 15%-30%）

### 5.5 标注可靠性门禁

1. 双标覆盖率 >= 20%（建议 >=30%）
2. `is_unknown_gold` 的 Cohen’s kappa >= 0.75
3. `gold_tools` 一致率达到预设阈值（建议 >=0.8）

---

## 6. known/unknown 判定规则（操作级）

### 6.1 优先判定顺序

1. 是否违反安全/合规
2. 是否缺少必要约束（无法安全执行）
3. 是否当前工具集无能力覆盖
4. 是否工具当前状态不可执行（offline / incompatible）

满足任一条件可判 unknown，但必须记录具体 `unknown_reason_type`。

### 6.2 易错场景特别说明

1. `tool_unavailable` 与 `missing_capability` 区分:
   - 前者“本应能做但当前不可用”
   - 后者“工具集合从能力上不支持”
2. `insufficient_constraints`:
   - 只要补充约束后可执行，优先该类型
3. `ambiguous_intent`:
   - 多目标冲突且无可判别偏好

---

## 7. one-to-many 标注规则

### 7.1 允许 one-to-many 的必要条件

1. 多工具均可达到任务目标
2. 这些工具在语义上并列可接受
3. 不是因为标注员不确定而“多选兜底”

### 7.2 禁止 one-to-many 的情形

1. 仅一个工具符合关键约束
2. 多选会掩盖模型错误
3. 只是上游候选列表包含多个工具

---

## 8. 示例（规范样本）

### 8.1 Known one-to-one 示例

```json
{
  "id": "test_001245",
  "split": "test",
  "query": "请提取这张发票上的总金额",
  "modality": "image",
  "media": [{"type": "image", "path": "data/raw/public/.../img_1245.jpg"}],
  "candidates": ["ocr_image", "qa_text"],
  "gold_tools": ["ocr_image"],
  "is_unknown_gold": false,
  "unknown_reason_type": "none",
  "mapping_type": "one_to_one",
  "ambiguity_type": "none",
  "tool_status": "stable"
}
```

### 8.2 Unknown 示例

```json
{
  "id": "test_009876",
  "split": "test",
  "query": "请直接调用当前不存在的医学影像诊断插件",
  "modality": "text",
  "media": [],
  "candidates": ["qa_text", "search_web"],
  "gold_tools": ["__unknown__"],
  "is_unknown_gold": true,
  "unknown_reason_type": "missing_capability",
  "mapping_type": "one_to_one",
  "ambiguity_type": "none",
  "tool_status": "stable"
}
```

---

## 9. 发布工件（Release Artifacts）

每次 `benchmark_v2` 发布至少包含:

1. `train/dev/test.jsonl`
2. `manifest.json`
3. `schema_validation_report.json`
4. `label_consistency_report.json`
5. `leakage_report.json`
6. `distribution_report.json`
7. `known_unknown_audit.json`

---

## 10. 与现有仓库的落地建议

### 10.1 建议新增脚本

1. `scripts/data/validate_benchmark_v2_schema.py`
2. `scripts/data/check_benchmark_v2_consistency.py`
3. `scripts/data/check_benchmark_v2_leakage.py`
4. `scripts/data/summarize_benchmark_v2_distribution.py`

### 10.2 建议新增配置

1. `configs/data/benchmark_v2.yaml`
2. `configs/data/benchmark_v2_quality_gates.yaml`

### 10.3 最小执行顺序

1. 构建 v2 数据
2. 跑 schema 校验
3. 跑一致性/泄漏/分布门禁
4. 生成 release artifacts
5. 冻结 manifest 与 hash

---

## 11. 当前草稿状态（下一步）

本手册为可执行草稿，下一步建议:

1. 增加“真实标注样例库”
2. 增加“边界案例 FAQ”
3. 与评测脚本字段做一轮严格对齐

