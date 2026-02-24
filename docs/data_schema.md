# Data Schema (Week3)

## 1) Tool Metadata Schema
用于 `scripts/data/build_tool_corpus.py` 的清洗后输出。

### Required fields
- `tool_id` (str): `<name>:<version>`
- `name` (str)
- `version` (str, semver `x.y.z`)
- `task` (str)
- `modalities` (list[str], subset of `text/image/audio/video`)
- `status` (str, one of `stable/offline/replaced/newly_added`)
- `snapshots` (list[str], subset of `t/t1`)

### Optional fields
- `source` (str)
- `description` (str)

### Artifacts
- corpus: `data/processed/tool_corpus.jsonl`
- snapshot `v_t`: `data/processed/tool_snapshot_vt.json`
- snapshot `v_t+1`: `data/processed/tool_snapshot_vt1.json`
- drift report: `data/processed/tool_version_diff.json`

## 2) Sample Schema
用于 train/dev/test 与 unknown split。

### Required fields
- `id` (str)
- `query` (str)
- `modality` (str, one of `text/image/audio/video`)
- `gold_tool` (str)

### Recommended fields
- `ambiguity_type` (str)
- `tool_status` (str, one of `stable/offline/replaced/newly_added`)
- `candidates` (list[str])
- `unknown_meta` (object)

## 3) Unknown Split Construction
脚本：`scripts/data/make_unknown_split.py`

### Inputs
- input split jsonl
- `unknown_ratio`
- `seed`
- strategy: `random/status_aware/ambiguity_hard`

### Behavior
- 在可选样本中按策略选取目标比例样本。
- 将选中样本 `gold_tool` 改为 `__unknown__`（或自定义 token）。
- 写入 `unknown_meta`，保留审计信息（source tool、seed、strategy）。
- 生成 audit json（记录实际比例与样本哈希）。

### Audit artifact
- default: `outputs/reports/unknown_split_audit.json`

## 4) Schema Validation
脚本：`scripts/data/validate_schema.py`

### Supported schema
- `tool`
- `sample`
- `auto`（根据第一条记录自动判断）

### Exit code
- `0`: schema valid
- `1`: schema invalid

## 5) Reproducibility Rules
- 所有构造脚本必须接受 `--seed`。
- unknown 比例必须通过参数配置，不允许硬编码。
- 输出路径必须在 `project_root` 内，禁止越界写盘。
