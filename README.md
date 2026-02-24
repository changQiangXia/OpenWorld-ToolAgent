# Open-World Multimodal Tool Agent

Last updated: 2026-02-24

This repository is a reproducible research workspace for an open-world multimodal tool-calling agent.
It already contains:
- full data/benchmark/training/evaluation pipelines,
- week-by-week experiment assets (Week01 to Week10),
- real-data ingestion support (COCO + LibriSpeech + UCF101),
- benchmark v2 schema + quality/freeze workflow,
- Qwen2.5-VL + Whisper integration,
- paper-planning assets for top-tier conference/journal push.

This README is written as an execution manual so the project can be restarted from a fresh clone with minimal ramp-up.

## 中文快速导读（先看这里）

如英文阅读不便，可先按本节执行，后续英文部分作为命令参考。

当前仓库已具备：
- 从数据构建到训练、评测、汇总的完整链路。
- Week01 到 Week10 的阶段性实验资产。
- 真实数据接入（COCO / LibriSpeech / UCF101）。
- Benchmark v2 构建、质量门禁、冻结报告流程。
- Qwen2.5-VL + Whisper 的独立评测通路。

建议在新机器上先做这 6 步：
1. 安装最小依赖：
```bash
python -m pip install -U pip setuptools wheel
python -m pip install pyyaml torch
```
2. 路径检查：
```bash
python scripts/setup/check_paths.py --project-root /root/autodl-tmp/project
```
3. 干跑训练入口（不真正训练）：
```bash
bash scripts/train/run_baseline1.sh --dry-run
bash scripts/train/run_main_v1.sh --dry-run
```
4. 跑 benchmark v1 全流程：
```bash
bash scripts/data/run_benchmark_v1.sh
```
5. 跑 benchmark v2 + 质量检查：
```bash
python scripts/data/build_benchmark_v2.py --project-root /root/autodl-tmp/project --config configs/data/benchmark_v2.yaml
python scripts/data/run_quality_gates_v2.py --project-root /root/autodl-tmp/project --schema-file docs/paper/benchmark_v2_row_schema_v1.json --train-file data/splits/train_v2.jsonl --dev-file data/splits/dev_v2.jsonl --test-file data/splits/test_v2.jsonl --report-out outputs/reports/benchmark_v2_quality_report_v2splits.json --fail-on-error
```
6. 跑 smoke test：
```bash
python -m unittest tests/test_main_v1_smoke.py
```

常用入口：
- 训练主模型：`bash scripts/train/run_main_v1.sh`
- 主模型评测：`bash scripts/eval/eval_main_v1.sh --split-name test`
- 真实数据全流程：`bash scripts/data/run_real_data_pipeline.sh configs/data/real_public_sources.yaml`
- 三个种子汇总：`bash scripts/report/run_week10_aggregate.sh --seeds 42 43 44`

## 项目概览（目的 / 方法 / 目标）

### A. 项目目的（Project Purpose）

本项目聚焦一个核心问题：
- 在开放世界（Open-World）条件下，面对文本/图像/音频/视频输入，智能体如何稳定地选择正确工具，且在不确定时能够拒答或澄清，而不是“瞎选工具”。

为什么要做这个问题：
- 真实场景中，工具会离线、替换、版本漂移，用户请求也常常模糊或超出能力范围。
- 仅看封闭集 accuracy 的方法不够，需要同时评估 unknown 检测、安全性（幻觉率）、鲁棒性和端到端恢复能力。

### B. 项目所用方法（Method Overview）

本项目采用“数据协议 + 模型 + 策略 + 执行恢复 + 多周实验验证”的完整研究方法。

1. 数据与协议层
- Benchmark v1：先建立可训练、可评测的标准 split 与质量门禁。
- Benchmark v2：升级到更严格行级 schema、语义约束、traceability 与 freeze 流程。
- Real-data-first：接入 COCO/LibriSpeech/UCF101，减少模板化样本偏差。

2. 模型层
- Baseline-1：多数类工具基线，用于低成本 sanity check。
- Main_v1：主模型训练与评估管线（可多配置、多 seed）。
- Qwen2.5-VL + Whisper：强多模态基线，验证更强模型在同协议下表现。

3. 开放世界策略层
- Unknown 阈值校准（calibration）。
- 严格/平衡/召回优先等策略对比（strict / balanced / recall-first）。
- 低置信度时澄清或拒答，降低错误执行风险。

4. 端到端执行层（PSER）
- Plan -> Select -> Execute -> Recover。
- 通过 mock executor 和 recover manager 评估失败后恢复路径与恢复成功率。

5. 证据与统计层
- Week06~Week10 覆盖：错误分析、策略比较、消融、鲁棒性、3-seed 聚合。
- 输出主结果表、异常样本、claim-evidence 矩阵，支撑论文论证链条。

### C. 项目目标（Research Goals）

短期工程目标：
- 让任意新机器 `git clone` 后可按 README 一键恢复实验流程。
- 保证数据构建、训练、评测、聚合全链路可复现、可追溯。

中期研究目标：
- 在 open-world 设置下同时提升：
  - known tool success
  - unknown detection quality
  - end-to-end success
- 同时控制：
  - hallucination rate
  - false reject / unknown miss
  - latency 与稳定性退化

长期论文目标（冲顶）：
- 建立“协议严谨 + 统计充分 + 可复现资产完整”的投稿包。
- 形成能经受审稿质询的证据链：
  - 数据协议与质量门禁
  - 多 seed 显著性与稳定性
  - 消融与鲁棒性解释
  - claim 到 artifact 的一一映射

### D. 当前进度与目标关系（Where We Are Now）

截至 2026-02-24，本仓库已经完成：
- Week01~Week10 主流程与资产沉淀。
- benchmark v2 构建/质量/冻结工具链。
- real-data-first 数据通道。
- Qwen + Whisper 集成评测路径。

也就是说，当前阶段不是“从 0 搭框架”，而是“在已有完整框架上继续做更高质量实验与论文化推进”。

### E. 成功判据（Success Criteria）

后续继续研究时，建议以以下标准判断是否达标：
- 协议达标：v2 quality gate 持续 PASS，版本与报告可追溯。
- 指标达标：核心指标相对基线稳定提升（不是单 seed 偶然提升）。
- 统计达标：多 seed 聚合后方向一致，异常样本可解释。
- 论文达标：每个 claim 都能在仓库内定位到对应数据、脚本、报告和图表。

## 1. Completed Work Snapshot

As of 2026-02-24, the following milestones are implemented in code and docs:
中文说明：这一节列的是已完成成果，不是计划。

1. Week01 to Week10 pipeline milestones:
- Week01: problem framing + metric protocol.
- Week02: baseline-1 train/predict/eval runner.
- Week03: tool corpus + unknown split + schema/data scripts.
- Week04: benchmark v1 build + quality gates + freeze reports.
- Week05: `main_v1` model train/eval + smoke tests.
- Week06: error taxonomy + matrix planning/aggregation scripts.
- Week07: open-world calibration/policy comparison.
- Week08: end-to-end PSER mock execution + recovery evaluation.
- Week09: ablation and robustness stress tests.
- Week10: 3-seed aggregation, anomaly audit, final table generation.

2. Real-data ingestion pipeline:
- `scripts/data/build_real_mm_pool.py`
- `scripts/data/split_real_mm_pool.py`
- `scripts/data/run_real_data_pipeline.sh`
- Config: `configs/data/real_public_sources.yaml`
- Doc: `docs/real_data_ingestion.md`

3. Benchmark v2 pipeline:
- Build: `scripts/data/build_benchmark_v2.py`
- Quality: `scripts/data/run_quality_gates_v2.py`
- Freeze: `scripts/data/freeze_benchmark_v2.py`
- Configs: `configs/data/benchmark_v2.yaml`, `configs/data/benchmark_v2_real.yaml`
- Protocol and schema:
  - `docs/paper/benchmark_v2_protocol.md`
  - `docs/paper/benchmark_v2_row_schema_v1.json`
  - `docs/paper/benchmark_v2_schema_and_annotation_playbook_20260224.md`

4. Qwen2.5-VL + Whisper path:
- Setup: `scripts/setup/download_qwen25_vl_whisper.sh`
- Asset check: `scripts/setup/check_qwen_whisper_assets.py`
- Eval: `scripts/eval/eval_qwen25_vl_whisper.py`
- Wrapper: `scripts/eval/eval_qwen25_vl_whisper.sh`
- Doc: `docs/qwen25_vl_whisper.md`

5. Paper-facing planning assets:
- `docs/paper/top_tier_research_blueprint_20260224.md`
- `docs/paper/top_tier_12week_execution_board_20260224.md`
- `docs/paper/claim_evidence_matrix_template_20260224.md`

## 2. Repository Layout
中文说明：核心目录是 `configs/ scripts/ src/ data/ outputs/ docs/ tests/`。

```text
project/
  configs/
    data/
    train/
    eval/
  scripts/
    setup/
    data/
    train/
    eval/
    report/
  src/
    agent/
    retriever/
    uncertainty/
    execution/
    metrics/
  data/
    raw/
    processed/
    splits/
  outputs/
    logs/
    checkpoints/
    predictions/
    reports/
    figures/
  docs/
    weekly/
    paper/
  tests/
```

## 3. Environment and Dependencies
中文说明：目前没有 `requirements.txt`，需要按下面命令手工安装依赖。

There is no `requirements.txt` yet. Install dependencies explicitly.

Recommended base:
- Python 3.10+
- Linux shell environment
- `ffmpeg` available in `PATH` (needed for video frame extraction in Qwen/Whisper path)

### 3.1 Minimal dependencies (core pipeline)

From repo root (`/root/autodl-tmp/project`):

```bash
python -m pip install -U pip setuptools wheel
python -m pip install pyyaml torch
```

### 3.2 Optional dependencies (Qwen2.5-VL + Whisper path)

```bash
python -m pip install transformers pillow qwen-vl-utils
python -m pip install huggingface_hub modelscope
```

If `ffmpeg` is missing:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

## 4. First 60 Minutes After Clone
中文说明：这节是“新环境恢复”的标准操作顺序，按顺序跑即可。

Run these in order.

1. Path sanity:
```bash
python scripts/setup/check_paths.py --project-root /root/autodl-tmp/project
```

2. Dry-run core training scripts:
```bash
bash scripts/train/run_baseline1.sh --dry-run
bash scripts/train/run_main_v1.sh --dry-run
```

3. Build and validate benchmark v1 quickly:
```bash
bash scripts/data/run_benchmark_v1.sh
```

4. Build and validate benchmark v2 (default config):
```bash
python scripts/data/build_benchmark_v2.py \
  --project-root /root/autodl-tmp/project \
  --config configs/data/benchmark_v2.yaml

python scripts/data/run_quality_gates_v2.py \
  --project-root /root/autodl-tmp/project \
  --schema-file docs/paper/benchmark_v2_row_schema_v1.json \
  --train-file data/splits/train_v2.jsonl \
  --dev-file data/splits/dev_v2.jsonl \
  --test-file data/splits/test_v2.jsonl \
  --report-out outputs/reports/benchmark_v2_quality_report_v2splits.json \
  --fail-on-error
```

5. Smoke tests:
```bash
python -m unittest tests/test_main_v1_smoke.py
```

6. Optional extended smoke:
```bash
python -m unittest tests/test_e2e_smoke.py
```

Note: `tests/test_e2e_smoke.py` expects a policy prediction artifact. If missing, generate it via Week07 flow first.

## 5. End-to-End Workflows
中文说明：这里是各阶段入口命令，按 5.1 -> 5.8 逐步升级复杂度。

### 5.1 Baseline-1 train/predict/eval
中文说明：最基础、最低成本的跑通链路。

```bash
bash scripts/train/run_baseline1.sh
bash scripts/eval/eval_baseline1.sh
```

Primary outputs:
- `outputs/logs/<exp_id>.log`
- `outputs/predictions/<exp_id>.jsonl`
- `outputs/reports/<exp_id>.json`
- `outputs/reports/<exp_id>_eval.json`

### 5.2 Benchmark v1 build/quality/freeze
中文说明：构建 v1 数据并做质量门禁。

```bash
bash scripts/data/run_benchmark_v1.sh
```

Primary outputs:
- `data/splits/train.jsonl`
- `data/splits/dev.jsonl`
- `data/splits/test.jsonl`
- `data/splits/benchmark_v1_manifest.json`
- `outputs/reports/benchmark_v1_quality_report.json`
- `docs/benchmark_v1_report.md`
- `data/splits/benchmark_v1_changelog.md`

### 5.3 Main model (`main_v1`) train/eval
中文说明：核心模型训练与测试都在这里。

Train (default config):
```bash
bash scripts/train/run_main_v1.sh
```

Train (explicit config + seed):
```bash
bash scripts/train/run_main_v1.sh \
  --train-config configs/train/main_v1.yaml \
  --seed 42
```

Evaluate latest checkpoint:
```bash
bash scripts/eval/eval_main_v1.sh --split-name dev
bash scripts/eval/eval_main_v1.sh --split-name test
```

Primary outputs:
- `outputs/checkpoints/<exp_id>.pt`
- `outputs/predictions/<exp_id>_dev.jsonl`
- `outputs/predictions/<exp_id>_<split>_eval.jsonl`
- `outputs/reports/<exp_id>.json`
- `outputs/reports/<exp_id>_<split>_eval.json`

### 5.4 Week06 matrix and error analysis
中文说明：用于系统化对比配置并聚合结果。

Generate matrix plan:
```bash
python scripts/report/plan_week06_matrix.py \
  --project-root /root/autodl-tmp/project
```

Run matrix:
```bash
bash scripts/report/run_week06_matrix.sh
```

Aggregate matrix:
```bash
python scripts/report/aggregate_week06_matrix.py \
  --project-root /root/autodl-tmp/project
```

### 5.5 Week07 open-world policy calibration/compare
中文说明：做 unknown 阈值校准和策略对比。

```bash
python scripts/eval/compare_open_world.py \
  --project-root /root/autodl-tmp/project \
  --dev-prediction outputs/predictions/<exp_id>_dev_eval.jsonl \
  --test-prediction outputs/predictions/<exp_id>_test_eval.jsonl \
  --strategy balanced \
  --output-report outputs/reports/open_world_compare_balanced_seed42.json \
  --output-dev-policy-jsonl outputs/predictions/<exp_id>_dev_policy_balanced.jsonl \
  --output-test-policy-jsonl outputs/predictions/<exp_id>_test_policy_balanced.jsonl
```

### 5.6 Week08 E2E PSER evaluation
中文说明：端到端计划-选择-执行-恢复（PSER）评测。

```bash
bash scripts/eval/eval_e2e.sh
```

Primary outputs:
- `outputs/predictions/<exp_id>_e2e_traces.jsonl`
- `outputs/reports/<exp_id>_e2e_eval.json`

### 5.7 Week09 ablation and robustness
中文说明：消融实验与鲁棒性压力测试。

```bash
bash scripts/eval/run_ablations.sh
bash scripts/eval/run_robustness.sh
```

Primary outputs:
- `outputs/reports/*_ablation_report.json`
- `outputs/reports/*_ablation_table.csv`
- `outputs/reports/*_robustness_report.json`
- `outputs/reports/*_robustness_table.csv`

### 5.8 Week10 seed aggregation and final tables
中文说明：多种子汇总，产出最终主表和稳定性结论。

```bash
bash scripts/report/run_week10_aggregate.sh --seeds 42 43 44
```

Primary outputs:
- `outputs/reports/week10_seed_aggregate_summary.json`
- `outputs/reports/final_main_table.csv`
- `outputs/reports/week10_seed_anomalies.jsonl`

## 6. Real Data Pipeline (COCO + LibriSpeech + UCF101)
中文说明：有真实数据时优先跑这一节，自动串联 v1 + v2。

Full pipeline (recommended):
```bash
bash scripts/data/run_real_data_pipeline.sh configs/data/real_public_sources.yaml
```

Or explicitly pass v2 config:
```bash
bash scripts/data/run_real_data_pipeline.sh \
  configs/data/real_public_sources.yaml \
  configs/data/benchmark_v2_real.yaml
```

This runs:
1. Build unified real pool.
2. Split into `baseline1_train/dev/test`.
3. Build benchmark v1 + quality + freeze.
4. Build benchmark v2 + quality + freeze.

Reference doc: `docs/real_data_ingestion.md`

## 7. Benchmark v2 Detailed Usage
中文说明：推荐重点看 7.2 和 7.4（real-data-first + 质量门禁）。

### 7.1 Standard v2 build

```bash
python scripts/data/build_benchmark_v2.py \
  --project-root /root/autodl-tmp/project \
  --config configs/data/benchmark_v2.yaml
```

### 7.2 Real-data-first build

```bash
python scripts/data/build_benchmark_v2.py \
  --project-root /root/autodl-tmp/project \
  --config configs/data/benchmark_v2_real.yaml
```

### 7.3 Real-data-first with explicit query handling switches

```bash
python scripts/data/build_benchmark_v2.py \
  --project-root /root/autodl-tmp/project \
  --config configs/data/benchmark_v2_real.yaml \
  --real-data-first \
  --query-strategy auto \
  --strip-split-tags \
  --drop-template-queries
```

### 7.4 Quality and freeze

```bash
python scripts/data/run_quality_gates_v2.py \
  --project-root /root/autodl-tmp/project \
  --schema-file docs/paper/benchmark_v2_row_schema_v1.json \
  --train-file data/splits/train_v2_real.jsonl \
  --dev-file data/splits/dev_v2_real.jsonl \
  --test-file data/splits/test_v2_real.jsonl \
  --report-out outputs/reports/benchmark_v2_real_quality_report.json \
  --fail-on-error

python scripts/data/freeze_benchmark_v2.py \
  --project-root /root/autodl-tmp/project \
  --manifest data/splits/benchmark_v2_real_manifest.json \
  --build-report outputs/reports/benchmark_v2_real_build_report.json \
  --quality-report outputs/reports/benchmark_v2_real_quality_report.json \
  --report-md docs/benchmark_v2_real_report.md \
  --changelog-md data/splits/benchmark_v2_real_changelog.md
```

## 8. Qwen2.5-VL + Whisper Workflow
中文说明：这条链路算高成本路径，先确认模型权重和 ffmpeg。

Download model assets:
```bash
bash scripts/setup/download_qwen25_vl_whisper.sh
```

Check assets:
```bash
python scripts/setup/check_qwen_whisper_assets.py \
  --project-root /root/autodl-tmp/project \
  --config configs/eval/qwen25_vl_whisper.yaml
```

Dry-run:
```bash
bash scripts/eval/eval_qwen25_vl_whisper.sh \
  configs/eval/qwen25_vl_whisper.yaml \
  --dry-run
```

Evaluate:
```bash
bash scripts/eval/eval_qwen25_vl_whisper.sh \
  configs/eval/qwen25_vl_whisper.yaml \
  --split-name dev

bash scripts/eval/eval_qwen25_vl_whisper.sh \
  configs/eval/qwen25_vl_whisper.yaml \
  --split-name test
```

Reference docs:
- `docs/qwen25_vl_whisper.md`
- `docs/manual_high_cost_commands_qwen_whisper.md`

## 9. Reproducibility Rules
中文说明：这几条是保证后续能复现实验结果的底线规范。

1. Keep all heavy assets inside this repo root (`/root/autodl-tmp/project`).
2. Always pass `--project-root /root/autodl-tmp/project` when script supports it.
3. Record exact configs used in each run.
4. Prefer seed sets (42/43/44 at minimum) for reportable conclusions.
5. Run quality gates before long training/evaluation on new splits.

Naming pattern used by runtime:
- `exp_id = YYYYMMDD_<task>_<model>_<seed>`

Main output conventions:
- logs: `outputs/logs/<exp_id>.log`
- predictions: `outputs/predictions/<exp_id>*.jsonl`
- reports: `outputs/reports/<exp_id>*.json`
- checkpoints: `outputs/checkpoints/<exp_id>.pt`

## 10. Important Docs to Read Before Continuing Research
中文说明：如需冲顶会/顶刊，这一节文档建议按顺序读一遍。

Execution and metrics:
- `docs/problem_statement.md`
- `docs/metric_protocol.md`
- `docs/open_world_policy.md`
- `docs/e2e_protocol.md`

Error and robustness:
- `docs/error_taxonomy.md`
- `docs/error_analysis_week06.md`
- `docs/ablation_report.md`
- `docs/robustness_report.md`
- `docs/stability_report.md`

Paper planning:
- `docs/paper/top_tier_research_blueprint_20260224.md`
- `docs/paper/top_tier_12week_execution_board_20260224.md`
- `docs/paper/claim_evidence_matrix_template_20260224.md`
- `docs/paper/benchmark_v2_protocol.md`

## 11. Common Failure Cases and Fixes
中文说明：报错时先对照这里，能省掉大部分排查时间。

1. `No checkpoint found under outputs/checkpoints`
- Run training first:
```bash
bash scripts/train/run_main_v1.sh --train-config configs/train/main_v1.yaml --seed 42
```

2. `No prediction file found` in eval scripts
- Run upstream predict/eval stage first, then rerun downstream analysis.

3. `Config requested cuda but CUDA is not available`
- Set `training.device: cpu` in train config or run on GPU-enabled environment.

4. Qwen/Whisper asset check fails
- Ensure model dirs exist:
  - `models/Qwen2.5-VL-7B-Instruct`
  - `models/whisper-large-v3`
- Re-run `bash scripts/setup/download_qwen25_vl_whisper.sh`.

5. `ffmpeg` not found
- Install ffmpeg and ensure it is in `PATH`.

## 12. Next-Step Checklist
中文说明：这是设备升级后重启项目的最短路径。

When resuming with stronger hardware, do this first:

1. Re-run baseline sanity (`check_paths`, dry-runs, smoke tests).
2. Rebuild v2 splits from real-data-first config and pass quality gates.
3. Re-run multi-seed (`42/43/44`, then expand to >=5 seeds for paper stats).
4. Run Week07/08/09/10 chain to regenerate a clean final evidence package.
5. Update claim-evidence matrix in `docs/paper/claim_evidence_matrix_template_20260224.md`.

## 13. Git Tracking Policy (Important)
中文说明：大文件和运行产物默认不进 git，源码/配置/文档保留。

This repo is configured so that `git clone` gives you all core research assets, while excluding heavy/regenerable files.

Tracked (expected in clone):
- source code under `src/` and all scripts under `scripts/`
- configs under `configs/`
- docs under `docs/` (including paper planning/protocol docs)
- lightweight processed assets and canonical benchmark metadata/splits
- curated reports in `outputs/reports/` (if committed)

Ignored by `.gitignore` (not expected in clone unless manually force-added):
- model weights and caches (`models/`, `.cache/`)
- raw public datasets and download archives (`data/raw/public/`, `data/tmp_downloads/`)
- large real-data pool and large generated real-data splits
- runtime artifacts (`outputs/logs/`, `outputs/checkpoints/`, `outputs/predictions/`, `outputs/figures/`)

If you intentionally want to version a currently ignored artifact for a paper release, add it explicitly with `git add -f <path>` and document why it is now part of the reproducibility package.
