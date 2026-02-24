# Storage Policy (Current Workspace Is Data Disk)

## Confirmed Fact
当前目录 `/root/autodl-tmp` 已作为数据盘使用。

## Mandatory Placement
以下内容统一放在 `/root/autodl-tmp/project` 下：
- caches: `project/.cache/*`
- model weights: `project/models/*`
- datasets: `project/data/*`
- checkpoints: `project/outputs/checkpoints/*`
- logs/reports/figures: `project/outputs/*`

## Environment Variables (recommended)
在 shell 配置中加入（如 `~/.bashrc`）：
```bash
export PROJECT_ROOT=/root/autodl-tmp/project
export HF_HOME=$PROJECT_ROOT/.cache/hf
export TRANSFORMERS_CACHE=$PROJECT_ROOT/.cache/hf/transformers
export HUGGINGFACE_HUB_CACHE=$PROJECT_ROOT/.cache/hf/hub
export TORCH_HOME=$PROJECT_ROOT/.cache/torch
export PIP_CACHE_DIR=$PROJECT_ROOT/.cache/pip
export TMPDIR=$PROJECT_ROOT/.cache/tmp
```

## Why
- 避免系统盘 30GB 被缓存、权重、日志占满。
- 所有实验资产可审计、可回收、可迁移。
