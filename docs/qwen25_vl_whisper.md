# Qwen2.5-VL-7B + Whisper Integration

## Goal
Run a strong multimodal baseline with:
- `Qwen2.5-VL-7B-Instruct` for tool routing
- `whisper-large-v3` for audio transcription

This baseline reads benchmark splits and writes prediction/report files in the same schema used by existing open-world scripts.

## Files Added
- config: `configs/eval/qwen25_vl_whisper.yaml`
- model download script: `scripts/setup/download_qwen25_vl_whisper.sh`
- model asset check: `scripts/setup/check_qwen_whisper_assets.py`
- selector module: `src/agent/qwen_vl_whisper.py`
- eval script: `scripts/eval/eval_qwen25_vl_whisper.py`
- eval shell wrapper: `scripts/eval/eval_qwen25_vl_whisper.sh`

## Model Cache/Weight Policy
All model/cache paths stay under project root:
- model weights: `project/models/`
- HF cache: `project/.cache/hf/`
- ModelScope cache: `project/.cache/modelscope/`

## Usage
From `/root/autodl-tmp/project`:

1. Download model weights (manual high-cost step):
```bash
bash scripts/setup/download_qwen25_vl_whisper.sh
```
Default behavior:
- try ModelScope first
- fallback to HuggingFace if ModelScope fails

2. Check assets:
```bash
python scripts/setup/check_qwen_whisper_assets.py \
  --project-root /root/autodl-tmp/project \
  --config configs/eval/qwen25_vl_whisper.yaml
```

3. Dry-run eval command:
```bash
bash scripts/eval/eval_qwen25_vl_whisper.sh configs/eval/qwen25_vl_whisper.yaml --dry-run
```

4. Evaluate on dev:
```bash
bash scripts/eval/eval_qwen25_vl_whisper.sh configs/eval/qwen25_vl_whisper.yaml --split-name dev
```

5. Evaluate on test:
```bash
bash scripts/eval/eval_qwen25_vl_whisper.sh configs/eval/qwen25_vl_whisper.yaml --split-name test
```

## Outputs
- predictions: `outputs/predictions/<exp_id>_<split>_qwen_eval.jsonl`
- report: `outputs/reports/<exp_id>_<split>_qwen_eval.json`
- log: `outputs/logs/<exp_id>_<split>_qwen_eval.log`

## Notes
- Audio samples are transcribed by Whisper first, then routed by Qwen.
- Video is handled with first-frame extraction (`ffmpeg`) in multimodal mode.
- If media decoding fails, the script falls back to text prompt only for that sample.
