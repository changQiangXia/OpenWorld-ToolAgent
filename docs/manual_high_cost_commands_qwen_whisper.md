# Qwen2.5-VL + Whisper Manual High-Cost Commands

> Rule: download/inference with large models is high-cost, run manually.

请你在服务器执行以下命令：

1) 设置缓存和模型目录（全部放数据盘）
```bash
mkdir -p /root/autodl-tmp/project/.cache/hf \
         /root/autodl-tmp/project/.cache/modelscope \
         /root/autodl-tmp/project/models
```

2) 下载 Qwen2.5-VL-7B 和 Whisper 权重（优先 ModelScope，失败自动回退 HuggingFace）
```bash
bash /root/autodl-tmp/project/scripts/setup/download_qwen25_vl_whisper.sh
```

3) 校验模型文件是否到位
```bash
python /root/autodl-tmp/project/scripts/setup/check_qwen_whisper_assets.py \
  --project-root /root/autodl-tmp/project \
  --config /root/autodl-tmp/project/configs/eval/qwen25_vl_whisper.yaml
```

4) 先做 dry-run，确认配置和路径
```bash
bash /root/autodl-tmp/project/scripts/eval/eval_qwen25_vl_whisper.sh \
  /root/autodl-tmp/project/configs/eval/qwen25_vl_whisper.yaml \
  --dry-run
```

5) 跑 dev/test 评测
```bash
bash /root/autodl-tmp/project/scripts/eval/eval_qwen25_vl_whisper.sh \
  /root/autodl-tmp/project/configs/eval/qwen25_vl_whisper.yaml \
  --split-name dev

bash /root/autodl-tmp/project/scripts/eval/eval_qwen25_vl_whisper.sh \
  /root/autodl-tmp/project/configs/eval/qwen25_vl_whisper.yaml \
  --split-name test
```

预计耗时：下载 30-120 分钟；评测 20-180 分钟（取决于样本量）
预计新增磁盘占用：约 30-45 GB
成功判据：
- 日志出现：`[DONE] model downloads finished`
- 文件存在：`/root/autodl-tmp/project/models/Qwen2.5-VL-7B-Instruct`
- 文件存在：`/root/autodl-tmp/project/models/whisper-large-v3`
- 结果文件存在：`outputs/predictions/*_qwen_eval.jsonl`

执行后请回传：
- 最后 80 行日志
- 结果文件路径
- 若失败：完整报错堆栈
