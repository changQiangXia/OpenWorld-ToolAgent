# Round 1 Manual High-Cost Commands

> Rule: high-cost tasks are listed only, not executed by AI.

请你在服务器执行以下命令：

1) 创建统一缓存与产物目录
```bash
mkdir -p /root/autodl-tmp/project/.cache/{pip,hf,torch,tmp} \
         /root/autodl-tmp/project/{data,outputs,models}
```

2) 安装基础依赖（示例，按你实际环境可微调）
```bash
python -m pip install -U pip setuptools wheel
python -m pip install pyyaml jsonlines numpy pandas scikit-learn tqdm
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

3) 初始化最小配置检查（后续脚本接入用）
```bash
python - << 'PY'
import os
required = [
    '/root/autodl-tmp/project/configs',
    '/root/autodl-tmp/project/scripts',
    '/root/autodl-tmp/project/src',
    '/root/autodl-tmp/project/data',
    '/root/autodl-tmp/project/outputs',
]
missing = [p for p in required if not os.path.exists(p)]
print('MISSING=', missing)
print('OK' if not missing else 'FAIL')
PY
```

预计耗时：20-60 分钟（取决于网络与 CUDA 包下载速度）
预计新增磁盘占用：8-20 GB
成功判据：
- 日志出现：`Successfully installed`（pip 安装阶段）
- 文件存在：`/root/autodl-tmp/project/.cache`、`/root/autodl-tmp/project/outputs`

执行后请回传：
- 最后 80 行日志
- 结果文件路径（如有）
- 若失败：完整报错堆栈
