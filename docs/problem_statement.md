# Problem Statement (Week1 Draft)

## Problem
在开放世界工具动态变化场景下，构建多模态工具调用 Agent，降低错选工具与幻觉，并提升端到端任务成功率。

## Scope
- 多模态输入: text/image/audio/video
- 工具状态变化: stable/offline/replaced/newly-added
- 支持 unknown 场景下拒答或澄清

## Non-goals
- 超大规模全参预训练
- 超出算力承受范围的冗余实验矩阵
- 与主线贡献弱相关的堆叠模型
