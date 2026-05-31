# MiniMind 项目补充文档

> 本文档为 [README.md](./README.md) 的补充参考，提供模型架构和高级特性的深入技术细节。
> 主要内容（项目结构、配置、训练流程等）请参阅 [README.md](./README.md)。

---

## 1. 高级模块详解 (model_advanced.py)

基于 DeepSeek V4 论文 (arxiv:2512.24880) 实现：

### mHC — Manifold-Constrained Hyper-Connections

- 将单流残差扩展为 n 条并行流
- 混合矩阵 B 通过 Sinkhorn-Knopp 迭代约束到 Birkhoff 多胞形（双随机矩阵）
- 谱范数 ≤ 1，确保深层不爆炸
- 类：`MHCConnection`, `MHCBlock`

### CSA — Compressed Sparse Attention

- KV 分块压缩 (compress_factor=4)：每 m 个 KV → 1 个
- Lightning Indexer：低维投影选 top-k 块
- Core Attention：仅在选中块上计算完整注意力
- 1M 上下文 FLOPs 仅为标准注意力的 27%
- 类：`CompressedSparseAttention`, `LightningIndexer`

### MHC_CSABlock — 集成块

- 兼容 `MiniMindBlock` 接口
- 支持 CSA 注意力和 mHC 流连接
- 可选 MoE / TTT FFN

```python
from model.model_advanced import MHC_CSABlock

block = MHC_CSABlock(
    layer_id=0,
    config=config,
    use_csa=True,      # 启用压缩稀疏注意力
    num_streams=2      # mHC 流数量
)
out, cache = block(hidden_states, position_embeddings)
```

---

## 2. Muon 优化器详解

基于动量梯度 + Newton-Schulz 正交归一化，存储在 `trainer/trainer_utils.py`：

| 类 | 说明 |
|------|------|
| `newton_schulz_5(G)` | 将梯度矩阵投影到最近正交矩阵（三次多项式初始化 + 5 次 NS 迭代） |
| `Muon` | 纯 Muon 优化器，所有参数用 NS 归一化 |
| `MixedMuonAdamW` | 混合优化器：≥2D 大矩阵 → Muon，1D/小矩阵 → AdamW |
| `create_muon_optimizer(model)` | 工厂函数，返回 MixedMuonAdamW |

使用方式：`--optimizer muon`

---

## 3. OmniConfig 详解 (多模态)

继承自 `MiniMindConfig`，新增参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_talker_hidden_layers` | 4 | Talker 模块层数 |
| `talker_hidden_size` | `hidden_size` | Talker 隐藏维度 |
| `audio_vocab_size` | 2112 | 音频 codec 词表大小 |
| `spk_emb_size` | 192 | 说话人嵌入维度 |
| `image_hidden_size` | 768 | 视觉嵌入维度 |
| `image_token_len` | 64 | 视觉 token 长度 |
| `bridge_layer` | `num_hidden_layers // 2 - 1` | Thinker→Talker 桥接层 |

Thinker-Talker 双路径结构：
- **Thinker**：处理文本 + 图像 + 音频理解
- **Talker**：生成语音输出，集成 Mimi 编解码器

---

## 4. 诚实性训练详解

- 检测模型"不知道"的场景（`trainer/honest_training.py`）
- 在 `know` vs `don't know` 场景间施加差别奖励
- 集成到 Agent RL (`train_agent.py`) 和 PPO/GRPO 训练中

---

## 5. 多模态编码器模型

配置文件已放置在 `model/` 对应目录中，模型权重需从 HuggingFace/ModelScope 下载：

| 编码器 | 用途 | 模型路径 |
|--------|------|----------|
| SenseVoiceSmall | 语音识别 (ASR) | `model/SenseVoiceSmall/` |
| SigLIP2 | 视觉理解 | `model/siglip2-base-p32-256-ve/` |
| Mimi | 音频编解码 | `model/mimi/` |
| CAMPPlus | 说话人识别 | `model/campplus/` |
| SileroVAD | 语音活动检测 | `model/vad/silero_vad.onnx` |

```python
from modelscope import snapshot_download

snapshot_download('iic/SenseVoiceSmall', local_dir='model/SenseVoiceSmall')
snapshot_download('iic/speech_campplus_sv_zh-cn_16k-common', local_dir='model/campplus')
```
