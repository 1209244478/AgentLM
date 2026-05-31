# MiniMind 项目文档

> "大道至简" — 从 0 训练 64M 超小语言模型，仅需 3 元成本和 2 小时。

---

## 目录

- [1. 项目结构](#1-项目结构)
- [2. 模型架构](#2-模型架构)
- [3. 模型配置](#3-模型配置)
- [4. 运行指南](#4-运行指南)
  - [4.1 环境安装](#41-环境安装)
  - [4.2 数据准备](#42-数据准备)
  - [4.3 训练流程](#43-训练流程)
  - [4.4 模型导出与部署](#44-模型导出与部署)
  - [4.5 推理与对话](#45-推理与对话)
- [5. 高级特性](#5-高级特性)
- [6. 多模态 (MiniMind-O)](#6-多模态-minimind-o)

---

## 1. 项目结构

```
minimind/
├── model/                           # 模型定义
│   ├── model_minimind.py            # 核心 LLM：MiniMindConfig, MiniMindForCausalLM,
│   │                                #   MiniMindBlock, TTTFeedForward, MTPHead, MoE,
│   │                                #   KVCache (预分配), Attention (FA2/SDPA/Manual 三级路由)
│   ├── model_advanced.py            # DeepSeek V4 高级模块：mHC 流形约束超连接 + CSA 压缩注意力
│   ├── model_omni.py                # 全模态 Thinker-Talker 双路径模型
│   ├── model_lora.py                # LoRA 微调模块
│   ├── minimind_rag.py              # RAG 检索增强生成
│   ├── tokenizer.json               # LLM tokenizer
│   ├── tokenizer_config.json        # tokenizer 配置
│   ├── mimi/                        # 音频编解码器配置 (Mimi)
│   ├── SenseVoiceSmall/             # ASR 语音识别编码器配置
│   ├── siglip2-base-p32-256-ve/     # 视觉编码器配置 (SigLIP2)
│   ├── campplus/                    # 说话人识别编码器配置 (CAMPPlus)
│   ├── speaker/                     # 声纹克隆模型
│   └── vad/                         # 语音活动检测模型 (Silero VAD)
│
├── trainer/                         # 训练脚本
│   ├── trainer_utils.py             # 训练工具集：学习率调度、DDP、checkpoint、Muon优化器
│   ├── train_full_sft.py            # 全量 SFT 微调（主入口，支持 --optimizer muon）
│   ├── train_pretrain.py            # 预训练
│   ├── train_distillation.py        # 知识蒸馏
│   ├── train_lora.py                # LoRA 微调
│   ├── train_dpo.py                 # DPO 对齐训练
│   ├── train_grpo.py                # GRPO 组相对策略优化
│   ├── train_ppo.py                 # PPO 近端策略优化
│   ├── train_agent.py               # Agent RL 工具使用训练
│   ├── train_meta_ttt.py            # Meta-TTT 推理时训练
│   ├── train_sft_omni.py            # 多模态 Omni SFT 训练
│   ├── train_tokenizer.py           # Tokenizer 训练
│   ├── honest_training.py           # 诚实性训练奖励模块
│   ├── rollout_engine.py            # Rollout 生成引擎
│   └── eval_utils.py                # TTT 评估工具
│
├── dataset/                         # 数据集
│   ├── lm_dataset.py                # 文本数据集加载器
│   ├── omni_dataset.py              # 多模态数据集加载器
│   ├── convert_mint_arxiv.py        # MINT-1T ArXiv 数据转换脚本（断点续传+超时重试）
│   ├── pretrain_combined.jsonl      # 预训练合并数据 (~2.1GB, 含 MINT-1T)
│   ├── sft_combined.jsonl           # SFT 合并数据 (~1.7GB, 含 MINT-1T)
│   ├── pretrain_t2t_mini.jsonl      # 预训练原始数据 (~1.2GB)
│   ├── sft_t2t_mini.jsonl           # SFT 原始数据 (~1.6GB)
│   ├── rlaif.jsonl                  # RL 对齐数据 (~23MB)
│   └── eval_omni/                   # 多模态评估数据 (44 个文件)
│
├── scripts/                         # 工具脚本
│   ├── web_demo.py                  # LLM 对话演示
│   ├── web_demo_omni.py             # 多模态对话演示
│   ├── convert_model.py             # 模型格式转换
│   ├── convert_omni.py              # 多模态模型格式转换
│   ├── chat_api.py                  # API 对话接口
│   ├── serve_openai_api.py          # OpenAI 兼容 API 服务
│   └── eval_toolcall.py             # Tool Call 评估
│
├── webui/                           # Web UI
│   ├── web_demo.py                  # 多模态 Web 后端
│   └── web_demo.html                # 前端界面
│
├── tests/                           # 测试
│   └── test_all.py                  # 统一测试：27 项覆盖构建/前向/TTT/MHC+CSA/Muon/OPD/生成
│
├── mini-RAG/                        # 外挂 RAG 系统 (LightRAG)
├── minimind-3/                      # MiniMind-3 预训练模型发布包
│   ├── model.safetensors            # 模型权重 (~123MB)
│   ├── config.json                  # MiniMind-3 配置 (768, 8L)
│   ├── config_5.json                # MiniMind-5 配置 (1024, 12L, 含MTP/TTT)
│   └── tokenizer 相关文件
│
├── eval_llm.py                      # LLM 评估脚本
├── eval_omni.py                     # 多模态评估脚本
└── requirements.txt                 # 依赖列表
```

---

## 2. 模型架构

### 2.1 MiniMindConfig 参数说明

核心配置类 `MiniMindConfig` 继承自 `transformers.PretrainedConfig`：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `hidden_size` | 768 | 隐藏层维度 |
| `num_hidden_layers` | 8 | Transformer 层数 |
| `num_attention_heads` | 8 | 注意力头数 |
| `num_key_value_heads` | 4 | KV 头数 (GQA) |
| `head_dim` | `hidden_size // num_attention_heads` | 单头维度 |
| `intermediate_size` | `hidden_size * π` 取整 | FFN 中间层维度 |
| `vocab_size` | 6400 | 词表大小 |
| `max_position_embeddings` | 32768 | 最大序列长度 |
| `rope_theta` | 1e6 | RoPE 基频 |
| `rms_norm_eps` | 1e-6 | RMSNorm epsilon |
| `dropout` | 0.0 | Dropout 概率 |
| `flash_attn` | True | Flash Attention |
| `use_moe` | False | MoE 混合专家 |
| `num_experts` | 4 | MoE 专家数 |
| `num_experts_per_tok` | 1 | 每 token 激活专家数 |
| **`layer_share_factor`** | 1 | 跨层参数共享（N=2 表示每 2 层共享） |
| **`mtp_num_heads`** | 0 | 多 token 预测头数 |
| **`mtp_loss_weight`** | 0.1 | MTP 辅助损失权重 |
| **`ttt_enabled`** | False | 推理时训练 (In-Place TTT) |
| **`ttt_lr`** | 1e-4 | TTT 学习率 |
| **`ttt_chunk_size`** | 512 | TTT chunk 大小 |

### 2.2 模型组件

```
MiniMindForCausalLM
├── MiniMindModel (Base)
│   ├── embed_tokens (nn.Embedding)
│   └── layers (共享组 × 重复次数)
│       └── MiniMindBlock
│           ├── Attention (三级路由)
│           │   ├── Flash Attention 2  ← 训练，最快 (需 pip install flash-attn)
│           │   ├── PyTorch SDPA       ← 推理+KV Cache，次快
│           │   └── Manual Attention   ← Fallback 兼容
│           │   ├── q_proj / k_proj / v_proj / o_proj
│           │   ├── q_norm / k_norm (RMSNorm)
│           │   └── RoPE 旋转位置编码 (优化实现)
│           ├── FeedForward (或 MOEFeedForward / TTTFeedForward)
│           │   ├── gate_proj → SiLU
│           │   ├── up_proj → SiLU → down_proj
│           │   └── (MoE: 多专家 + Router + 共享专家)
│           │   └── (TTT: chunk 内 self-supervised 权重更新)
│           └── RMSNorm (pre-norm, torch.compile 友好)
├── mtp_heads (可选)
│   └── MTPHead (多 token 预测头)
└── lm_head (输出投影)
```

**性能优化要点：**
- **Flash Attention 2**：训练时自动启用（需安装 `flash-attn`），O(N²) → O(N) 显存
- **KVCache 预分配**：推理时固定大小 buffer + `copy_` 原地写入，避免 `torch.cat` 的 O(n²) 分配
- **SDPA + KV Cache 兼容**：推理时正确构造 `attn_mask`，走 SDPA 快速路径
- **RoPE 优化**：直接拆分计算 `q1*cos - q2*sin`，减少中间张量分配
- **RMSNorm 内联**：消除独立 `norm()` 方法，1 次 float 转换
- **Generate Prefill**：先一次处理完整 prompt，再逐 token decode，首 token 延迟降低 10x+

### 2.3 高级模块 (model_advanced.py)

基于 DeepSeek V4 论文 (arxiv:2512.24880) 实现：

**mHC — Manifold-Constrained Hyper-Connections**
- 将单流残差扩展为 n 条并行流
- 混合矩阵 B 通过 Sinkhorn-Knopp 迭代约束到 Birkhoff 多胞形（双随机矩阵）
- 谱范数 ≤ 1，确保深层不爆炸
- 类：`MHCConnection`, `MHCBlock`

**CSA — Compressed Sparse Attention**
- KV 分块压缩 (compress_factor=4)：每 m 个 KV → 1 个
- Lightning Indexer：低维投影选 top-k 块
- Core Attention：仅在选中块上计算完整注意力
- 1M 上下文 FLOPs 仅为标准注意力的 27%
- 类：`CompressedSparseAttention`, `LightningIndexer`

**MHC_CSABlock** — 集成块
- 兼容 `MiniMindBlock` 接口
- 支持 CSA 注意力和 mHC 流连接
- 可选 MoE / TTT FFN

### 2.4 OmniConfig (多模态)

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

## 3. 模型配置

### 3.1 MiniMind-3

文件：`minimind-3/config.json`

```
hidden_size:              768
num_hidden_layers:        8
num_attention_heads:      8
num_key_value_heads:      4
head_dim:                 96
intermediate_size:        2432
vocab_size:               6400
max_position_embeddings:  32768
rope_theta:               1,000,000
参数量:                    ~64M (模型维度) / ~128M (含嵌入)
```

### 3.2 MiniMind-5 (增强版，默认配置)

文件：`minimind-3/config_5.json`

```
hidden_size:              1024
num_hidden_layers:        12
num_attention_heads:      16
num_key_value_heads:      8
head_dim:                 64
intermediate_size:        3520
vocab_size:               32000
layer_share_factor:       2       ← 跨层参数共享
mtp_num_heads:            2       ← 多 token 预测
ttt_enabled:              True    ← 推理时训练
ttt_chunk_size:           512
参数量:                    ~186M (含 TTT + MTP 头)
```

> MiniMind-5 已设为训练脚本默认配置（`hidden_size=1024`, `num_hidden_layers=12`），
> 数据路径默认指向合并数据集 `pretrain_combined.jsonl` / `sft_combined.jsonl`。

### 3.3 Omni 配置

使用 `OmniConfig`，继承 MiniMind-3 配置，额外包含 Thinker-Talker 参数。

---

## 4. 运行指南

### 4.1 环境安装

```bash
pip install torch transformers datasets wandb
# 多模态额外依赖
pip install onnxruntime soundfile librosa
```

### 4.2 数据准备

预训练和 SFT 数据格式为 JSONL，每行一个 JSON 对象，包含 `conversations` 字段。

数据文件默认位置（可覆盖）：
| 文件 | 用途 | 大小 |
|------|------|------|
| `dataset/pretrain_combined.jsonl` | 预训练（含 MINT-1T ArXiv） | ~2.1 GB |
| `dataset/sft_combined.jsonl` | SFT 微调（含 MINT-1T ArXiv） | ~1.7 GB |
| `dataset/pretrain_t2t_mini.jsonl` | 预训练（原始） | ~1.2 GB |
| `dataset/sft_t2t_mini.jsonl` | SFT 微调（原始） | ~1.6 GB |
| `dataset/pretrain_mint_arxiv.jsonl` | MINT-1T ArXiv 预训练 | ~889 MB |
| `dataset/sft_mint_arxiv.jsonl` | MINT-1T ArXiv SFT | ~21 MB |
| `dataset/rlaif.jsonl` | RL 对齐 | ~23 MB |

> MINT-1T ArXiv 数据通过 `dataset/convert_mint_arxiv.py` 从 HuggingFace 下载并转换，
> 支持 SSL 超时重试和断点续传。合并后的 `*_combined.jsonl` 为训练默认数据。

### 4.3 训练流程

#### 第一步：训练 Tokenizer

```bash
python trainer/train_tokenizer.py
```

#### 第二步：预训练

```bash
# MiniMind-5 默认配置 (1024, 12L, ~186M)，数据默认 pretrain_combined.jsonl
python trainer/train_pretrain.py \
    --epochs 2 --batch_size 32 --learning_rate 5e-4 \
    --accumulation_steps 8

# MiniMind-3 旧配置 (768, 8L, ~64M)
python trainer/train_pretrain.py \
    --hidden_size 768 --num_hidden_layers 8 \
    --data_path ../dataset/pretrain_t2t_mini.jsonl \
    --epochs 2 --batch_size 32 --learning_rate 5e-4
```

#### 第三步：SFT 监督微调

```bash
# MiniMind-5 默认配置，数据默认 sft_combined.jsonl
python trainer/train_full_sft.py \
    --epochs 2 --batch_size 16 --learning_rate 1e-5 \
    --from_weight pretrain

# 使用 Muon 优化器 (DeepSeek V4 风格，节省 50% 显存)
python trainer/train_full_sft.py \
    --epochs 2 --batch_size 16 --learning_rate 1e-3 \
    --optimizer muon --from_weight pretrain

# 自定义配置
python trainer/train_full_sft.py \
    --hidden_size 768 --num_hidden_layers 8 --max_seq_len 768 \
    --from_weight pretrain --save_weight full_sft \
    --epochs 2 --batch_size 16 --learning_rate 1e-5
```

#### 第四步：可选训练

```bash
# LoRA 微调
python trainer/train_lora.py --epochs 6 --batch_size 16

# 知识蒸馏
python trainer/train_distillation.py \
    --alpha 0.5 --temperature 1.5 \
    --teacher_hidden_size 768 --student_hidden_size 768

# DPO 对齐
python trainer/train_dpo.py --epochs 1 --batch_size 4

# GRPO 组相对策略优化
python trainer/train_grpo.py

# PPO 近端策略优化
python trainer/train_ppo.py

# Agent RL 工具使用
python trainer/train_agent.py

# Meta-TTT 推理时训练
python trainer/train_meta_ttt.py

# 多模态 Omni SFT
python trainer/train_sft_omni.py
```

### 4.4 模型导出与部署

```bash
# 转换为 HuggingFace 格式
python scripts/convert_model.py

# 部署 OpenAI 兼容 API 服务
python scripts/serve_openai_api.py --port 8000
```

### 4.5 推理与对话

```bash
# LLM 评估
python eval_llm.py

# Web 对话演示 (LLM)
python scripts/web_demo.py

# Web 对话演示 (多模态)
python scripts/web_demo_omni.py

# API 对话
python scripts/chat_api.py
```

---

## 5. 高级特性

### 5.1 Muon 优化器

基于动量梯度 + Newton-Schulz 正交归一化，存储在 `trainer/trainer_utils.py`：

| 类 | 说明 |
|------|------|
| `newton_schulz_5(G)` | 将梯度矩阵投影到最近正交矩阵（三次多项式初始化 + 5 次 NS 迭代） |
| `Muon` | 纯 Muon 优化器，所有参数用 NS 归一化 |
| `MixedMuonAdamW` | 混合优化器：≥2D 大矩阵 → Muon，1D/小矩阵 → AdamW |
| `create_muon_optimizer(model)` | 工厂函数，返回 MixedMuonAdamW |

使用方式：`--optimizer muon`（见 [步骤三](#第三步sft-监督微调)）

### 5.2 跨层参数共享

通过 `layer_share_factor` 控制：
- `layer_share_factor=1`：每层独立参数（默认）
- `layer_share_factor=2`：每 2 层共享一组参数，总层数不变，唯一参数组减半

配置示例：
```json
{
  "hidden_size": 1024,
  "num_hidden_layers": 12,
  "layer_share_factor": 2
}
```

### 5.3 多 Token 预测 (MTP)

通过 `mtp_num_heads` 控制：
- `mtp_num_heads=0`：不启用（默认）
- `mtp_num_heads=2`：额外预测接下来 2 个 token

配置示例：
```json
{
  "mtp_num_heads": 2,
  "mtp_loss_weight": 0.1
}
```

### 5.4 In-Place TTT (推理时训练)

推理过程中对选定层的 FFN 权重进行在线更新：
- `ttt_enabled=True` 启用
- `ttt_chunk_size` 控制 chunk 大小（默认 512）
- `ttt_lr` 控制更新学习率（默认 1e-4）
- `ttt_layers` 指定启用层（`None`=最后 25%）

### 5.5 MHC + CSA 高级注意力

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

### 5.6 诚实性训练 (Honest Training)

- 检测模型"不知道"的场景（`trainer/honest_training.py`）
- 在 `know` vs `don't know` 场景间施加差别奖励
- 集成到 Agent RL (`train_agent.py`) 和 PPO/GRPO 训练中

### 5.7 Transformer 性能优化

MiniMind 的 Attention 层实现了三级路由，自动选择最快路径：

| 路径 | 条件 | 场景 | 加速效果 |
|------|------|------|----------|
| Flash Attention 2 | `flash-attn` 已安装 + 训练 | 训练 | 显存 O(N²)→O(N)，速度 2-3x |
| PyTorch SDPA | `torch>=2.0` + KV Cache | 推理 | 推理速度 2x |
| Manual Attention | Fallback | 兼容 | 基准 |

**KVCache 预分配**（`model/model_minimind.py` 中的 `KVCache` 类）：
- 推理时预分配固定大小 buffer，避免每步 `torch.cat` 的 O(n²) 显存分配
- `generate()` 自动使用 Prefill + Decode 模式：先一次处理完整 prompt，再逐 token decode
- 首 token 延迟降低 10x+

**安装 Flash Attention 2：**
```bash
pip install flash-attn --no-build-isolation
```

无需修改代码，安装后自动启用 FA2 路径。

### 5.8 运行测试

```bash
python tests/test_all.py
# 27 项测试覆盖：模型构建、前向、TTT、MHC+CSA、Muon、OPD、生成
```

---

## 6. 多模态 (MiniMind-O)

### 6.1 Thinker-Talker 架构

```
输入 (文本 / 图片 / 音频)
  ↓
[Encoder] SigLIP2 视觉 → MMVisionProjector
[Encoder] SenseVoice ASR → MMAudioProjector
  ↓
[Thinker] MiniMind Block × N → 多模态理解
  ↓ Bridge Layer
[Talker] MiniMind Block × K → 语音生成
  ↓ Mimi Decoder
输出 (语音)
```

### 6.2 编码器模型（通过 snapshot_download 按需下载）

| 编码器 | 用途 | 模型路径 |
|--------|------|----------|
| SenseVoiceSmall | 语音识别 (ASR) | `model/SenseVoiceSmall/` |
| SigLIP2 | 视觉理解 | `model/siglip2-base-p32-256-ve/` |
| Mimi | 音频编解码 | `model/mimi/` |
| CAMPPlus | 说话人识别 | `model/campplus/` |
| SileroVAD | 语音活动检测 | `model/vad/silero_vad.onnx` |

配置文件已放置在 `model/` 对应目录中，模型权重需从 HuggingFace/ModelScope 下载：

```python
from modelscope import snapshot_download

snapshot_download('iic/SenseVoiceSmall', local_dir='model/SenseVoiceSmall')
snapshot_download('iic/speech_campplus_sv_zh-cn_16k-common', local_dir='model/campplus')
# ... etc
```

### 6.3 多模态评估

```bash
python eval_omni.py --model_path ./out/llm_768.pth \
    --audio_dir ./dataset/eval_omni/ \
    --image_dir ./dataset/eval_omni/
```
