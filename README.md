<div align="center">
  <h3>MiniMind Agent — 从零训练具备工具使用能力的语言模型</h3>
</div>

<div align="center">

中文 | [English](./README_en.md)

</div>

基于 [MiniMind](https://github.com/jingyaogong/minimind) 扩展的独立项目，聚焦于 **Agent 工作流增强**：让小参数语言模型学会多轮工具调用、结构化推理与任务分解。

---

## 核心特性

- **23 个标准化工具定义**：数学计算、天气查询、汇率转换、文件操作、数据库查询、代码执行、HTTP 请求、邮件发送、股票查询、路线规划、日程管理、文本摘要等，统一管理于 `trainer/agent_tools.py`
- **ReAct / Plan-Execute 推理模板**：通过 `--system_prompt_type` 切换，支持"观察-思考-行动"和"规划-执行"两种推理范式
- **多轮工具调用训练**：Agent RL 训练支持最多 5 轮工具交互，延迟奖励机制覆盖工具合法性、格式闭合、GT 命中等多维度
- **SFT 工具调用混合**：`SFTDataset` 支持按比例混合工具调用样本（`--tool_call_ratio`），无需额外独立训练
- **自动数据生成**：`dataset/generate_agent_data.py` 可生成多样化 Agent 训练数据，含 10 种多轮模板
- **119 个评估测试用例**：覆盖全部 23 个工具 + 10 个多轮场景 + 基准评测模式
- **完整训练链路**：Pretrain → SFT → RLAIF (PPO/GRPO/CISPO) → Agent RL，所有核心算法从 0 实现
- **高级架构模块**：In-Place TTT、跨层参数共享、MTP 多 Token 预测、Muon 优化器、mHC+CSA+MSA 注意力
- **全模态扩展**：MiniMind-O Thinker-Talker 双路径架构，支持文本/音频/视觉三模态

---

## 项目结构

```
minimind/
├── model/                           # 模型定义
│   ├── model_minimind.py            # 核心 LLM（Dense + MoE + TTT + MTP）
│   ├── model_advanced.py            # 高级模块（mHC + CSA + MSA）
│   ├── model_omni.py                # 全模态 Thinker-Talker 模型
│   ├── model_lora.py                # LoRA 微调
│   ├── minimind_rag.py              # RAG 检索增强
│   └── tokenizer 相关文件
│
├── trainer/                         # 训练脚本
│   ├── train_pretrain.py            # 预训练
│   ├── train_full_sft.py            # SFT 微调（支持 --tool_call_ratio）
│   ├── train_lora.py                # LoRA 微调
│   ├── train_dpo.py                 # DPO 对齐
│   ├── train_grpo.py                # GRPO（支持 CISPO loss）
│   ├── train_ppo.py                 # PPO
│   ├── train_agent.py               # Agent RL 工具使用训练
│   ├── agent_tools.py               # 23 个工具定义 + ReAct/Plan-Execute 模板
│   ├── train_distillation.py        # 知识蒸馏
│   ├── train_meta_ttt.py            # Meta-TTT
│   ├── train_sft_omni.py            # 多模态 SFT
│   ├── honest_training.py           # 诚实训练奖励
│   ├── rollout_engine.py            # Rollout 生成引擎
│   └── trainer_utils.py             # 训练工具集（含 Muon 优化器）
│
├── dataset/                         # 数据集
│   ├── lm_dataset.py                # 文本数据集（支持 tool_call_ratio 混合）
│   ├── omni_dataset.py              # 多模态数据集
│   ├── generate_agent_data.py       # Agent RL 数据生成脚本
│   └── eval_omni/                   # 多模态评估数据
│
├── scripts/                         # 工具脚本
│   ├── eval_toolcall.py             # Tool Call 评估（119 测试用例 + 基准）
│   ├── chat_api.py                  # API 对话接口
│   ├── serve_openai_api.py          # OpenAI 兼容 API 服务
│   ├── web_demo.py / web_demo_omni.py
│   └── convert_model.py / convert_omni.py
│
├── tests/                           # 测试
│   ├── test_all.py                  # 27 项统一测试
│   ├── test_new_arch.py             # 54 项新架构测试
│   └── benchmark_performance.py     # 19 项性能基准
│
├── webui/                           # Streamlit Web UI
├── mini-RAG/                        # 外挂 RAG 系统 (LightRAG)
└── minimind-3/                      # 模型配置与 Tokenizer
```

---

## 快速开始

### 环境准备

```bash
pip install -r requirements.txt
# 可选：安装 Flash Attention 2 加速训练
pip install flash-attn --no-build-isolation
```

### 数据集下载

训练数据集下载地址：[ModelScope](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files) | [HuggingFace](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main)

将下载的文件放到 `./dataset/` 目录下，核心文件：

| 文件 | 用途 | 大小 |
|------|------|------|
| `pretrain_t2t_mini.jsonl` | 预训练 | ~1.2 GB |
| `sft_t2t_mini.jsonl` | SFT 微调 | ~1.6 GB |
| `rlaif.jsonl` | RLAIF 训练 | ~24 MB |
| `agent_rl.jsonl` | Agent RL 训练 | ~86 MB |
| `agent_rl_math.jsonl` | Agent RL 数学 | ~18 MB |

### 训练流程

> 所有训练脚本均在 `cd ./trainer` 目录下执行

**1. 预训练**

```bash
python train_pretrain.py
```

**2. SFT 微调（含工具调用混合）**

```bash
python train_full_sft.py
# 混合 20% 工具调用样本
python train_full_sft.py --tool_call_ratio 0.2 --tool_call_data_path ../dataset/agent_rl.jsonl
```

**3. RLAIF（可选）**

```bash
# GRPO
python train_grpo.py
# PPO
python train_ppo.py
# CISPO（在 GRPO 基础上切换 loss_type）
python train_grpo.py --loss_type cispo
```

**4. Agent RL 训练**

```bash
# 默认 torch rollout
python train_agent.py
# ReAct 推理模板
python train_agent.py --system_prompt_type react
# Plan-Execute 推理模板
python train_agent.py --system_prompt_type plan_execute
# 指定模型配置
python train_agent.py --config ../minimind-3/config_8.json
# sglang rollout（需先启动 sglang server）
python train_agent.py --rollout_engine sglang --sglang_base_url http://localhost:8998
```

**5. 生成 Agent 训练数据**

```bash
python dataset/generate_agent_data.py --output dataset/agent_rl.jsonl --num_samples 10000
python dataset/generate_agent_data.py --num_samples 5000 --multi_turn_ratio 0.6
```

### 评估

```bash
# LLM 评估
python eval_llm.py --weight full_sft

# Tool Call 评估
python scripts/eval_toolcall.py --weight agent
# 完整基准测试（119 个测试用例）
python scripts/eval_toolcall.py --weight agent --benchmark 1
```

---

## Agent 工作流详解

### 工具定义

`trainer/agent_tools.py` 是工具的单一来源，提供：

| 组件 | 说明 |
|------|------|
| `TOOL_MAP` | 23 个工具的名称、描述、参数 schema 定义 |
| `CHECK_ARGS` | 每个工具的参数校验函数 |
| `MOCK_RESULTS` | 每个工具的模拟执行函数（训练时使用） |
| `parse_tool_calls()` | 从模型输出中解析工具调用 |
| `execute_tool()` | 执行工具并返回结果（兼容 Windows） |
| `SYSTEM_PROMPTS` | 3 种系统提示模板（default / react / plan_execute） |

### 23 个工具一览

| 类别 | 工具 |
|------|------|
| 数学 | `calculate_math`, `unit_converter`, `random_number` |
| 信息查询 | `get_current_weather`, `get_current_time`, `get_exchange_rate`, `get_stock_price`, `get_location_info` |
| 文本处理 | `translate_text`, `summarize_text`, `text_length` |
| 文件操作 | `read_file`, `write_file`, `list_directory` |
| 数据与代码 | `sql_query`, `python_exec`, `http_request` |
| 通信 | `send_email` |
| 日程 | `create_event`, `check_schedule` |
| 导航 | `get_route` |
| 工具 | `countdown_timer` |

### 推理模板

**ReAct（观察-思考-行动）**：

```
System: 你是一个智能助手，使用 ReAct 模式解决问题。
每一步请按以下格式思考：
Thought: 分析当前情况
Action: 调用工具
Observation: 工具返回结果
...（重复直到得出最终答案）
Answer: 最终回答
```

**Plan-Execute（规划-执行）**：

```
System: 你是一个智能助手，使用 Plan-Execute 模式解决问题。
先制定计划，再逐步执行：
Plan: 制定步骤计划
Step 1: 执行第一步
Step 2: 执行第二步
...（根据中间结果调整计划）
Answer: 最终回答
```

### Agent RL 训练流程

```
rollout batch → calculate rewards → policy update
```

奖励函数覆盖多维度：

$$R(\tau) = R_{\text{answer}} + R_{\text{tool}} + R_{\text{format}} + R_{\text{rm}} - R_{\text{unfinished}}$$

- `R_answer`：最终回答与 GT 匹配
- `R_tool`：工具调用合法性
- `R_format`：格式闭合（tool_calls → tool observation）
- `R_rm`：Reward Model 分数
- `R_unfinished`：未完成惩罚

---

## 模型架构

### 基础模型（Dense + MoE）

对齐 `Qwen3 / Qwen3-MoE` 生态的 Transformer Decoder-Only 结构：

- Pre-Norm + RMSNorm + SwiGLU
- RoPE 旋转位置编码（支持 YaRN 外推）
- GQA：`q_heads=8, kv_heads=4`
- MoE：`4 experts / top-1 routing`（去除 shared expert）

| 配置 | hidden_size | layers | 参数量 |
|------|-------------|--------|--------|
| minimind-3 | 768 | 8 | ~64M |
| minimind-3-moe | 768 | 8 | ~198M-A64M |
| minimind-5 | 1024 | 12 | ~186M |
| minimind-6 | 1280 | 20 | ~500M |
| minimind-7 | 2048 | 26 | ~1.5B |
| minimind-8 | 4096 | 32 | ~6.1B |

### 高级模块

| 模块 | 文件 | 说明 |
|------|------|------|
| In-Place TTT | `model_minimind.py` | 推理时 FFN 权重更新，`use_ttt=True` |
| 跨层参数共享 | `model_minimind.py` | `layer_share_factor` 控制每 N 层共享 |
| MTP 多 Token 预测 | `model_minimind.py` | `mtp_num_heads` 个预测头 |
| mHC + CSA | `model_advanced.py` | DeepSeek V4 风格流形约束 + 压缩稀疏注意力 |
| MSA | `model_advanced.py` | MiniMax M3 风格稀疏注意力 |
| Muon 优化器 | `trainer_utils.py` | Newton-Schulz 正交归一化，`--optimizer muon` |

### Attention 三级路由

| 路径 | 条件 | 场景 |
|------|------|------|
| Flash Attention 2 | `flash-attn` 已安装 | 训练 |
| PyTorch SDPA | `torch>=2.0` + KV Cache | 推理 |
| Manual Attention | Fallback | 兼容 |

---

## RLAIF 训练算法

### PPO

$$\mathcal{L}_{PPO} = -\mathbb{E}\left[\min(r_t \cdot A_t, \text{clip}(r_t, 1-\varepsilon, 1+\varepsilon) \cdot A_t)\right] + \beta \cdot \mathbb{E}[\text{KL}]$$

Actor + Critic 双网络，GAE 优势估计。

### GRPO

$$\mathcal{L}_{GRPO} = -\mathbb{E}\left[\min(r_t \cdot A_t, \mathrm{clip}(r_t, 1-\varepsilon, 1+\varepsilon) \cdot A_t) - \beta \cdot \text{KL}_t\right]$$

组内相对优势 $A_t = \frac{R - \mu_{group}}{\sigma_{group}}$，无需 Critic 网络。

### CISPO

$$\mathcal{L}_{CISPO} = -\mathbb{E}\left[\min(r_t, \varepsilon_{max}) \cdot A_t \cdot \log \pi_\theta(a_t|s) - \beta \cdot \text{KL}_t\right]$$

裁剪权重 × log 概率，避免梯度截断。在 `train_grpo.py` 中设置 `--loss_type cispo` 即可。

---

## 测试

```bash
# 功能测试（27 项）
python tests/test_all.py

# 新架构测试（54 项）
python tests/test_new_arch.py

# 性能基准（19 项，含长上下文压力测试）
python tests/benchmark_performance.py
```

---

## 更新日志

<details>
<summary>2026-06</summary>

- Agent 工作流全面增强：23 个标准化工具、ReAct/Plan-Execute 模板、5 轮多轮交互
- 新增 `trainer/agent_tools.py` 共享工具模块
- 新增 `dataset/generate_agent_data.py` 数据生成脚本（10K 条）
- SFTDataset 支持 `tool_call_ratio` 工具调用混合训练
- `eval_toolcall.py` 扩展到 119 个测试用例 + 基准评测模式

</details>

<details>
<summary>2026-05</summary>

- MiniMind-5 成为默认训练配置（~186M）
- In-Place TTT、跨层参数共享、MTP 多 Token 预测
- Muon 优化器、mHC+CSA+MSA 高级注意力
- Meta-TTT 元学习、诚实训练与置信度校准
- 全模态 Omni 模型（Thinker-Talker 双路径）
- 数据集扩展：MINT-1T ArXiv 合并数据

</details>

<details>
<summary>2026-04</summary>

- 发布 minimind-3 / minimind-3-moe
- 结构对齐 Qwen3 / Qwen3-MoE 生态
- 原生 Agentic RL 训练脚本
- Tokenizer 更新（BPE + ByteLevel + 工具调用标记）

</details>

---

## 致谢

本项目基于 [MiniMind](https://github.com/jingyaogong/minimind) 开发，感谢原作者 [Jingyao Gong](https://github.com/jingyaogong) 的工作。

MiniMind 参考了以下项目与论文：

- [llama3](https://github.com/meta-llama/llama3), [llama2.c](https://github.com/karpathy/llama2.c)
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434)
- [TTT](https://arxiv.org/abs/2407.04620), [MAML](https://arxiv.org/abs/1703.03400)

---

## 引用

```bibtex
@misc{minimind,
  title = {MiniMind: Train a Tiny LLM from Scratch},
  author = {Jingyao Gong},
  year = {2024},
  url = {https://github.com/jingyaogong/minimind}
}
```

## 开源协议

本项目采用 [Apache License 2.0](LICENSE) 开源协议。
