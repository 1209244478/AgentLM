<div align="center">
  <h3>MiniMind Agent — Training Language Models with Tool-Use Capabilities from Scratch</h3>
</div>

<div align="center">

[中文](./README.md) | English

</div>

An independent project extended from [MiniMind](https://github.com/jingyaogong/minimind), focused on **Agent workflow enhancement**: enabling small-parameter language models to learn multi-turn tool calling, structured reasoning, and task decomposition.

---

## Key Features

- **23 Standardized Tool Definitions**: Math calculation, weather query, exchange rate, file operations, database query, code execution, HTTP request, email, stock query, route planning, schedule management, text summarization, etc. — all managed in `trainer/agent_tools.py`
- **ReAct / Plan-Execute Reasoning Templates**: Switch via `--system_prompt_type`, supporting "Observe-Think-Act" and "Plan-Execute" reasoning paradigms
- **Multi-turn Tool Calling Training**: Agent RL training supports up to 5 rounds of tool interaction, with delayed reward covering tool legality, format closure, GT matching, and more
- **SFT Tool-Call Mixing**: `SFTDataset` supports mixing tool-call samples by ratio (`--tool_call_ratio`), no separate training needed
- **Automatic Data Generation**: `dataset/generate_agent_data.py` generates diverse agent training data with 10 multi-turn templates
- **119 Evaluation Test Cases**: Covering all 23 tools + 10 multi-turn scenarios + benchmark mode
- **Complete Training Pipeline**: Pretrain → SFT → RLAIF (PPO/GRPO/CISPO) → Agent RL, all core algorithms implemented from scratch
- **Advanced Architecture Modules**: In-Place TTT, cross-layer parameter sharing, MTP multi-token prediction, Muon optimizer, mHC+CSA+MSA attention
- **Full-Modal Extension**: MiniMind-O Thinker-Talker dual-path architecture, supporting text/audio/visual modalities

---

## Project Structure

```
minimind/
├── model/                           # Model definitions
│   ├── model_minimind.py            # Core LLM (Dense + MoE + TTT + MTP)
│   ├── model_advanced.py            # Advanced modules (mHC + CSA + MSA)
│   ├── model_omni.py                # Full-modal Thinker-Talker model
│   ├── model_lora.py                # LoRA fine-tuning
│   ├── minimind_rag.py              # RAG retrieval-augmented generation
│   └── tokenizer files
│
├── trainer/                         # Training scripts
│   ├── train_pretrain.py            # Pretraining
│   ├── train_full_sft.py            # SFT fine-tuning (supports --tool_call_ratio)
│   ├── train_lora.py                # LoRA fine-tuning
│   ├── train_dpo.py                 # DPO alignment
│   ├── train_grpo.py                # GRPO (supports CISPO loss)
│   ├── train_ppo.py                 # PPO
│   ├── train_agent.py               # Agent RL tool-use training
│   ├── agent_tools.py               # 23 tool definitions + ReAct/Plan-Execute templates
│   ├── train_distillation.py        # Knowledge distillation
│   ├── train_meta_ttt.py            # Meta-TTT
│   ├── train_sft_omni.py            # Multimodal SFT
│   ├── honest_training.py           # Honesty training rewards
│   ├── rollout_engine.py            # Rollout generation engine
│   └── trainer_utils.py             # Training utilities (incl. Muon optimizer)
│
├── dataset/                         # Datasets
│   ├── lm_dataset.py                # Text dataset (supports tool_call_ratio mixing)
│   ├── omni_dataset.py              # Multimodal dataset
│   ├── generate_agent_data.py       # Agent RL data generation script
│   └── eval_omni/                   # Multimodal evaluation data
│
├── scripts/                         # Utility scripts
│   ├── eval_toolcall.py             # Tool Call evaluation (119 test cases + benchmark)
│   ├── chat_api.py                  # API chat interface
│   ├── serve_openai_api.py          # OpenAI-compatible API server
│   ├── web_demo.py / web_demo_omni.py
│   └── convert_model.py / convert_omni.py
│
├── tests/                           # Tests
│   ├── test_all.py                  # 27 unified tests
│   ├── test_new_arch.py             # 54 new architecture tests
│   └── benchmark_performance.py     # 19 performance benchmarks
│
├── webui/                           # Streamlit Web UI
├── mini-RAG/                        # External RAG system (LightRAG)
└── minimind-3/                      # Model configs & Tokenizer
```

---

## Quick Start

### Environment Setup

```bash
pip install -r requirements.txt
# Optional: install Flash Attention 2 for faster training
pip install flash-attn --no-build-isolation
```

### Dataset Download

Training data available at: [ModelScope](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files) | [HuggingFace](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main)

Place downloaded files in `./dataset/`. Core files:

| File | Purpose | Size |
|------|---------|------|
| `pretrain_t2t_mini.jsonl` | Pretraining | ~1.2 GB |
| `sft_t2t_mini.jsonl` | SFT fine-tuning | ~1.6 GB |
| `rlaif.jsonl` | RLAIF training | ~24 MB |
| `agent_rl.jsonl` | Agent RL training | ~86 MB |
| `agent_rl_math.jsonl` | Agent RL math | ~18 MB |

### Training Pipeline

> All training scripts run under `cd ./trainer`

**1. Pretraining**

```bash
python train_pretrain.py
```

**2. SFT Fine-tuning (with tool-call mixing)**

```bash
python train_full_sft.py
# Mix 20% tool-call samples
python train_full_sft.py --tool_call_ratio 0.2 --tool_call_data_path ../dataset/agent_rl.jsonl
```

**3. RLAIF (optional)**

```bash
# GRPO
python train_grpo.py
# PPO
python train_ppo.py
# CISPO (switch loss_type on top of GRPO)
python train_grpo.py --loss_type cispo
```

**4. Agent RL Training**

```bash
# Default torch rollout
python train_agent.py
# ReAct reasoning template
python train_agent.py --system_prompt_type react
# Plan-Execute reasoning template
python train_agent.py --system_prompt_type plan_execute
# Specify model config
python train_agent.py --config ../minimind-3/config_8.json
# sglang rollout (start sglang server first)
python train_agent.py --rollout_engine sglang --sglang_base_url http://localhost:8998
```

**5. Generate Agent Training Data**

```bash
python dataset/generate_agent_data.py --output dataset/agent_rl.jsonl --num_samples 10000
python dataset/generate_agent_data.py --num_samples 5000 --multi_turn_ratio 0.6
```

### Evaluation

```bash
# LLM evaluation
python eval_llm.py --weight full_sft

# Tool Call evaluation
python scripts/eval_toolcall.py --weight agent
# Full benchmark (119 test cases)
python scripts/eval_toolcall.py --weight agent --benchmark 1
```

---

## Agent Workflow Details

### Tool Definitions

`trainer/agent_tools.py` is the single source of truth for tools, providing:

| Component | Description |
|-----------|-------------|
| `TOOL_MAP` | Name, description, and parameter schema for 23 tools |
| `CHECK_ARGS` | Argument validation function for each tool |
| `MOCK_RESULTS` | Mock execution function for each tool (used during training) |
| `parse_tool_calls()` | Parse tool calls from model output |
| `execute_tool()` | Execute tool and return result (Windows compatible) |
| `SYSTEM_PROMPTS` | 3 system prompt templates (default / react / plan_execute) |

### 23 Tools Overview

| Category | Tools |
|----------|-------|
| Math | `calculate_math`, `unit_converter`, `random_number` |
| Information | `get_current_weather`, `get_current_time`, `get_exchange_rate`, `get_stock_price`, `get_location_info` |
| Text | `translate_text`, `summarize_text`, `text_length` |
| File | `read_file`, `write_file`, `list_directory` |
| Data & Code | `sql_query`, `python_exec`, `http_request` |
| Communication | `send_email` |
| Schedule | `create_event`, `check_schedule` |
| Navigation | `get_route` |
| Utility | `countdown_timer` |

### Reasoning Templates

**ReAct (Observe-Think-Act)**:

```
System: You are an intelligent assistant using the ReAct pattern.
For each step, think in the following format:
Thought: Analyze the current situation
Action: Call a tool
Observation: Tool returns result
... (repeat until final answer)
Answer: Final answer
```

**Plan-Execute**:

```
System: You are an intelligent assistant using the Plan-Execute pattern.
First make a plan, then execute step by step:
Plan: Create a step-by-step plan
Step 1: Execute first step
Step 2: Execute second step
... (adjust plan based on intermediate results)
Answer: Final answer
```

### Agent RL Training Flow

```
rollout batch → calculate rewards → policy update
```

Reward function covers multiple dimensions:

$$R(\tau) = R_{\text{answer}} + R_{\text{tool}} + R_{\text{format}} + R_{\text{rm}} - R_{\text{unfinished}}$$

- `R_answer`: Final answer matches GT
- `R_tool`: Tool call legality
- `R_format`: Format closure (tool_calls → tool observation)
- `R_rm`: Reward Model score
- `R_unfinished`: Unfinished penalty

---

## Model Architecture

### Base Model (Dense + MoE)

Transformer Decoder-Only architecture aligned with `Qwen3 / Qwen3-MoE` ecosystem:

- Pre-Norm + RMSNorm + SwiGLU
- RoPE rotary position encoding (with YaRN extrapolation support)
- GQA: `q_heads=8, kv_heads=4`
- MoE: `4 experts / top-1 routing` (no shared expert)

| Config | hidden_size | layers | Parameters |
|--------|-------------|--------|------------|
| minimind-3 | 768 | 8 | ~64M |
| minimind-3-moe | 768 | 8 | ~198M-A64M |
| minimind-5 | 1024 | 12 | ~186M |
| minimind-6 | 1280 | 20 | ~500M |
| minimind-7 | 2048 | 26 | ~1.5B |
| minimind-8 | 4096 | 32 | ~6.1B |

### Advanced Modules

| Module | File | Description |
|--------|------|-------------|
| In-Place TTT | `model_minimind.py` | Test-time FFN weight update, `use_ttt=True` |
| Layer Sharing | `model_minimind.py` | `layer_share_factor` controls sharing every N layers |
| MTP | `model_minimind.py` | `mtp_num_heads` prediction heads |
| mHC + CSA | `model_advanced.py` | DeepSeek V4 style manifold-constrained hyper-connections + compressed sparse attention |
| MSA | `model_advanced.py` | MiniMax M3 style sparse attention |
| Muon Optimizer | `trainer_utils.py` | Newton-Schulz orthogonal normalization, `--optimizer muon` |

### Attention Three-Level Routing

| Path | Condition | Scenario |
|------|-----------|----------|
| Flash Attention 2 | `flash-attn` installed | Training |
| PyTorch SDPA | `torch>=2.0` + KV Cache | Inference |
| Manual Attention | Fallback | Compatibility |

---

## RLAIF Training Algorithms

### PPO

$$\mathcal{L}_{PPO} = -\mathbb{E}\left[\min(r_t \cdot A_t, \text{clip}(r_t, 1-\varepsilon, 1+\varepsilon) \cdot A_t)\right] + \beta \cdot \mathbb{E}[\text{KL}]$$

Actor + Critic dual networks, GAE advantage estimation.

### GRPO

$$\mathcal{L}_{GRPO} = -\mathbb{E}\left[\min(r_t \cdot A_t, \mathrm{clip}(r_t, 1-\varepsilon, 1+\varepsilon) \cdot A_t) - \beta \cdot \text{KL}_t\right]$$

Group-relative advantage $A_t = \frac{R - \mu_{group}}{\sigma_{group}}$, no Critic network needed.

### CISPO

$$\mathcal{L}_{CISPO} = -\mathbb{E}\left[\min(r_t, \varepsilon_{max}) \cdot A_t \cdot \log \pi_\theta(a_t|s) - \beta \cdot \text{KL}_t\right]$$

Clipped weight × log probability, avoiding gradient truncation. Set `--loss_type cispo` in `train_grpo.py`.

---

## Testing

```bash
# Functional tests (27 items)
python tests/test_all.py

# New architecture tests (54 items)
python tests/test_new_arch.py

# Performance benchmarks (19 items, including long-context stress tests)
python tests/benchmark_performance.py
```

---

## Changelog

<details>
<summary>2026-06</summary>

- Agent workflow enhancement: 23 standardized tools, ReAct/Plan-Execute templates, 5-round multi-turn interaction
- Added `trainer/agent_tools.py` shared tool module
- Added `dataset/generate_agent_data.py` data generation script (10K samples)
- SFTDataset supports `tool_call_ratio` tool-call mixing
- `eval_toolcall.py` expanded to 119 test cases + benchmark mode

</details>

<details>
<summary>2026-05</summary>

- MiniMind-5 as default training config (~186M)
- In-Place TTT, cross-layer parameter sharing, MTP multi-token prediction
- Muon optimizer, mHC+CSA+MSA advanced attention
- Meta-TTT meta-learning, honesty training & confidence calibration
- Full-modal Omni model (Thinker-Talker dual-path)
- Dataset expansion: MINT-1T ArXiv merged data

</details>

<details>
<summary>2026-04</summary>

- Released minimind-3 / minimind-3-moe
- Architecture aligned with Qwen3 / Qwen3-MoE ecosystem
- Native Agentic RL training script
- Tokenizer update (BPE + ByteLevel + tool-call tokens)

</details>

---

## Acknowledgments

This project is built on [MiniMind](https://github.com/jingyaogong/minimind). Thanks to the original author [Jingyao Gong](https://github.com/jingyaogong).

MiniMind references the following projects and papers:

- [llama3](https://github.com/meta-llama/llama3), [llama2.c](https://github.com/karpathy/llama2.c)
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434)
- [TTT](https://arxiv.org/abs/2407.04620), [MAML](https://arxiv.org/abs/1703.03400)

---

## Citation

```bibtex
@misc{minimind,
  title = {MiniMind: Train a Tiny LLM from Scratch},
  author = {Jingyao Gong},
  year = {2024},
  url = {https://github.com/jingyaogong/minimind}
}
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).
