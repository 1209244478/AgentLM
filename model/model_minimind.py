import math, torch, torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

try:
    from flash_attn import flash_attn_func
    _flash_attn2_available = True
except ImportError:
    _flash_attn2_available = False

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Config
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        ### MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)
        ### Cross-layer parameter sharing
        self.layer_share_factor = kwargs.get("layer_share_factor", 1)  # 每 N 层共享一组参数，1=不共享
        ### Multi-token prediction
        self.mtp_num_heads = kwargs.get("mtp_num_heads", 0)  # 额外预测头数量，0=不启用
        self.mtp_loss_weight = kwargs.get("mtp_loss_weight", 0.1)  # MTP 辅助损失权重
        ### In-Place TTT
        self.ttt_enabled = kwargs.get("ttt_enabled", False)  # 是否启用推理时训练
        self.ttt_lr = kwargs.get("ttt_lr", 1e-4)  # TTT 学习率
        self.ttt_chunk_size = kwargs.get("ttt_chunk_size", 512)  # TTT chunk 大小
        self.ttt_layers = kwargs.get("ttt_layers", None)  # 哪些层启用 TTT，None=最后25%
        ### Attention type: "standard" | "linear" | "alibi"
        self.attention_type = kwargs.get("attention_type", "standard")
        ### Parallel Attention + FFN (PaLM style)
        self.parallel_attn_ffn = kwargs.get("parallel_attn_ffn", False)
        ### LoRA-FFN: 在 FFN 内部加低秩旁路
        self.lora_ffn = kwargs.get("lora_ffn", False)
        self.lora_ffn_r = kwargs.get("lora_ffn_r", 8)
        ### Mamba hybrid: 底层 Mamba + 顶层 Attention
        self.mamba_hybrid = kwargs.get("mamba_hybrid", False)
        self.mamba_ratio = kwargs.get("mamba_ratio", 0.5)  # 底层 Mamba 占比

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Model
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_float = x.float()
        x_normed = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x_normed).type_as(x)

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None: # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    half = q.shape[-1] // 2
    q1, q2 = q[..., :half], q[..., half:]
    k1, k2 = k[..., :half], k[..., half:]
    q_embed = (q1 * cos[..., :half] - q2 * sin[..., :half]).to(q.dtype)
    q_embed = torch.cat([q_embed, (q2 * cos[..., :half] + q1 * sin[..., :half]).to(q.dtype)], dim=-1)
    k_embed = (k1 * cos[..., :half] - k2 * sin[..., :half]).to(k.dtype)
    k_embed = torch.cat([k_embed, (k2 * cos[..., :half] + k1 * sin[..., :half]).to(k.dtype)], dim=-1)
    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim))


class KVCache:
    """Pre-allocated KV Cache for autoregressive generation.

    Avoids torch.cat on every step by writing into a fixed-size buffer.
    Usage: past_kv = KVCache(n_layers, bsz, max_len, n_kv_heads, head_dim, device, dtype)
    """

    def __init__(self, n_layers, bsz, max_len, n_kv_heads, head_dim, device, dtype):
        self.n_layers = n_layers
        self.max_len = max_len
        k_shape = (bsz, max_len, n_kv_heads, head_dim)
        self.k_cache = [torch.zeros(k_shape, device=device, dtype=dtype) for _ in range(n_layers)]
        self.v_cache = [torch.zeros(k_shape, device=device, dtype=dtype) for _ in range(n_layers)]
        self.len = [0] * n_layers

    def update(self, layer_idx, new_k, new_v):
        """Insert new_k/v at the current position, return the full cached K/V."""
        cur_len = self.len[layer_idx]
        new_len = cur_len + new_k.shape[1]
        self.k_cache[layer_idx][:, cur_len:new_len].copy_(new_k)
        self.v_cache[layer_idx][:, cur_len:new_len].copy_(new_v)
        self.len[layer_idx] = new_len
        return self.k_cache[layer_idx][:, :new_len], self.v_cache[layer_idx][:, :new_len]

    def get(self, layer_idx):
        """Get cached K/V up to current length."""
        cur_len = self.len[layer_idx]
        return self.k_cache[layer_idx][:, :cur_len], self.v_cache[layer_idx][:, :cur_len]

    def to_legacy_format(self):
        """Convert to the list-of-tuples format expected by existing code."""
        return [(self.k_cache[i][:, :self.len[i]], self.v_cache[i][:, :self.len[i]]) for i in range(self.n_layers)]

class Attention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn
        self.use_flash_attn2 = _flash_attn2_available and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        kv_len = xk.shape[1]
        xq_4d = xq.transpose(1, 2)
        xk_4d = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv_4d = repeat_kv(xv, self.n_rep).transpose(1, 2)

        # Flash Attention 2 path (training: full seq, inference: single token with kv cache)
        if self.use_flash_attn2 and seq_len > 1 and self.training:
            output = flash_attn_func(
                xq_4d, xk_4d, xv_4d,
                dropout_p=self.dropout if self.training else 0.0,
                causal=True,
            )
        # PyTorch SDPA path (supports KV cache via attention_mask)
        elif self.flash:
            attn_mask = None
            if attention_mask is not None and not torch.all(attention_mask == 1):
                attn_mask = attention_mask[:, None, None, :].to(dtype=xq_4d.dtype)
                attn_mask = (1.0 - attn_mask) * torch.finfo(xq_4d.dtype).min
            is_causal = self.is_causal and (seq_len > 1) and (past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1))
            output = F.scaled_dot_product_attention(
                xq_4d, xk_4d, xv_4d,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
            )
        # Fallback: manual attention
        else:
            scores = (xq_4d @ xk_4d.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if self.is_causal and seq_len > 1:
                causal_mask = torch.triu(torch.full((seq_len, kv_len), float("-inf"), device=scores.device), diagonal=kv_len - seq_len + 1)
                scores[:, :, -seq_len:, :] += causal_mask[:, :kv_len]
            if attention_mask is not None:
                scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq_4d)) @ xv_4d

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv

class LinearAttention(nn.Module):
    """线性注意力：用 ELU+1 特征映射替代 softmax，复杂度 O(N)

    核心思想：φ(Q)(φ(K)^T V) 先算 K^T V (d×d)，再乘 Q，避免 N×N 注意力矩阵。
    因果掩码通过 cumulative sum 实现。
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.eps = config.rms_norm_eps

    @staticmethod
    def _feature_map(x):
        return F.elu(x) + 1

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape
        xq = self.q_proj(x).view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = self.k_proj(x).view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = self.v_proj(x).view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        xq = self._feature_map(xq)
        xk = self._feature_map(xk)
        if self.n_rep > 1:
            xk = repeat_kv(xk, self.n_rep)
            xv = repeat_kv(xv, self.n_rep)
        total_len = xk.shape[1]
        xq_t = xq.transpose(1, 2)
        xk_t = xk.transpose(1, 2)
        xv_t = xv.transpose(1, 2)
        if past_key_value is not None:
            kv_global = torch.einsum('bhsd,bhsm->bhdm', xk_t, xv_t)
            k_sum_global = xk_t.sum(dim=2)
            output = torch.einsum('bhsd,bhdm->bhsm', xq_t, kv_global)
            normalizer = torch.einsum('bhsd,bhsd->bhs', xq_t, k_sum_global.unsqueeze(2)).unsqueeze(-1)
            normalizer = normalizer.clamp(min=self.eps)
            output = output / normalizer
        else:
            kv_pairs = torch.einsum('bhsd,bhsm->bhsdm', xk_t, xv_t)
            kv_cumsum = kv_pairs.cumsum(dim=2)
            k_sum = xk_t.cumsum(dim=2)
            output = torch.einsum('bhsd,bhsdm->bhsm', xq_t, kv_cumsum)
            normalizer = torch.einsum('bhsd,bhsd->bhs', xq_t, k_sum).unsqueeze(-1)
            normalizer = normalizer.clamp(min=self.eps)
            output = output / normalizer
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class ALiBiAttention(nn.Module):
    """ALiBi 注意力：用线性偏置替代位置编码，天然支持长度外推

    不使用 RoPE，而是在注意力分数上加上与距离成正比的偏置。
    偏置斜率固定为 2^(-8/n_heads * i)，无需学习。
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn
        slopes = 2.0 ** (-8.0 / self.n_local_heads * torch.arange(1, self.n_local_heads + 1))
        self.register_buffer("alibi_slopes", slopes.view(1, self.n_local_heads, 1, 1), persistent=False)

    def _get_alibi_bias(self, seq_len_q, seq_len_k, device):
        dists = torch.arange(seq_len_k, device=device).float().unsqueeze(0) \
                - torch.arange(seq_len_q, device=device).float().unsqueeze(1)
        alibi = -self.alibi_slopes * dists.abs().unsqueeze(0)
        causal_mask = torch.triu(torch.ones(seq_len_q, seq_len_k, device=device), diagonal=seq_len_k - seq_len_q + 1).bool()
        alibi = alibi.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        return alibi

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape
        xq = self.q_proj(x).view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = self.k_proj(x).view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = self.v_proj(x).view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        kv_len = xk.shape[1]
        xq_4d = xq.transpose(1, 2)
        xk_4d = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv_4d = repeat_kv(xv, self.n_rep).transpose(1, 2)
        alibi_bias = self._get_alibi_bias(seq_len, kv_len, xq_4d.device)
        if attention_mask is not None and not torch.all(attention_mask == 1):
            pad_mask = (1.0 - attention_mask[:, None, None, :]).to(dtype=xq_4d.dtype) * torch.finfo(xq_4d.dtype).min
            alibi_bias = alibi_bias + pad_mask
        if self.flash:
            output = F.scaled_dot_product_attention(
                xq_4d, xk_4d, xv_4d,
                attn_mask=alibi_bias,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
        else:
            scores = (xq_4d @ xk_4d.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores + alibi_bias
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq_4d)) @ xv_4d
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class TTTFeedForward(FeedForward):
    """支持 In-Place TTT 的 FeedForward 层

    推理时将 down_proj 权重作为可更新的 fast weights，
    通过 next-token prediction 损失做梯度更新，将上下文信息压缩进权重。
    参考: In-Place Test-Time Training (ICLR 2026 Oral)
    """

    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__(config, intermediate_size)
        self.ttt_enabled = False
        self.ttt_lr = config.ttt_lr
        self.ttt_chunk_size = config.ttt_chunk_size
        # 保存初始权重用于重置
        self._W0 = None
        # TTT 自监督用的轻量线性头：将 hidden_size 映射到 hidden_size
        # 用于在 FFN 层内部构建 next-token prediction 信号
        self.ttt_predictor = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def enable_ttt(self, lr=None, chunk_size=None):
        """启用推理时训练"""
        self.ttt_enabled = True
        if lr is not None:
            self.ttt_lr = lr
        if chunk_size is not None:
            self.ttt_chunk_size = chunk_size
        # 保存初始权重快照（包括 down_proj 和 ttt_predictor）
        self._W0 = self.down_proj.weight.data.clone()
        self._predictor_W0 = self.ttt_predictor.weight.data.clone()

    def disable_ttt(self):
        """禁用推理时训练，恢复初始权重"""
        self.ttt_enabled = False
        if self._W0 is not None:
            self.down_proj.weight.data.copy_(self._W0)
            self._W0 = None
        if hasattr(self, '_predictor_W0') and self._predictor_W0 is not None:
            self.ttt_predictor.weight.data.copy_(self._predictor_W0)
            self._predictor_W0 = None

    def reset_ttt_weights(self):
        """重置 TTT 权重到初始值（不改变 ttt_enabled 状态）"""
        if self._W0 is not None:
            self.down_proj.weight.data.copy_(self._W0)
        if hasattr(self, '_predictor_W0') and self._predictor_W0 is not None:
            self.ttt_predictor.weight.data.copy_(self._predictor_W0)

    def forward(self, x, use_ttt=False):
        """前向传播

        Args:
            x: 输入 hidden_states [batch, seq_len, hidden_size]
            use_ttt: 是否在本次前向中执行 TTT 更新（仅推理时有效）
        """
        # 计算中间激活
        Z = self.act_fn(self.gate_proj(x)) * self.up_proj(x)

        if use_ttt and self.ttt_enabled and not self.training:
            # In-Place TTT: 按 chunk 更新 down_proj 权重
            batch_size, seq_len, _ = Z.shape
            chunk_size = self.ttt_chunk_size
            outputs = []

            for start in range(0, seq_len, chunk_size):
                end = min(start + chunk_size, seq_len)
                Z_chunk = Z[:, start:end, :]

                # Apply: 用当前 down_proj 权重计算输出（不需要梯度）
                with torch.no_grad():
                    Y_chunk = F.linear(Z_chunk, self.down_proj.weight, self.down_proj.bias)
                outputs.append(Y_chunk)

                # Update: 用 next-token prediction 自监督更新 down_proj 权重
                # 核心思想：Y_chunk[t] 经 ttt_predictor 后应接近 x[t+1]（下一位置的输入表示）
                if end < seq_len:
                    with torch.enable_grad():
                        # 确保 W 参与计算图（Z_chunk 作为常数，W 作为可微参数）
                        W = self.down_proj.weight
                        # Z_chunk 已在 no_grad 中计算，作为常数参与线性运算
                        # F.linear(constant, W) 的梯度 = d(loss)/d(W) = d(loss)/d(Y) * Z_chunk^T
                        Y_for_grad = F.linear(Z_chunk, W, self.down_proj.bias)
                        pred_len = min(Y_for_grad.shape[1], seq_len - start - 1)
                        if pred_len > 0:
                            Y_pred = Y_for_grad[:, :pred_len, :]
                            target = x[:, start + 1:start + 1 + pred_len, :].detach()
                            # 预测损失：Y_chunk 经 ttt_predictor 后应接近下一位置的输入
                            prediction = self.ttt_predictor(Y_pred)
                            ntp_loss = F.mse_loss(prediction, target)
                            # 正则化：防止权重偏离初始值太远（降低系数避免压制有效梯度）
                            if self._W0 is not None:
                                reg_loss = 1e-4 * F.mse_loss(W, self._W0)
                            else:
                                reg_loss = 1e-5 * W.pow(2).mean()
                            loss = ntp_loss + reg_loss
                            grads = torch.autograd.grad(
                                loss, W,
                                create_graph=False, allow_unused=True
                            )
                            if grads[0] is not None:
                                with torch.no_grad():
                                    # 梯度归一化（方向）+ 权重范数缩放（幅度）：
                                    # update = lr * (weight_norm / grad_norm) * grad
                                    # 使得单次更新幅度 ≈ lr * weight_norm，不受模型大小影响
                                    grad_norm = grads[0].norm().clamp(min=1e-8)
                                    weight_norm = W.data.norm()
                                    scale = self.ttt_lr * weight_norm / grad_norm
                                    update = scale * grads[0]
                                    # 限制单次更新幅度不超过权重的 5%
                                    max_update = 0.05 * weight_norm
                                    if update.norm() > max_update:
                                        update = update * (max_update / update.norm())
                                    self.down_proj.weight.data -= update

            return torch.cat(outputs, dim=1)
        else:
            return self.down_proj(Z)


class LoRAFeedForward(FeedForward):
    """在 FFN 的 gate_proj 和 up_proj 上加低秩旁路

    旁路: hidden_size → r → intermediate_size，初始化 B=0 保证开始时无贡献。
    仅增加约 2*r*(hidden_size+intermediate_size) 个参数。
    """

    def __init__(self, config: MiniMindConfig, intermediate_size: int = None, lora_r: int = 8):
        super().__init__(config, intermediate_size)
        self.lora_r = lora_r
        self.gate_lora_A = nn.Linear(config.hidden_size, lora_r, bias=False)
        self.gate_lora_B = nn.Linear(lora_r, self.gate_proj.out_features, bias=False)
        self.up_lora_A = nn.Linear(config.hidden_size, lora_r, bias=False)
        self.up_lora_B = nn.Linear(lora_r, self.up_proj.out_features, bias=False)
        nn.init.zeros_(self.gate_lora_B.weight)
        nn.init.zeros_(self.up_lora_B.weight)
        self.lora_scale = 1.0

    def forward(self, x):
        gate = self.act_fn(self.gate_proj(x) + self.lora_scale * self.gate_lora_B(self.gate_lora_A(x)))
        up = self.up_proj(x) + self.lora_scale * self.up_lora_B(self.up_lora_A(x))
        return self.down_proj(gate * up)


class MambaLayer(nn.Module):
    """简化版 Mamba SSM 层，替代 Attention

    使用选择性状态空间模型 (S6) 的核心递推结构：
    h_t = exp(Δ_t * A) * h_{t-1} + Δ_t * B_t * x_t
    y_t = C_t * h_t + D * x_t

    训练时使用并行扫描，推理时 O(1) 递推。
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.expand_factor = 2
        self.d_inner = self.hidden_size * self.expand_factor
        self.d_state = 64
        self.dt_rank = max(self.hidden_size // 16, 16)
        self.in_proj = nn.Linear(config.hidden_size, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=3,
                                padding=1, groups=self.d_inner)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1).clone()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, config.hidden_size, bias=False)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, position_embeddings=None, past_key_value=None, use_cache=False, attention_mask=None):
        _ = position_embeddings
        residual = x
        x = self.norm(x)
        batch, seq_len, _ = x.shape
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)
        x_proj = self.conv1d(x_proj.transpose(1, 2)).transpose(1, 2)
        x_proj = F.silu(x_proj)
        A = -torch.exp(self.A_log)
        dtBC = self.x_proj(x_proj)
        dt, B, C = dtBC.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        y = self._ssm_scan(x_proj, dt, A, B, C)
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_proj
        output = y * F.silu(z)
        output = self.out_proj(output)
        return output + residual, (x.new_zeros(0, 0, 0, 0), x.new_zeros(0, 0, 0, 0))

    def _ssm_scan(self, x, dt, A, B, C):
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]
        h = x.new_zeros(batch, d_inner, d_state)
        ys = []
        for t in range(seq_len):
            dt_t = dt[:, t, :].unsqueeze(-1)
            dA = torch.exp(dt_t * A)
            dBx = dt_t * B[:, t, :].unsqueeze(1) * x[:, t, :].unsqueeze(-1)
            h = dA * h + dBx
            y = torch.einsum('bdn,bn->bd', h, C[:, t, :])
            ys.append(y)
        return torch.stack(ys, dim=1)


class MTPHead(nn.Module):
    """Multi-Token Prediction 额外预测头

    每个 MTPHead 预测未来第 n+1 个 token。
    使用独立的 RMSNorm + 线性投影，与主 lm_head 共享词表但参数独立。
    参考: DeepSeek-V3 Multi-Token Prediction
    """

    def __init__(self, config: MiniMindConfig, head_index: int):
        super().__init__()
        self.head_index = head_index
        # 每个预测头有独立的 norm 和投影
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 轻量投影：将 hidden_size 映射到 hidden_size 再到 vocab
        self.projector = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, hidden_states):
        """预测未来第 head_index+1 个 token

        Args:
            hidden_states: [batch, seq_len, hidden_size]
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        h = self.norm(hidden_states)
        h = self.projector(h)
        return self.lm_head(h)

class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        scores = F.softmax(self.gate(x_flat), dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)
        if self.config.norm_topk_prob: topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = topk_weight[mask].view(-1, 1)
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
        if self.training and self.config.router_aux_loss_coef > 0:
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()
        return y.view(batch_size, seq_len, hidden_dim)

class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig, use_ttt: bool = False):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.parallel_attn_ffn = config.parallel_attn_ffn
        attn_type = getattr(config, 'attention_type', 'standard')
        if getattr(config, 'mamba_hybrid', False) and layer_id < int(config.num_hidden_layers * getattr(config, 'mamba_ratio', 0.5)):
            self.self_attn = MambaLayer(config)
        elif attn_type == 'linear':
            self.self_attn = LinearAttention(config)
        elif attn_type == 'alibi':
            self.self_attn = ALiBiAttention(config)
        else:
            self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if config.use_moe:
            self.mlp = MOEFeedForward(config)
        elif use_ttt:
            self.mlp = TTTFeedForward(config)
        elif getattr(config, 'lora_ffn', False):
            self.mlp = LoRAFeedForward(config, lora_r=getattr(config, 'lora_ffn_r', 8))
        else:
            self.mlp = FeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None, use_ttt=False):
        if self.parallel_attn_ffn:
            normed = self.input_layernorm(hidden_states)
            attn_out, present_key_value = self.self_attn(
                normed, position_embeddings, past_key_value, use_cache, attention_mask
            )
            if isinstance(self.mlp, TTTFeedForward):
                mlp_out = self.mlp(self.post_attention_layernorm(hidden_states), use_ttt=use_ttt)
            else:
                mlp_out = self.mlp(self.post_attention_layernorm(hidden_states))
            return hidden_states + attn_out + mlp_out, present_key_value
        else:
            residual = hidden_states
            hidden_states, present_key_value = self.self_attn(
                self.input_layernorm(hidden_states), position_embeddings,
                past_key_value, use_cache, attention_mask
            )
            hidden_states += residual
            if isinstance(self.mlp, TTTFeedForward):
                hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states), use_ttt=use_ttt)
            else:
                hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
            return hidden_states, present_key_value

class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        # 计算 TTT 层集合
        ttt_layer_set = set()
        if config.ttt_enabled:
            if config.ttt_layers is not None:
                ttt_layer_set = set(config.ttt_layers)
            else:
                ttt_layer_set = set(range(config.num_hidden_layers * 3 // 4, config.num_hidden_layers))
        # 跨层参数共享：每 layer_share_factor 层共享一组参数
        share_factor = config.layer_share_factor
        if share_factor > 1:
            num_unique = math.ceil(self.num_hidden_layers / share_factor)
            self.unique_layers = nn.ModuleList()
            for i in range(num_unique):
                # 该 unique block 代表的虚拟层范围: [i*share_factor, (i+1)*share_factor)
                # 如果范围内任一虚拟层需要 TTT，则该 block 使用 TTT
                virtual_layers = range(i * share_factor, min((i + 1) * share_factor, self.num_hidden_layers))
                block_use_ttt = any(vl in ttt_layer_set for vl in virtual_layers)
                self.unique_layers.append(MiniMindBlock(i, config, use_ttt=block_use_ttt))
            self.layers = None
        else:
            self.unique_layers = None
            self.layers = nn.ModuleList([
                MiniMindBlock(l, config, use_ttt=(l in ttt_layer_set))
                for l in range(self.num_hidden_layers)
            ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def _get_layer(self, layer_idx):
        """获取指定层的模块（支持参数共享）"""
        if self.unique_layers is not None:
            return self.unique_layers[layer_idx // self.config.layer_share_factor]
        return self.layers[layer_idx]

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, use_ttt=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        use_kv_cache = isinstance(past_key_values, KVCache)
        if hasattr(past_key_values, 'layers'): past_key_values = None
        if not use_kv_cache:
            past_key_values = past_key_values or [None] * self.num_hidden_layers
            start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        else:
            start_pos = past_key_values.len[0]
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])
        presents = []
        for layer_idx in range(self.num_hidden_layers):
            layer = self._get_layer(layer_idx)
            if use_kv_cache:
                past_kv_for_layer = past_key_values.get(layer_idx)
            else:
                past_kv_for_layer = past_key_values[layer_idx]
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_kv_for_layer,
                use_cache=use_cache,
                attention_mask=attention_mask,
                use_ttt=use_ttt,
            )
            if use_kv_cache and use_cache:
                new_k, new_v = present
                past_key_values.update(layer_idx, new_k, new_v)
                presents.append(present)
            else:
                presents.append(present)
        hidden_states = self.norm(hidden_states)
        # 计算 aux_loss（兼容参数共享）
        if self.unique_layers is not None:
            aux_loss = sum([l.mlp.aux_loss for l in self.unique_layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        else:
            aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss

class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings: self.model.embed_tokens.weight = self.lm_head.weight
        # Multi-Token Prediction 额外预测头
        self.mtp_heads = nn.ModuleList([
            MTPHead(self.config, i) for i in range(self.config.mtp_num_heads)
        ]) if self.config.mtp_num_heads > 0 else None
        self.post_init()

    def enable_ttt(self, lr=None, chunk_size=None):
        """启用所有 TTT 层的推理时训练"""
        layers = self.model.unique_layers if self.model.unique_layers is not None else self.model.layers
        for layer in layers:
            if isinstance(layer.mlp, TTTFeedForward):
                layer.mlp.enable_ttt(lr=lr, chunk_size=chunk_size)

    def disable_ttt(self):
        """禁用所有 TTT 层的推理时训练，恢复初始权重"""
        layers = self.model.unique_layers if self.model.unique_layers is not None else self.model.layers
        for layer in layers:
            if isinstance(layer.mlp, TTTFeedForward):
                layer.mlp.disable_ttt()

    def reset_ttt_weights(self):
        """重置所有 TTT 层权重到初始值"""
        layers = self.model.unique_layers if self.model.unique_layers is not None else self.model.layers
        for layer in layers:
            if isinstance(layer.mlp, TTTFeedForward):
                layer.mlp.reset_ttt_weights()

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, use_ttt=False, **kwargs):
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, use_ttt=use_ttt, **kwargs)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            # 主预测头损失
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
            # MTP 辅助损失
            if self.mtp_heads is not None:
                mtp_loss = torch.tensor(0.0, device=loss.device)
                for head in self.mtp_heads:
                    # 第 i 个头预测未来第 i+2 个 token
                    offset = head.head_index + 2
                    if hidden_states.size(1) > offset:
                        mtp_logits = head(hidden_states[:, :-offset, :])
                        mtp_labels = labels[:, offset:]
                        if mtp_labels.size(1) > 0 and mtp_logits.size(1) == mtp_labels.size(1):
                            mtp_loss += F.cross_entropy(
                                mtp_logits.reshape(-1, mtp_logits.size(-1)),
                                mtp_labels.reshape(-1),
                                ignore_index=-100
                            )
                if mtp_loss.item() > 0:
                    loss = loss + self.config.mtp_loss_weight * mtp_loss
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
    
    # https://github.com/jingyaogong/minimind/discussions/611
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, use_ttt=False, ttt_interval=64, **kwargs):
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        ttt_step_counter = 0
        has_linear_or_mamba = getattr(self.config, 'attention_type', 'standard') == 'linear' or getattr(self.config, 'mamba_hybrid', False)
        use_kv_cache = use_cache and not use_ttt and not has_linear_or_mamba
        if use_kv_cache:
            kv_cache = KVCache(
                n_layers=self.config.num_hidden_layers,
                bsz=input_ids.shape[0],
                max_len=self.config.max_position_embeddings,
                n_kv_heads=self.config.num_key_value_heads or self.config.num_attention_heads,
                head_dim=self.config.head_dim,
                device=input_ids.device,
                dtype=next(self.parameters()).dtype,
            )
        else:
            kv_cache = None
        no_grad_ctx = torch.no_grad() if not use_ttt else torch.enable_grad()
        with no_grad_ctx:
            if streamer: streamer.put(input_ids.cpu())
            # Prefill: process the full prompt at once
            if use_kv_cache:
                outputs = self.forward(input_ids, attention_mask, kv_cache, use_cache=True, use_ttt=False, **kwargs)
                past_len = input_ids.shape[1]
            else:
                past_len = 0
            for _ in range(max_new_tokens):
                if use_kv_cache and past_len > 0:
                    outputs = self.forward(input_ids[:, -1:], None, kv_cache, use_cache=True, use_ttt=False, **kwargs)
                else:
                    do_ttt = use_ttt and (ttt_step_counter % ttt_interval == 0 and ttt_step_counter > 0)
                    if do_ttt and use_cache:
                        outputs = self.forward(input_ids, attention_mask, None, use_cache=True, use_ttt=True, **kwargs)
                    else:
                        cur_past = past_key_values if not use_kv_cache else None
                        cur_start = past_len if use_kv_cache else (past_key_values[0][0].shape[1] if past_key_values else 0)
                        outputs = self.forward(input_ids[:, cur_start:], attention_mask, cur_past, use_cache=use_cache, use_ttt=do_ttt, **kwargs)
                        if use_cache and not use_kv_cache:
                            past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :] / temperature
                if repetition_penalty != 1.0:
                    for i in range(input_ids.shape[0]):
                        seen = torch.unique(input_ids[i]); score = logits[i, seen]; logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
                if top_k > 0: 
                    logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                    mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                    logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
                next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
                if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                past_len += 1
                ttt_step_counter += 1
                if streamer: streamer.put(next_token.cpu())
                if eos_token_id is not None:
                    finished |= next_token.squeeze(-1).eq(eos_token_id)
                    if finished.all(): break
            if streamer: streamer.end()
        if kwargs.get("return_kv"):
            return {'generated_ids': input_ids, 'past_kv': kv_cache.to_legacy_format() if kv_cache else past_key_values}
        return input_ids