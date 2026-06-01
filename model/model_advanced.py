"""
DeepSeek V4 高级架构模块：mHC + CSA + MSA
基于 arxiv:2512.24880 (mHC) 的流形约束超连接 + CSA 压缩稀疏注意力 + MSA 稀疏注意力

============================================================
mHC: Manifold-Constrained Hyper-Connections
============================================================
将单流残差扩展为 n_hc 条并行流，混合矩阵 B 约束到 Birkhoff 多胞形（双随机矩阵），
通过 Sinkhorn-Knopp 迭代保证谱范数 <=1，深层不爆炸。

============================================================
CSA: Compressed Sparse Attention
============================================================
每 m 个 KV 压缩为 1 个 → Lightning Indexer 选 top-k → Core Attention on selected blocks。
长上下文下大幅减少计算量（1M 上下文 FLOPs 仅为标准注意力的 27%）。

============================================================
用法：
  from model.model_advanced import MHC_CSABlock
  block = MHC_CSABlock(layer_id=0, config=config, use_csa=True)
  out, cache = block(hidden_states, position_embeddings)
"""
import math
import torch
import torch.nn.functional as F
from torch import nn
from model.model_minimind import MiniMindConfig, RMSNorm, FeedForward, MOEFeedForward, TTTFeedForward, repeat_kv, apply_rotary_pos_emb


# ============================================================
# mHC — Manifold-Constrained Hyper-Connections
# ============================================================

class MHCConnection(nn.Module):
    def __init__(self, hidden_size: int, num_streams: int = 2, num_iter: int = 10):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_streams = num_streams
        self.num_iter = num_iter
        self.A = nn.Parameter(torch.empty(num_streams, hidden_size, hidden_size))
        self.C = nn.Parameter(torch.empty(num_streams, hidden_size, hidden_size))
        self.B = nn.Parameter(torch.empty(num_streams, num_streams))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.A, std=0.02 / math.sqrt(self.num_streams))
        nn.init.normal_(self.C, std=0.02 / math.sqrt(self.num_streams))
        nn.init.eye_(self.B)
        self.B.data += torch.randn_like(self.B) * 0.01

    def _sinkhorn_knopp(self, M):
        B = torch.abs(M)
        B = B / (B.sum(dim=-1, keepdim=True) + 1e-12)
        for _ in range(self.num_iter):
            B = B / (B.sum(dim=0, keepdim=True) + 1e-12)
            B = B / (B.sum(dim=1, keepdim=True) + 1e-12)
        return B

    def forward(self, streams):
        B = self._sinkhorn_knopp(self.B)
        A_gate = torch.sigmoid(self.A)
        C_gate = torch.sigmoid(self.C)
        pre_out = sum(streams[:, :, i, :] @ A_gate[i].T for i in range(self.num_streams))
        self._cached_A = A_gate
        self._cached_C = C_gate
        self._cached_B = B
        self._cached_pre_out = pre_out
        return pre_out


class MHCBlock(nn.Module):
    def __init__(self, layer_id: int, hidden_size: int, num_streams: int = 2, num_iter: int = 10):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.num_streams = num_streams
        self.mhc = MHCConnection(hidden_size, num_streams, num_iter)
        self.input_layernorm = nn.LayerNorm(hidden_size, eps=1e-5)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=1e-5)

    def forward(self, streams, attention_fn, ffn_fn):
        B, C = self.mhc._cached_B, self.mhc._cached_C
        h = self.mhc._cached_pre_out
        h = h + attention_fn(self.input_layernorm(h))
        h = h + ffn_fn(self.post_attention_layernorm(h))
        post_streams = torch.stack([h @ C[i].T for i in range(self.num_streams)], dim=2)
        mixed = torch.einsum('ij,bsjd->bsid', B, streams)
        return mixed + post_streams


# ============================================================
# CSA — Compressed Sparse Attention
# ============================================================

class CompressedSparseAttention(nn.Module):
    def __init__(self, config: MiniMindConfig, compress_factor: int = 4):
        super().__init__()
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.compress_factor = compress_factor
        self.is_causal = True
        self.dropout = config.dropout

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.n_local_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.n_local_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        idx_dim = max(self.head_dim // 4, 16)
        self.indexer_q_proj = nn.Linear(self.head_dim, idx_dim, bias=False)
        self.indexer_k_proj = nn.Linear(self.head_dim, idx_dim, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def _compress_kv(self, xk, xv):
        b, seq_len, n_kv_heads, head_dim = xk.shape
        m = self.compress_factor
        if seq_len <= m:
            return xk, xv, 1
        n_blocks = (seq_len + m - 1) // m
        pad_len = n_blocks * m - seq_len
        if pad_len > 0:
            xk = F.pad(xk, (0, 0, 0, 0, 0, pad_len))
            xv = F.pad(xv, (0, 0, 0, 0, 0, pad_len))
        xk_blocks = xk.view(b, n_blocks, m, n_kv_heads, head_dim)
        xv_blocks = xv.view(b, n_blocks, m, n_kv_heads, head_dim)
        k_last = xk_blocks[:, :, -1, :, :]
        attn_weights = F.softmax(
            torch.einsum('bnhd,bnmhd->bnmh', k_last, xk_blocks) / math.sqrt(head_dim), dim=2
        ).unsqueeze(-1)
        return (xk_blocks * attn_weights).sum(dim=2), (xv_blocks * attn_weights).sum(dim=2), n_blocks

    def _indexer_topk(self, xq, compressed_k):
        b, seq_len, n_heads, head_dim = xq.shape
        _, n_blocks, _, _ = compressed_k.shape
        q_low = self.indexer_q_proj(xq.mean(dim=2).reshape(b * seq_len, head_dim)).view(b, seq_len, -1)
        k_low = self.indexer_k_proj(compressed_k.mean(dim=2).reshape(b * n_blocks, head_dim)).view(b, n_blocks, -1)
        scores = torch.einsum('bsd,bnd->bsn', q_low, k_low)
        block_pos = torch.arange(n_blocks, device=scores.device).view(1, 1, -1)
        token_block_idx = torch.arange(seq_len, device=scores.device).view(1, -1, 1) // self.compress_factor
        scores.masked_fill_(block_pos > token_block_idx, float('-inf'))
        topk = min(max(1, n_blocks // 2), n_blocks)
        _, topk_indices = torch.topk(scores, k=topk, dim=-1)
        return topk_indices.unsqueeze(2).expand(-1, -1, n_heads, -1)

    def _core_attention(self, xq, compressed_k, compressed_v, topk_indices):
        b, seq_len, n_heads, head_dim = xq.shape
        n_kv_heads = compressed_k.shape[2]
        n_groups = n_heads // n_kv_heads
        topk = topk_indices.shape[-1]
        k_expanded = compressed_k.repeat_interleave(n_groups, dim=2)
        v_expanded = compressed_v.repeat_interleave(n_groups, dim=2)
        batch_idx = torch.arange(b, device=topk_indices.device).view(b, 1, 1, 1).expand(b, seq_len, n_heads, topk)
        head_idx = torch.arange(n_heads, device=topk_indices.device).view(1, 1, n_heads, 1).expand(b, seq_len, n_heads, topk)
        k_selected = k_expanded[batch_idx, topk_indices, head_idx]
        v_selected = v_expanded[batch_idx, topk_indices, head_idx]
        scores = torch.einsum('bqhd,bqhtd->bqht', xq, k_selected) / math.sqrt(head_dim)
        attn_weights = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(scores))
        return torch.einsum('bqht,bqhtd->bqhd', attn_weights, v_selected)

    def _standard_attention(self, xq, xk, xv, seq_len):
        bsz = xq.shape[0]
        xq_t, xk_t = xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv_t = repeat_kv(xv, self.n_rep).transpose(1, 2)
        if self.flash:
            out = F.scaled_dot_product_attention(xq_t, xk_t, xv_t, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            scores = (xq_t @ xk_t.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=scores.device, dtype=scores.dtype), 1)
            scores[:, :, -seq_len:, -seq_len:] += mask
            out = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq_t)) @ xv_t
        return out.transpose(1, 2).reshape(bsz, seq_len, -1)

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
        total_len = xk.shape[1]
        if total_len <= self.compress_factor * 4:
            out = self._standard_attention(xq, xk, xv, total_len)
            return self.resid_dropout(self.o_proj(out)), past_kv
        compressed_k, compressed_v, _ = self._compress_kv(xk, xv)
        topk_indices = self._indexer_topk(xq, compressed_k)
        attn_out = self._core_attention(xq, compressed_k, compressed_v, topk_indices)
        return self.resid_dropout(self.o_proj(attn_out.reshape(bsz, seq_len, -1))), past_kv


# ============================================================
# MHC-CSA Block — 集成 mHC+CSA 的 Transformer Block
# ============================================================

class MHC_CSABlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig, use_ttt: bool = False,
                 use_csa: bool = True, num_streams: int = 2):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.num_streams = num_streams
        if use_csa:
            self.self_attn = CompressedSparseAttention(config)
        else:
            from model.model_minimind import Attention
            self.self_attn = Attention(config)
        self.use_mhc = True
        self.mhc = MHCConnection(config.hidden_size, num_streams=num_streams)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if config.use_moe:
            self.mlp = MOEFeedForward(config)
        elif use_ttt:
            self.mlp = TTTFeedForward(config)
        else:
            self.mlp = FeedForward(config)

    def forward(self, hidden_states, position_embeddings,
                past_key_value=None, use_cache=False, attention_mask=None, use_ttt=False):
        h_streams = hidden_states.unsqueeze(2).expand(-1, -1, self.num_streams, -1).clone()
        h_pre = self.mhc(h_streams)
        attn_out, present_key_value = self.self_attn(
            self.input_layernorm(h_pre), position_embeddings, past_key_value, use_cache, attention_mask)
        h = h_pre + attn_out
        if isinstance(self.mlp, TTTFeedForward):
            h = h + self.mlp(self.post_attention_layernorm(h), use_ttt=use_ttt)
        else:
            h = h + self.mlp(self.post_attention_layernorm(h))
        C = self.mhc._cached_C
        post_streams = torch.stack([h @ C[i].T for i in range(self.num_streams)], dim=2)
        mixed = torch.einsum('ij,bsjd->bsid', self.mhc._cached_B, h_streams)
        return (mixed + post_streams).mean(dim=2), present_key_value

    def forward_without_mhc(self, hidden_states, position_embeddings,
                            past_key_value=None, use_cache=False, attention_mask=None, use_ttt=False):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings, past_key_value, use_cache, attention_mask)
        hidden_states = hidden_states + residual
        if isinstance(self.mlp, TTTFeedForward):
            hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states), use_ttt=use_ttt)
        else:
            hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


# ============================================================
# MSA — MiniMax Sparse Attention (from MiniMax M3)
# ============================================================
# 两阶段稀疏注意力：Index Branch + Sparse Branch
#
# Index Branch:
#   - 每个 GQA 组共享一个 index query (Q_idx)
#   - K_idx 只有 1 个头（所有头共享），计算成本极低
#   - Block Max Pool 将 token-level 分数压缩为 block-level 分数
#   - TopK 选择每个 GQA 组要关注的 KV 块
#
# Sparse Branch:
#   - 在选中的原始 K/V 块上运行标准 softmax 注意力
#   - 同一 GQA 组内的 query 头共享 top-k 选择
#   - 不压缩 K/V，保持 softmax 注意力的完整表达能力
#
# 关键设计选择（vs DeepSeek NSA/CSA）：
#   - GQA 而非 MLA → 可复用 FlashAttention kernel
#   - block-level 选择 + 原始 K/V 注意力 → 不损失精度
#   - 单分支选择 → 工程简洁，无滑动窗口/压缩分支
# ============================================================

class MiniMaxSparseAttention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = config.num_key_value_heads or config.num_attention_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.n_gqa_groups = self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.block_size = config.msa_block_size
        self.topk_ratio = config.msa_topk_ratio
        self.fallback_len = config.msa_fallback_len
        self.dropout = config.dropout

        self.q_proj = nn.Linear(config.hidden_size, self.n_local_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.n_local_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.n_local_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_local_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        idx_dim = config.msa_idx_dim
        self.idx_q_proj = nn.Linear(self.head_dim, idx_dim, bias=False)
        self.idx_k_proj = nn.Linear(self.head_dim, idx_dim, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def _index_branch(self, xq, xk):
        b, seq_len, n_heads, head_dim = xq.shape
        _, kv_len, n_kv_heads, _ = xk.shape
        block_size = self.block_size
        n_blocks = (kv_len + block_size - 1) // block_size

        pad_len = n_blocks * block_size - kv_len
        if pad_len > 0:
            xk_padded = F.pad(xk, (0, 0, 0, 0, 0, pad_len), value=0.0)
        else:
            xk_padded = xk

        xk_blocks = xk_padded.view(b, n_blocks, block_size, n_kv_heads, head_dim)

        q_per_group = n_heads // n_kv_heads
        xq_grouped = xq.reshape(b, seq_len, n_kv_heads, q_per_group, head_dim)
        q_idx = xq_grouped.mean(dim=3)

        q_idx_feat = self.idx_q_proj(q_idx.reshape(b * seq_len * n_kv_heads, head_dim))
        q_idx_feat = q_idx_feat.view(b, seq_len, n_kv_heads, -1)

        k_idx_feat = self.idx_k_proj(xk_blocks.reshape(b * n_blocks * block_size * n_kv_heads, head_dim))
        k_idx_feat = k_idx_feat.view(b, n_blocks, block_size, n_kv_heads, -1)

        token_scores = torch.einsum('bsgd,bmngd->bsgnm', q_idx_feat, k_idx_feat) / math.sqrt(q_idx_feat.shape[-1])

        q_block_idx = torch.arange(seq_len, device=token_scores.device).view(1, seq_len, 1, 1, 1) // block_size
        k_block_idx = torch.arange(n_blocks, device=token_scores.device).view(1, 1, 1, 1, n_blocks)
        causal_mask = k_block_idx > q_block_idx
        token_scores.masked_fill_(causal_mask, float('-inf'))

        block_scores = token_scores.amax(dim=3)

        topk = max(1, min(int(n_blocks * self.topk_ratio), n_blocks))
        _, topk_indices = torch.topk(block_scores, k=topk, dim=-1)

        return topk_indices

    def _sparse_attention(self, xq, xk, xv, topk_indices):
        b, seq_len, n_heads, head_dim = xq.shape
        _, kv_len, n_kv_heads, _ = xk.shape
        block_size = self.block_size
        n_blocks = (kv_len + block_size - 1) // block_size
        topk = topk_indices.shape[-1]

        pad_len = n_blocks * block_size - kv_len
        if pad_len > 0:
            xk = F.pad(xk, (0, 0, 0, 0, 0, pad_len), value=0.0)
            xv = F.pad(xv, (0, 0, 0, 0, 0, pad_len), value=0.0)

        xk_exp = repeat_kv(xk, self.n_rep).contiguous()
        xv_exp = repeat_kv(xv, self.n_rep).contiguous()

        xk_blocks = xk_exp.reshape(b, n_blocks, block_size, n_heads, head_dim)
        xv_blocks = xv_exp.reshape(b, n_blocks, block_size, n_heads, head_dim)

        q_per_group = n_heads // n_kv_heads
        group_topk = topk_indices.unsqueeze(3).expand(-1, -1, -1, q_per_group, -1).reshape(b, seq_len, n_heads, topk)

        batch_idx = torch.arange(b, device=topk_indices.device).view(b, 1, 1, 1).expand(b, seq_len, n_heads, topk)

        k_selected = xk_blocks[batch_idx, group_topk]  # [b, seq_len, n_heads, topk, block_size, n_heads, head_dim]
        v_selected = xv_blocks[batch_idx, group_topk]

        head_idx = torch.arange(n_heads, device=topk_indices.device).view(1, 1, n_heads, 1, 1, 1, 1).expand(b, seq_len, -1, topk, block_size, 1, head_dim)
        k_selected = k_selected.gather(5, head_idx).squeeze(5)
        v_selected = v_selected.gather(5, head_idx).squeeze(5)

        k_selected = k_selected.reshape(b, seq_len, n_heads, topk * block_size, head_dim)
        v_selected = v_selected.reshape(b, seq_len, n_heads, topk * block_size, head_dim)

        xq_t = xq.transpose(1, 2).unsqueeze(3)  # [b, n_heads, seq_len, 1, head_dim]
        k_t = k_selected.transpose(1, 2)  # [b, n_heads, seq_len, topk*block_size, head_dim]
        v_t = v_selected.transpose(1, 2)

        scores = (xq_t @ k_t.transpose(-2, -1)).squeeze(3) / math.sqrt(head_dim)  # [b, n_heads, seq_len, topk*block_size]

        q_token_pos = torch.arange(seq_len, device=scores.device).view(1, 1, -1, 1)
        k_block_idx = group_topk.transpose(1, 2)
        k_token_start = k_block_idx * block_size
        k_token_offset = torch.arange(block_size, device=scores.device).view(1, 1, 1, 1, -1)
        k_token_idx = k_token_start.unsqueeze(-1) + k_token_offset
        k_token_idx = k_token_idx.reshape(b, n_heads, seq_len, topk * block_size)
        causal_mask = k_token_idx > q_token_pos
        scores.masked_fill_(causal_mask, float('-inf'))

        attn_weights = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(scores))
        out = attn_weights.unsqueeze(3) @ v_t  # [b, n_heads, seq_len, 1, head_dim]
        out = out.squeeze(3)  # [b, n_heads, seq_len, head_dim]

        return out.transpose(1, 2).reshape(b, seq_len, -1)

    def _standard_attention(self, xq, xk, xv, kv_len):
        bsz, q_len, _, _ = xq.shape
        xq_t = xq.transpose(1, 2)
        xk_t = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv_t = repeat_kv(xv, self.n_rep).transpose(1, 2)
        if self.flash:
            out = F.scaled_dot_product_attention(
                xq_t, xk_t, xv_t,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=q_len > 1,
            )
        else:
            scores = (xq_t @ xk_t.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if q_len > 1:
                mask = torch.triu(torch.full((q_len, kv_len), float("-inf"), device=scores.device, dtype=scores.dtype), diagonal=kv_len - q_len + 1)
                scores[:, :, -q_len:, -kv_len:] += mask
            out = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq_t)) @ xv_t
        return out.transpose(1, 2).reshape(bsz, q_len, -1)

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape
        xq = self.q_proj(x).view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = self.k_proj(x).view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = self.v_proj(x).view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        if past_key_value is not None:
            is_kv_cache = isinstance(past_key_value, tuple) and len(past_key_value) == 2 and isinstance(past_key_value[0], type(None)) == False and hasattr(past_key_value[0], 'shape') == False
            if is_kv_cache:
                from model.model_minimind import KVCache
                is_kv_cache = isinstance(past_key_value[0], KVCache)
            if is_kv_cache:
                full_k, full_v = past_key_value[0].update(past_key_value[1], xk, xv)
            else:
                full_k = torch.cat([past_key_value[0], xk], dim=1)
                full_v = torch.cat([past_key_value[1], xv], dim=1)
        else:
            full_k, full_v = xk, xv

        past_kv = (xk, xv) if use_cache else None
        total_len = full_k.shape[1]

        if total_len <= self.fallback_len:
            out = self._standard_attention(xq, full_k, full_v, total_len)
            return self.resid_dropout(self.o_proj(out)), past_kv

        topk_indices = self._index_branch(xq, full_k)
        out = self._sparse_attention(xq, full_k, full_v, topk_indices)
        return self.resid_dropout(self.o_proj(out)), past_kv
