from typing import Optional, Tuple

import math
import torch
import torch.nn.functional as F

from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    apply_rotary_pos_emb,
    repeat_kv,
)

from .covariant_gauge_adapter_v2 import CovariantGaugeAdapter


class GaugeQwen2Attention(Qwen2Attention):
    """
    使用官方 attention 主体 + Gauge bias/value 修正的 Qwen2 Attention

    最终输出：
        out = AttentionOfficial(Q,K,V; mask + gauge_bias) + delta_v
    """

    def __init__(
        self,
        config,
        layer_idx: int,
        gauge_rank: int = 16,
        gauge_dropout: float = 0.0,
        gauge_use_layernorm: bool = True,
        gauge_smoothness_weight: float = 0.0,
        gauge_field_l2_weight: float = 0.0,
        gauge_init_scale: float = 1e-3,
    ):
        super().__init__(config=config, layer_idx=layer_idx)

        self.gauge_adapter = CovariantGaugeAdapter(
            d_model=config.hidden_size,
            num_heads=config.num_attention_heads,
            rank=gauge_rank,
            dropout=gauge_dropout,
            use_layernorm=gauge_use_layernorm,
            smoothness_weight=gauge_smoothness_weight,
            field_l2_weight=gauge_field_l2_weight,
            init_scale=gauge_init_scale,
        )

        self._last_q = None
        self._last_k = None
        self._last_v = None
        self._last_k_rep = None
        self._last_v_rep = None
        self._last_attn_weights = None

    def _combine_mask_and_bias(
        self,
        attention_mask: Optional[torch.Tensor],
        gauge_bias: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        将 base attention_mask 与 gauge_bias 合成为 additive mask/bias

        返回形状：
            [B, H, S, S]
        """
        if attention_mask is None:
            return gauge_bias.to(dtype)

        # bool mask -> additive
        if attention_mask.dtype == torch.bool:
            additive = torch.zeros_like(gauge_bias, dtype=dtype)
            additive = additive.masked_fill(~attention_mask, float("-inf"))
            return additive + gauge_bias.to(dtype)

        # 0/1 mask -> additive
        if attention_mask.max() <= 1 and attention_mask.min() >= 0:
            additive = torch.zeros_like(gauge_bias, dtype=dtype)
            additive = additive.masked_fill(attention_mask == 0, float("-inf"))
            return additive + gauge_bias.to(dtype)

        # 已经是 additive mask
        return attention_mask.to(dtype) + gauge_bias.to(dtype)

    def _eager_attention(
        self,
        q: torch.Tensor,  # [B,H,S,Hd]
        k: torch.Tensor,  # [B,H,S,Hd]
        v: torch.Tensor,  # [B,H,S,Hd]
        attn_bias: Optional[torch.Tensor],  # [B,H,S,S]
        dropout_p: float,
    ):
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        if attn_bias is not None:
            scores = scores + attn_bias

        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=self.training)
        attn_out = torch.matmul(attn_weights, v)
        return attn_out, attn_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values=None,
        **kwargs,
    ):
        """
        对齐当前 HF Qwen2Attention.forward 签名
        """
        output_attentions = kwargs.get("output_attentions", False)

        input_shape = hidden_states.shape[:-1]  # [B, S]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # 1) q/k/v projection
        q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # [B,Hq,S,Hd]
        k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # [B,Hkv,S,Hd]
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # [B,Hkv,S,Hd]

        # 2) RoPE
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 3) cache update
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)

        # 4) GQA repeat
        k_rep = repeat_kv(k, self.num_key_value_groups)
        v_rep = repeat_kv(v, self.num_key_value_groups)

        # 5) Gauge 只生成 bias 和 delta_v
        gauge_bias, delta_v = self.gauge_adapter(
            hidden_states=hidden_states,
            q_base=q,
            k_base=k_rep,
        )

        # 6) 合并 base mask 与 gauge bias
        attn_bias = self._combine_mask_and_bias(attention_mask, gauge_bias, q.dtype)

        dropout_p = self.attention_dropout if self.training else 0.0

        # 7) attention 主体：优先官方 SDPA
        if output_attentions:
            attn_out, attn_weights = self._eager_attention(
                q=q,
                k=k_rep,
                v=v_rep,
                attn_bias=attn_bias,
                dropout_p=dropout_p,
            )
        else:
            attn_out = F.scaled_dot_product_attention(
                q,
                k_rep,
                v_rep,
                attn_mask=attn_bias,
                dropout_p=dropout_p,
                is_causal=False,  # causal 已由 attention_mask 体现
            )
            attn_weights = None

        # 8) merge heads + o_proj
        attn_out = attn_out.transpose(1, 2).contiguous().view(*input_shape, -1)

        # 最终输出：官方 attention + 局域势修正
        attn_out = attn_out + delta_v
        attn_out = self.o_proj(attn_out)

        # 9) 保存中间量，便于分析
        self._last_q = q
        self._last_k = k
        self._last_v = v
        self._last_k_rep = k_rep
        self._last_v_rep = v_rep
        self._last_attn_weights = attn_weights

        return attn_out, attn_weights
