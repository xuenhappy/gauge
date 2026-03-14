"""最小 Gauge-Qwen attention patch。需要时请根据本地 transformers 版本调整。"""
from typing import Optional, Tuple
import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, apply_rotary_pos_emb, repeat_kv
from src.models.covariant_gauge_adapter_v2 import CovariantGaugeAdapter
class GaugeQwen2Attention(Qwen2Attention):
    def __init__(self, config, layer_idx, gauge_rank=16, gauge_dropout=0.0, gauge_use_layernorm=True, gauge_smoothness_weight=0.0, gauge_field_l2_weight=0.0, gauge_init_scale=1e-3, return_base_attn_weights=False):
        super().__init__(config=config, layer_idx=layer_idx)
        self.gauge_adapter=CovariantGaugeAdapter(config.hidden_size, config.num_attention_heads, rank=gauge_rank, dropout=gauge_dropout, use_layernorm=gauge_use_layernorm, smoothness_weight=gauge_smoothness_weight, field_l2_weight=gauge_field_l2_weight, init_scale=gauge_init_scale)
    def forward(self, hidden_states: torch.Tensor, position_embeddings: Tuple[torch.Tensor, torch.Tensor], attention_mask: Optional[torch.Tensor], past_key_value=None, **kwargs):
        input_shape=hidden_states.shape[:-1]; hidden_shape=(*input_shape,-1,self.head_dim)
        q=self.q_proj(hidden_states).view(hidden_shape).transpose(1,2); k=self.k_proj(hidden_states).view(hidden_shape).transpose(1,2); v=self.v_proj(hidden_states).view(hidden_shape).transpose(1,2)
        cos,sin=position_embeddings; q,k=apply_rotary_pos_emb(q,k,cos,sin)
        if past_key_value is not None: k,v=past_key_value.update(k,v,self.layer_idx)
        k_rep=repeat_kv(k,self.num_key_value_groups); v_rep=repeat_kv(v,self.num_key_value_groups)
        base_out, attn_weights = super().forward(hidden_states=hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask, past_key_value=past_key_value, **kwargs)
        return base_out + self.gauge_adapter(hidden_states, q, k_rep, v_rep, attention_mask), attn_weights
