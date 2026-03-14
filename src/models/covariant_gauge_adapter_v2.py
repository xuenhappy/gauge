import math
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CovariantGaugeAdapter(nn.Module):
    """
    Gauge Adapter（bias/value 生成器版）

    输出：
        gauge_bias: [B, H, S, S]
        delta_v:    [B, S, D]

    不负责 attention 主体计算，只负责生成：
        1) 连接修正 DeltaS = gauge_bias
        2) 局域势修正 DeltaV = delta_v
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rank: int = 16,
        dropout: float = 0.0,
        use_layernorm: bool = True,
        smoothness_weight: float = 0.0,
        field_l2_weight: float = 0.0,
        init_scale: float = 1e-3,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.smoothness_weight = smoothness_weight
        self.field_l2_weight = field_l2_weight

        self.pre_norm = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()

        # 生成 A_q, A_k, A_v
        self.field_generator = nn.Sequential(
            nn.Linear(d_model, rank, bias=False),
            nn.SiLU(),
            nn.Linear(rank, 3 * d_model, bias=False),
        )

        # connection 空间投影
        self.aq_proj = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.ak_proj = nn.Linear(self.head_dim, self.head_dim, bias=False)

        # 相对边势向量
        self.rel_bias_vec = nn.Parameter(torch.zeros(num_heads, self.head_dim))

        # value 局域势修正
        self.value_proj = nn.Linear(d_model, d_model, bias=False)

        # 耦合系数
        self.g_attn = nn.Parameter(torch.zeros(num_heads))
        self.g_rel = nn.Parameter(torch.zeros(num_heads))
        self.g_val = nn.Parameter(torch.zeros(d_model))

        self.dropout = nn.Dropout(dropout)

        self._last_fields: Optional[Dict[str, torch.Tensor]] = None
        self._reset_parameters(init_scale)

    def _reset_parameters(self, init_scale: float):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.uniform_(self.rel_bias_vec, a=-0.2, b=0.2)
        nn.init.uniform_(self.g_attn, a=-0.2, b=0.2)
        nn.init.uniform_(self.g_rel, a=-0.2, b=0.2)
        nn.init.zeros_(self.g_val)
    

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        """
        [B, S, D] -> [B, H, S, Hd]
        """
        B, S, D = x.shape
        return x.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

    def _compute_gauge_bias(
            self,
            q_base: torch.Tensor,  # [B, H, S, Hd]
            k_base: torch.Tensor,  # [B, H, S, Hd]
            A_q: torch.Tensor,  # [B, H, S, Hd]
            A_k: torch.Tensor,  # [B, H, S, Hd]
    ) -> torch.Tensor:
        """
        DeltaS = g_attn*(b1+b2) + g_rel*b3
        """
        A_qp = self.aq_proj(A_q)
        A_kp = self.ak_proj(A_k)

        # b1 = <Q_i, A_k(j)>
        b1 = torch.matmul(q_base, A_kp.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # b2 = <A_q(i), K_j>
        b2 = torch.matmul(A_qp, k_base.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # b3 = w_h^T tanh(A_k(j) - A_q(i))
        rel = torch.tanh(A_k.unsqueeze(2) - A_q.unsqueeze(3))  # [B,H,S,S,Hd]
        relv = self.rel_bias_vec.view(1, self.num_heads, 1, 1, self.head_dim)
        b3 = (rel * relv).sum(dim=-1)  # [B,H,S,S]

        gauge_bias = (self.g_attn.view(1, self.num_heads, 1, 1) * (b1 + b2) +
            self.g_rel.view(1, self.num_heads, 1, 1) * b3)
        return gauge_bias

    def forward(
            self,
            hidden_states: torch.Tensor,  # [B, S, D]
            q_base: torch.Tensor,  # [B, H, S, Hd]
            k_base: torch.Tensor,  # [B, H, S, Hd]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回：
            gauge_bias: [B, H, S, S]
            delta_v:    [B, S, D]
        """
        x = self.pre_norm(hidden_states)

        A_q_raw, A_k_raw, A_v_raw = self.field_generator(x).chunk(3, dim=-1)
        A_q = self._split(A_q_raw)
        A_k = self._split(A_k_raw)

        gauge_bias = self._compute_gauge_bias(q_base, k_base, A_q, A_k)

        # 局域势修正
        delta_v = torch.tanh(self.g_val).view(1, 1, self.d_model) * self.value_proj(A_v_raw)
        delta_v = self.dropout(delta_v)

        self._last_fields = {
            "A_q_raw": A_q_raw,
            "A_k_raw": A_k_raw,
            "A_v_raw": A_v_raw,
            "gauge_bias": gauge_bias,
        }

        return gauge_bias, delta_v

    def regularization_loss(self) -> torch.Tensor:
        if self._last_fields is None:
            return torch.tensor(0.0, device=self.g_val.device)

        loss = torch.tensor(0.0, device=self.g_val.device)

        if self.field_l2_weight > 0:
            l2 = 0.0
            for name in ["A_q_raw", "A_k_raw", "A_v_raw"]:
                field = self._last_fields[name]
                l2 = l2 + field.pow(2).mean()
            loss = loss + self.field_l2_weight * l2

        if self.smoothness_weight > 0:
            smooth = 0.0
            for name in ["A_q_raw", "A_k_raw", "A_v_raw"]:
                field = self._last_fields[name]
                if field.size(1) > 1:
                    diff = field[:, 1:, :] - field[:, :-1, :]
                    smooth = smooth + diff.pow(2).mean()
            loss = loss + self.smoothness_weight * smooth

        return loss
