import math, torch
import torch.nn as nn
import torch.nn.functional as F
class CovariantGaugeAdapter(nn.Module):
    def __init__(self, d_model, num_heads, rank=16, dropout=0.0, use_layernorm=True, smoothness_weight=0.0, field_l2_weight=0.0, init_scale=1e-3):
        super().__init__(); assert d_model % num_heads == 0
        self.d_model=d_model; self.num_heads=num_heads; self.head_dim=d_model//num_heads; self.smoothness_weight=smoothness_weight; self.field_l2_weight=field_l2_weight
        self.pre_norm=nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
        self.field_generator=nn.Sequential(nn.Linear(d_model, rank, bias=False), nn.SiLU(), nn.Linear(rank, 3*d_model, bias=False))
        self.aq_proj=nn.Linear(self.head_dim, self.head_dim, bias=False); self.ak_proj=nn.Linear(self.head_dim, self.head_dim, bias=False); self.value_proj=nn.Linear(d_model, d_model, bias=False)
        self.rel_bias_vec=nn.Parameter(torch.zeros(num_heads, self.head_dim)); self.g_attn=nn.Parameter(torch.zeros(num_heads)); self.g_rel=nn.Parameter(torch.zeros(num_heads)); self.g_val=nn.Parameter(torch.zeros(d_model)); self.out_scale=nn.Parameter(torch.zeros(1)); self.dropout=nn.Dropout(dropout); self._last_fields=None
        for m in self.modules():
            if isinstance(m, nn.Linear): nn.init.normal_(m.weight, mean=0.0, std=init_scale)
        nn.init.zeros_(self.rel_bias_vec); nn.init.zeros_(self.g_attn); nn.init.zeros_(self.g_rel); nn.init.zeros_(self.g_val); nn.init.zeros_(self.out_scale)
    def _split(self,x): B,S,D=x.shape; return x.view(B,S,self.num_heads,self.head_dim).transpose(1,2)
    def _merge(self,x): B,H,S,Hd=x.shape; return x.transpose(1,2).contiguous().view(B,S,H*Hd)
    def forward(self, hidden_states, q_base, k_base, v_base, attention_mask=None):
        x=self.pre_norm(hidden_states); A_q_raw, A_k_raw, A_v_raw=self.field_generator(x).chunk(3, dim=-1); A_q=self._split(A_q_raw); A_k=self._split(A_k_raw)
        A_qp=self.aq_proj(A_q); A_kp=self.ak_proj(A_k); scores=torch.matmul(q_base, k_base.transpose(-2,-1))/math.sqrt(self.head_dim)
        b1=torch.matmul(q_base, A_kp.transpose(-2,-1))/math.sqrt(self.head_dim); b2=torch.matmul(A_qp, k_base.transpose(-2,-1))/math.sqrt(self.head_dim); rel=torch.tanh(A_k.unsqueeze(2)-A_q.unsqueeze(3)); relv=self.rel_bias_vec.view(1,self.num_heads,1,1,self.head_dim); b3=(rel*relv).sum(dim=-1)
        scores=scores+self.g_attn.view(1,self.num_heads,1,1)*(b1+b2)+self.g_rel.view(1,self.num_heads,1,1)*b3
        if attention_mask is not None:
            if attention_mask.dtype==torch.bool: scores=scores.masked_fill(~attention_mask, float('-inf'))
            elif attention_mask.max()<=1 and attention_mask.min()>=0: scores=scores.masked_fill(attention_mask==0, float('-inf'))
            else: scores=scores+attention_mask
        attn_probs=F.softmax(scores, dim=-1); attn_probs=self.dropout(attn_probs); gauge_attn_out=self._merge(torch.matmul(attn_probs, v_base)); delta_v=torch.tanh(self.g_val).view(1,1,self.d_model)*self.value_proj(A_v_raw); delta_out=torch.tanh(self.out_scale)*self.dropout(gauge_attn_out+delta_v); self._last_fields={'A_q_raw':A_q_raw,'A_k_raw':A_k_raw,'A_v_raw':A_v_raw,'attn_probs':attn_probs}; return delta_out
    def regularization_loss(self):
        if self._last_fields is None: return torch.tensor(0.0, device=self.out_scale.device)
        loss=torch.tensor(0.0, device=self.out_scale.device)
        if self.field_l2_weight>0: loss=loss+self.field_l2_weight*sum(self._last_fields[n].pow(2).mean() for n in ['A_q_raw','A_k_raw','A_v_raw'])
        if self.smoothness_weight>0:
            smooth=torch.tensor(0.0, device=self.out_scale.device)
            for n in ['A_q_raw','A_k_raw','A_v_raw']:
                f=self._last_fields[n]
                if f.size(1)>1: smooth=smooth+(f[:,1:,:]-f[:,:-1,:]).pow(2).mean()
            loss=loss+self.smoothness_weight*smooth
        return loss
