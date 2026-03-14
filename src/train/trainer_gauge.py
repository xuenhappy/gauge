import os, json, torch
import torch.nn.functional as F
from transformers import Trainer


class GaugeTrainer(Trainer):

    def __init__(self,
        *args,
        gauge_config=None,
        base_reference_model=None,
        base_kl_weight=0.0,
        output_dir=None,
        **kwargs):
        super().__init__(*args, **kwargs)
        self.gauge_config = gauge_config or {}
        self.base_reference_model = base_reference_model
        self.base_kl_weight = base_kl_weight
        self.output_dir = output_dir
        if self.base_reference_model is not None:
            self.base_reference_model.to(self.model.device).eval()

    def _collect_gauge_reg_loss(self, model):
        reg = None
        for m in model.modules():
            if hasattr(m, 'gauge_adapter'):
                cur = m.gauge_adapter.regularization_loss()
                reg = cur if reg is None else reg + cur
        return reg if reg is not None else torch.tensor(0.0, device=self.model.device)

    @torch.no_grad()
    def _compute_base_kl(self, inputs, model_outputs):
        if self.base_reference_model is None or self.base_kl_weight <= 0:
            return torch.tensor(0.0, device=self.model.device)
        ref = self.base_reference_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        ref_probs = F.log_softmax(ref.logits, dim=-1).exp()
        cur_log_probs = F.log_softmax(model_outputs.logits, dim=-1)
        valid = (inputs['labels'] != -100).unsqueeze(-1)
        kl = F.kl_div(cur_log_probs, ref_probs, reduction='none', log_target=False).sum(dim=-1, keepdim=True)
        return (kl * valid).sum() / valid.sum().clamp_min(1)

    def dump_gauge_stats(self, tag='latest'):
        if self.output_dir is None:
            return
        rows = []
        for name, m in self.model.named_modules():
            if hasattr(m, 'gauge_adapter'):
                ga = m.gauge_adapter
                row = {
                    'module': name,
                    'g_attn_norm': float(ga.g_attn.detach().norm().cpu()),
                    'g_val_norm': float(ga.g_val.detach().norm().cpu()),
                    'g_rel_norm': float(ga.g_rel.detach().norm().cpu())
                }
                if getattr(ga, '_last_fields', None) is not None:
                    row['A_q_norm'] = float(ga._last_fields['A_q_raw'].detach().norm().cpu())
                    row['A_k_norm'] = float(ga._last_fields['A_k_raw'].detach().norm().cpu())
                    row['A_v_norm'] = float(ga._last_fields['A_v_raw'].detach().norm().cpu())
                rows.append(row)
        os.makedirs(os.path.join(self.output_dir, 'analysis'), exist_ok=True)
        with open(os.path.join(self.output_dir, 'analysis', f'gauge_stats_{tag}.json'), 'w', encoding='utf-8') as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['labels'])
        ce = outputs.loss
        reg = self._collect_gauge_reg_loss(model)
        base_kl = self._compute_base_kl(inputs, outputs)
        total = ce + reg + self.base_kl_weight * base_kl
        self.log({
            'loss_ce': float(ce.detach().cpu()),
            'loss_reg': float(reg.detach().cpu()),
            'loss_base_kl': float(base_kl.detach().cpu()),
            'loss_total': float(total.detach().cpu())
        })
        return (total, outputs) if return_outputs else total

    def evaluate(self, *args, **kwargs):
        metrics = super().evaluate(*args, **kwargs)
        self.dump_gauge_stats(tag=f"step_{int(self.state.global_step):06d}")
        return metrics
