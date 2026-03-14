from src.models.qwen_gauge_attention import GaugeQwen2Attention

def patch_qwen_with_gauge(model, gauge_cfg):
    target_layers=set(gauge_cfg['target_layers'])
    for i, layer in enumerate(model.model.layers):
        if i not in target_layers: continue
        old=layer.self_attn
        new=GaugeQwen2Attention(model.config, i, gauge_rank=gauge_cfg.get('rank',16), gauge_dropout=gauge_cfg.get('dropout',0.0), gauge_use_layernorm=gauge_cfg.get('use_layernorm',True), gauge_smoothness_weight=gauge_cfg.get('smoothness_weight',0.0), gauge_field_l2_weight=gauge_cfg.get('field_l2_weight',0.0), gauge_init_scale=gauge_cfg.get('init_scale',1e-3))
        new.load_state_dict(old.state_dict(), strict=False)
        layer.self_attn=new.to(device=next(old.parameters()).device, dtype=next(old.parameters()).dtype)
    return model

def freeze_base_model_except_gauge(model):
    for _,p in model.named_parameters(): p.requires_grad=False
    for m in model.modules():
        if hasattr(m,'gauge_adapter'):
            for p in m.gauge_adapter.parameters(): p.requires_grad=True
    return model
