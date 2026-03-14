from typing import Dict, Any

import torch.nn as nn

from .qwen_gauge_attention import GaugeQwen2Attention


def _copy_attention_weights(src_attn: nn.Module, dst_attn: nn.Module):
    missing, unexpected = dst_attn.load_state_dict(src_attn.state_dict(), strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys when copying attention weights: {unexpected}")
    return missing


def patch_qwen_with_gauge(model, gauge_cfg: Dict[str, Any]):
    target_layers = set(gauge_cfg["target_layers"])

    for layer_idx, layer in enumerate(model.model.layers):
        if layer_idx not in target_layers:
            continue

        old_attn = layer.self_attn

        new_attn = GaugeQwen2Attention(
            config=model.config,
            layer_idx=layer_idx,
            gauge_rank=gauge_cfg.get("rank", 16),
            gauge_dropout=gauge_cfg.get("dropout", 0.0),
            gauge_use_layernorm=gauge_cfg.get("use_layernorm", True),
            gauge_smoothness_weight=gauge_cfg.get("smoothness_weight", 0.0),
            gauge_field_l2_weight=gauge_cfg.get("field_l2_weight", 0.0),
            gauge_init_scale=gauge_cfg.get("init_scale", 1e-3),
        )

        _copy_attention_weights(old_attn, new_attn)

        new_attn.to(
            device=next(old_attn.parameters()).device,
            dtype=next(old_attn.parameters()).dtype,
        )

        layer.self_attn = new_attn

    return model


def freeze_base_model_except_gauge(model):
    for _, p in model.named_parameters():
        p.requires_grad = False

    for module in model.modules():
        if hasattr(module, "gauge_adapter"):
            for p in module.gauge_adapter.parameters():
                p.requires_grad = True

    return model


def collect_gauge_modules(model):
    mods = []
    for module in model.modules():
        if hasattr(module, "gauge_adapter"):
            mods.append(module.gauge_adapter)
    return mods
