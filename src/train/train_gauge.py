import os, yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from ..data.qa_dataset import QADataset
from ..data.collators import QACollator
from ..train.prompts import build_prompt_from_style
from ..train.trainer_gauge import GaugeTrainer
from ..models.patch_qwen_gauge import patch_qwen_with_gauge, freeze_base_model_except_gauge
from ..utils.config import align_model_and_tokenizer
from ..eval.evaluate import run_evaluation

def build_base_reference_model(cfg):
    if not cfg['gauge'].get('use_base_kl', False): 
        return None
    model = AutoModelForCausalLM.from_pretrained(cfg['model']['base_model_name_or_path'],
        torch_dtype='auto',
        trust_remote_code=cfg['model'].get('trust_remote_code', True),
        attn_implementation=cfg['model'].get('attn_implementation', 'eager'))
    model.eval()
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad = False
    return model


def run_gauge(cfg):
    out = cfg['experiment']['output_dir']
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'config.yaml'), 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    model_name = cfg['model']['base_model_name_or_path']
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=cfg['model'].get('trust_remote_code', True))
    model = AutoModelForCausalLM.from_pretrained(model_name,
        torch_dtype='auto',
        trust_remote_code=cfg['model'].get('trust_remote_code', True),
        attn_implementation=cfg['model'].get('attn_implementation', 'eager'))
    align_model_and_tokenizer(model, tokenizer)
    if cfg['model'].get('gradient_checkpointing', False): model.gradient_checkpointing_enable()
    model = patch_qwen_with_gauge(model, cfg['gauge'])
    model = freeze_base_model_except_gauge(model)
    model.config.use_cache = False
    train_dataset = QADataset(cfg['data']['train_file'], build_prompt_from_style(cfg['data']['prompt_style']))
    eval_dataset = QADataset(cfg['data']['validation_file'], build_prompt_from_style(cfg['data']['prompt_style']))
    collator = QACollator(tokenizer,
        max_seq_length=cfg['data']['max_seq_length'],
        train_on_inputs=cfg['data'].get('train_on_inputs', False))

    
    args = TrainingArguments(output_dir=out,
        per_device_train_batch_size=cfg['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=cfg['training']['per_device_eval_batch_size'],
        gradient_accumulation_steps=cfg['training']['gradient_accumulation_steps'],
        num_train_epochs=cfg['training']['num_train_epochs'],
        learning_rate=cfg['training']['learning_rate'],
        weight_decay=cfg['training']['weight_decay'],
        warmup_steps=int(len(train_dataset) * cfg['training']['num_train_epochs'] * cfg['training']['warmup_ratio']),
        logging_steps=cfg['training']['log_interval'],
        eval_steps=cfg['training']['eval_interval'],
        save_steps=cfg['training']['save_interval'],
        save_total_limit=cfg['training']['save_total_limit'],
        bf16=cfg['training']['bf16'],
        fp16=cfg['training']['fp16'],
        max_grad_norm=cfg['training']['max_grad_norm'],
        eval_strategy='steps',
        save_strategy='steps',
        report_to=['tensorboard'],
        remove_unused_columns=False,
        ddp_find_unused_parameters=False)
    trainer = GaugeTrainer(model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        gauge_config=cfg['gauge'],
        base_reference_model=build_base_reference_model(cfg),
        base_kl_weight=cfg['gauge'].get('base_kl_weight', 0.0),
        output_dir=out)
    trainer.train()
    final_dir = os.path.join(out, 'final')
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    run_evaluation(
        cfg=cfg,
        checkpoint_path=final_dir,
        output_dir=out,
        model=trainer.model,
        tokenizer=tokenizer,
    )
    for name, module in model.named_modules():
        if hasattr(module, "gauge_adapter"):
            ga = module.gauge_adapter
            print(
                name,
                "g_attn_norm=", float(ga.g_attn.detach().norm().cpu()),
                "g_rel_norm=", float(ga.g_rel.detach().norm().cpu()),
                "g_val_norm=", float(ga.g_val.detach().norm().cpu()),
            )
    trainer.dump_gauge_stats(tag='final')
    return final_dir
