import os, yaml
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from ..data.qa_dataset import QADataset
from ..data.collators import QACollator
from ..train.prompts import build_prompt_from_style
from ..eval.evaluate import run_evaluation
from ..utils.config import align_model_and_tokenizer


def run_lora(cfg):
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
    if cfg['model'].get('gradient_checkpointing', False): 
        model.gradient_checkpointing_enable()
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=cfg['lora']['r'],
        lora_alpha=cfg['lora']['alpha'],
        lora_dropout=cfg['lora']['dropout'],
        target_modules=cfg['lora']['target_modules'],
        bias='none')
    model = get_peft_model(model, peft_config)
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
    trainer = Trainer(model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer)
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
    return final_dir
