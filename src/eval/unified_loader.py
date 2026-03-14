import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from src.models.registry import load_run_config, detect_run_method, get_final_dir
from src.models.patch_qwen_gauge import patch_qwen_with_gauge
from src.eval.gauge_infer import load_checkpoint_state_dict
from src.train.prompts import build_prompt_from_style


def _dtype(cfg):
    return torch.bfloat16 if cfg['model']['torch_dtype'] == 'bfloat16' else torch.float16


def load_frozen_model(run_dir, device='cuda'):
    cfg = load_run_config(run_dir)
    model_name = cfg['model']['base_model_name_or_path']
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=cfg['model'].get('trust_remote_code', True))
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name,
        torch_dtype=_dtype(cfg),
        trust_remote_code=cfg['model'].get('trust_remote_code', True),
        attn_implementation=cfg['model'].get('attn_implementation', 'eager')).to(device).eval()
    return model, tokenizer, cfg


def load_lora_model(run_dir, device='cuda'):
    cfg = load_run_config(run_dir)
    model_name = cfg['model']['base_model_name_or_path']
    final_dir = get_final_dir(run_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=cfg['model'].get('trust_remote_code', True))
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(model_name,
        torch_dtype=_dtype(cfg),
        trust_remote_code=cfg['model'].get('trust_remote_code', True),
        attn_implementation=cfg['model'].get('attn_implementation', 'eager'))
    model = PeftModel.from_pretrained(base, final_dir).to(device).eval()
    return model, tokenizer, cfg


def load_gauge_model(run_dir, device='cuda'):
    cfg = load_run_config(run_dir)
    model_name = cfg['model']['base_model_name_or_path']
    final_dir = get_final_dir(run_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=cfg['model'].get('trust_remote_code', True))
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name,
        torch_dtype=_dtype(cfg),
        trust_remote_code=cfg['model'].get('trust_remote_code', True),
        attn_implementation=cfg['model'].get('attn_implementation', 'eager'))
    model = patch_qwen_with_gauge(model, cfg['gauge'])
    model.load_state_dict(load_checkpoint_state_dict(final_dir), strict=False)
    model = model.to(device).eval()
    return model, tokenizer, cfg


def build_prompt(cfg, context, question):
    return build_prompt_from_style(cfg['data']['prompt_style'])({
        'id': 'infer_case',
        'context': context,
        'question': question,
        'answer': ''
    })


@torch.no_grad()
def generate_answer(model, tokenizer, prompt, cfg):
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    outputs = model.generate(**inputs,
        max_new_tokens=cfg['evaluation'].get('generation_max_new_tokens', 128),
        temperature=cfg['evaluation'].get('temperature', 0.0),
        top_p=cfg['evaluation'].get('top_p', 1.0),
        do_sample=cfg['evaluation'].get('do_sample', False))
    gen = outputs[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


class UnifiedQAPipeline:

    def __init__(self, run_dir, device='cuda'):
        self.method = detect_run_method(run_dir)
        if self.method == 'frozen': self.model, self.tokenizer, self.cfg = load_frozen_model(run_dir, device=device)
        elif self.method == 'lora': self.model, self.tokenizer, self.cfg = load_lora_model(run_dir, device=device)
        elif self.method == 'gauge': self.model, self.tokenizer, self.cfg = load_gauge_model(run_dir, device=device)
        else: raise ValueError(self.method)

    def answer(self, context, question):
        return generate_answer(self.model, self.tokenizer, build_prompt(self.cfg, context, question), self.cfg)
