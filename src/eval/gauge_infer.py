import os, json, glob, yaml, argparse, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from ..models.patch_qwen_gauge import patch_qwen_with_gauge
from ..train.prompts import build_prompt_from_style
from ..utils.config import align_model_and_tokenizer

def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _load_single_safetensors(path):
    from safetensors.torch import load_file
    return load_file(path)


def _load_sharded_safetensors(final_dir):
    with open(os.path.join(final_dir, 'model.safetensors.index.json'), 'r', encoding='utf-8') as f:
        index = json.load(f)
    state = {}
    for shard in sorted(set(index['weight_map'].values())):
        state.update(_load_single_safetensors(os.path.join(final_dir, shard)))
    return state


def _load_torch_bin(path):
    obj = torch.load(path, map_location='cpu')
    return obj['state_dict'] if isinstance(obj, dict) and 'state_dict' in obj else obj


def load_checkpoint_state_dict(final_dir):
    if os.path.exists(os.path.join(final_dir, 'model.safetensors')):
        return _load_single_safetensors(os.path.join(final_dir, 'model.safetensors'))
    if os.path.exists(os.path.join(final_dir, 'model.safetensors.index.json')):
        return _load_sharded_safetensors(final_dir)
    if os.path.exists(os.path.join(final_dir, 'pytorch_model.bin')):
        return _load_torch_bin(os.path.join(final_dir, 'pytorch_model.bin'))
    raise FileNotFoundError(final_dir)


def build_gauge_model_from_run(run_dir, device='cuda'):
    cfg = load_yaml(os.path.join(run_dir, 'config.yaml'))
    final_dir = os.path.join(run_dir, 'final')
    model_name = cfg['model']['base_model_name_or_path']
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=cfg['model'].get('trust_remote_code', True))
    model = AutoModelForCausalLM.from_pretrained(model_name,
        torch_dtype=torch.bfloat16 if cfg['model']['torch_dtype'] == 'bfloat16' else torch.float16,
        trust_remote_code=cfg['model'].get('trust_remote_code', True),
        attn_implementation=cfg['model'].get('attn_implementation', 'eager'))
    model = patch_qwen_with_gauge(model, cfg['gauge'])
    align_model_and_tokenizer(model, tokenizer)
    model.load_state_dict(load_checkpoint_state_dict(final_dir), strict=False)
    model.to(device).eval()
    return model, tokenizer, cfg


def build_qa_prompt_from_cfg(cfg, context, question):
    return build_prompt_from_style(cfg['data']['prompt_style'])({
        'id': 'infer_case',
        'context': context,
        'question': question,
        'answer': ''
    })


@torch.no_grad()
def infer_answer(model, tokenizer, prompt, max_new_tokens=128, temperature=0.0, top_p=1.0, do_sample=False):
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    outputs = model.generate(**inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample)
    gen = outputs[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--context')
    parser.add_argument('--question')
    parser.add_argument('--interactive', action='store_true')
    args = parser.parse_args()
    model, tokenizer, cfg = build_gauge_model_from_run(args.run_dir)
    if args.interactive:
        while True:
            context = input('Context: ').strip()
            question = input('Question: ').strip()
            if not question: break
            print(
                infer_answer(model,
                tokenizer,
                build_qa_prompt_from_cfg(cfg, context, question),
                max_new_tokens=cfg['evaluation'].get('generation_max_new_tokens', 128),
                temperature=cfg['evaluation'].get('temperature', 0.0),
                top_p=cfg['evaluation'].get('top_p', 1.0),
                do_sample=cfg['evaluation'].get('do_sample', False)))
    else:
        print(
            infer_answer(model,
            tokenizer,
            build_qa_prompt_from_cfg(cfg, args.context, args.question),
            max_new_tokens=cfg['evaluation'].get('generation_max_new_tokens', 128),
            temperature=cfg['evaluation'].get('temperature', 0.0),
            top_p=cfg['evaluation'].get('top_p', 1.0),
            do_sample=cfg['evaluation'].get('do_sample', False)))


if __name__ == '__main__':
    main()
