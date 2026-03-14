import os, json, torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from ..data.qa_dataset import QADataset
from ..train.prompts import build_prompt_from_style
from ..metrics.qa_metrics import compute_em_f1_rougel


@torch.no_grad()
def generate_answer(model, tokenizer, prompt, max_new_tokens=128):
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0, top_p=1.0)
    return tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()


def run_evaluation(cfg, checkpoint_path, output_dir):
    os.makedirs(os.path.join(output_dir, 'metrics'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'analysis'), exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path,
        trust_remote_code=cfg['model'].get('trust_remote_code', True))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path,
        torch_dtype=torch.bfloat16 if cfg['model']['torch_dtype'] == 'bfloat16' else torch.float16,
        trust_remote_code=cfg['model'].get('trust_remote_code', True),
        attn_implementation=cfg['model'].get('attn_implementation', 'eager')).cuda().eval()
    ds = QADataset(cfg['data']['test_file'], build_prompt_from_style(cfg['data']['prompt_style']))
    preds = []
    refs = []
    rows = []
    for ex in tqdm(ds):
        pred = generate_answer(model,
            tokenizer,
            ex['prompt'],
            max_new_tokens=cfg['evaluation']['generation_max_new_tokens'])
        preds.append(pred)
        refs.append(ex['answer'])
        rows.append({'id': ex['id'], 'question': ex['question'], 'answer': ex['answer'], 'prediction': pred})
    metrics = compute_em_f1_rougel(preds, refs)
    with open(os.path.join(output_dir, 'metrics', 'test_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, 'analysis', 'predictions.jsonl'), 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics
