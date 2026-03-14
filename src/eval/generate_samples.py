import os, json, argparse
from src.data.qa_dataset import QADataset
from src.train.prompts import build_prompt_from_style
from src.eval.unified_loader import UnifiedQAPipeline

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--sample_file', required=True); parser.add_argument('--prompt_style', default='qa_standard'); parser.add_argument('--output_dir', required=True); parser.add_argument('--run', action='append', required=True); args=parser.parse_args(); os.makedirs(args.output_dir, exist_ok=True)
    dataset=QADataset(args.sample_file, build_prompt_from_style(args.prompt_style)); pipes={}
    for item in args.run:
        name, run_dir = item.split('=',1); pipes[name]=UnifiedQAPipeline(run_dir)
    rows=[]; lines=[]
    for ex in dataset:
        row={'id': ex['id'], 'question': ex['question'], 'ground_truth': ex['answer'], 'predictions': {}}
        lines += ['='*80, f"ID: {ex['id']}", f"QUESTION: {ex['question']}", f"GROUND TRUTH: {ex['answer']}", '']
        for name, pipe in pipes.items():
            pred=pipe.answer(ex['context'], ex['question']); row['predictions'][name]=pred; lines.append(f'[{name}] {pred}')
        lines.append(''); rows.append(row)
    with open(os.path.join(args.output_dir,'sample_generations.json'),'w',encoding='utf-8') as f: json.dump(rows,f,indent=2,ensure_ascii=False)
    with open(os.path.join(args.output_dir,'sample_generations.txt'),'w',encoding='utf-8') as f: f.write('\n'.join(lines))
if __name__=='__main__': main()
