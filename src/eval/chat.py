import argparse
from src.eval.unified_loader import UnifiedQAPipeline

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--run_dir', required=True); parser.add_argument('--device', default='cuda'); parser.add_argument('--interactive', action='store_true'); parser.add_argument('--context'); parser.add_argument('--question'); args=parser.parse_args(); pipe=UnifiedQAPipeline(args.run_dir, device=args.device)
    if args.interactive:
        print(f'[chat] loaded method={pipe.method} from {args.run_dir}')
        while True:
            context=input('Context: ').strip(); question=input('Question: ').strip();
            if not question: break
            print(f"\nAnswer: {pipe.answer(context, question)}\n")
    else:
        print(pipe.answer(args.context, args.question))
if __name__=='__main__': main()
