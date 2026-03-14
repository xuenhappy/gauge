import json
from torch.utils.data import Dataset

class QADataset(Dataset):
    def __init__(self, path, prompt_builder):
        self.samples=[]; self.prompt_builder=prompt_builder
        with open(path,'r',encoding='utf-8') as f:
            for line in f:
                ex=json.loads(line); ex['prompt']=prompt_builder(ex); self.samples.append(ex)
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        ex=self.samples[idx]
        return {**ex, 'target': ex['answer']}
