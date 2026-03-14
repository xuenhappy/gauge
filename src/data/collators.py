import torch
class QACollator:
    def __init__(self, tokenizer, max_seq_length=4096, train_on_inputs=False):
        self.tokenizer=tokenizer; self.max_seq_length=max_seq_length; self.train_on_inputs=train_on_inputs
    def __call__(self,batch):
        ids=[]; labels=[]; metadata=[]
        for ex in batch:
            full_text=ex['prompt']+ex['target']
            prompt_ids=self.tokenizer(ex['prompt'], add_special_tokens=False)['input_ids']
            full_ids=self.tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=self.max_seq_length)['input_ids']
            lab=full_ids.copy()
            if not self.train_on_inputs:
                plen=min(len(prompt_ids), len(full_ids)); lab[:plen]=[-100]*plen
            ids.append(torch.tensor(full_ids)); labels.append(torch.tensor(lab)); metadata.append(ex)
        input_ids=torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels=torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_mask=(input_ids!=self.tokenizer.pad_token_id).long()
        return {'input_ids': input_ids, 'labels': labels, 'attention_mask': attention_mask, 'metadata': metadata}
