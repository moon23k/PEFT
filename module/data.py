import json, torch
from torch.utils.data import DataLoader



class Dataset(torch.utils.data.Dataset):
    def __init__(self, task, split):
        super().__init__()
        self.task = task
        self.data = self.load_data(task, split)

    @staticmethod
    def load_data(task, split):
        with open(f"data/{task}/{split}.json", 'r') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src = self.data[idx]['src']
        trg = self.data[idx]['trg']

        if self.task == 'sum':
            return src, 'summarize: ' + trg
        return src, trg



class Collator(object):
    def __init__(self, config, tokenizer):
        self.task = config.task
        self.tokenizer = tokenizer
        self.pad_id = config.pad_id

    def __call__(self, batch):
        src_batch, trg_batch = [], []

        for src, trg in batch:
            src_batch.append(src)
            trg_batch.append(trg)

        src_tokenized = self.tokenizer(
            src_batch, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        )

        labels = self.tokenizer(
            trg_batch, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        ).input_ids
        
        if self.task == 'sum':
            labels[labels==self.pad_id] = -100

        return {'input_ids': src_tokenized.input_ids, 
                'attention_mask': src_tokenized.attention_mask,
                'labels': labels}



def load_dataloader(config, tokenizer, split):
    return DataLoader(Dataset(config.task, split), 
                      batch_size=config.batch_size if config.mode=='train' else 1, 
                      shuffle=True if config.mode=='train' else False,
                      collate_fn=Collator(config, tokenizer),
                      pin_memory=True,
                      num_workers=2)