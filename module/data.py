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
        x = self.data[idx]['x']
        y = self.data[idx]['y']

        return x, y



class Collator(object):
    def __init__(self, config, tokenizer):
        self.task = config.task
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, batch):
        x_batch, y_batch = zip(*batch)

        x_encodings = self.tokenizer(
            x_batch, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        )

        y_encodings = self.tokenizer(
            trg_batch, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        ).input_ids

        labels = y_encodings.input_ids        
        labels[labels==self.pad_id] = -100

        return {'input_ids': x_encodings.input_ids, 
                'attention_mask': x_encodings.attention_mask,
                'labels': labels}



def load_dataloader(config, tokenizer, split):
    return DataLoader(
        Dataset(config.task, split), 
        batch_size=config.batch_size, 
        shuffle=split == 'train',
        collate_fn=Collator(config, tokenizer),
        pin_memory=True,
        num_workers=2
    )