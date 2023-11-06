import json, torch
from torch.utils.data import DataLoader




class Dataset(torch.utils.data.Dataset):

    def __init__(self, split):
        super().__init__()
        self.data = self.load_data(split)


    @staticmethod
    def load_data(split):
        with open(f"data/{split}.json", 'r') as f:
            data = json.load(f)
        return data


    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        return self.data[idx]['x'], self.data[idx]['y']



class Collator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def tokenize(self, x):
        return self.tokenizer(x, padding=True, truncation=True, return_tensors='pt')

    def __call__(self, batch):
        x_batch, y_batch = zip(*batch)     

        model_inputs = self.tokenize(x_batch)
        labels = self.tokenize(y_batch).input_ids
        labels[labels == self.tokenizer.pad_token_id] = -100 
        model_inputs['labels'] = labels

        return model_inputs




def load_dataloader(config, tokenizer, split):
    return DataLoader(
        Dataset(split), 
        batch_size=config.batch_size, 
        shuffle=split == 'train',
        collate_fn=Collator(tokenizer),
        pin_memory=True,
        num_workers=2
    )