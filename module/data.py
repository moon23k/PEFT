import json, torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence



class Dataset(torch.utils.data.Dataset):
    def __init__(self, task, split):
        super().__init__()
        self.data = self.load_data(task, split)

    @staticmethod
    def load_data(task, split):
        with open(f"data/{task}/{split}.json", 'r') as f:
            data = json.load(f)
        if split=='train':
            return data[::3]
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_ids = self.data[idx]['input_ids']
        attention_mask = self.data[idx]['attention_mask']
        labels = self.data[idx]['labels']
        return input_ids, attention_mask, labels



def load_dataloader(config, split):
    global pad_id
    pad_id = config.pad_id    

    def base_collate(batch):
        input_ids_batch = []
        attention_mask_batch = []
        labels_batch = []

        for input_ids, attention_mask, labels in batch:
            input_ids_batch.append(torch.LongTensor(input_ids))
            attention_mask_batch.append(torch.LongTensor(attention_mask))
            labels_batch.append(torch.LongTensor(labels))

        input_ids_batch = pad_sequence(input_ids_batch,
                                       batch_first=True,
                                       padding_value=pad_id)

        attention_mask_batch = pad_sequence(attention_mask_batch,
                                            batch_first=True,
                                            padding_value=pad_id)        
        
        labels_batch = pad_sequence(labels_batch,
                                    batch_first=True,
                                    padding_value=pad_id)

        labels_batch[labels_batch == 0] = -100

        return {'input_ids': input_ids_batch, 
                'attention_mask': attention_mask_batch,
                'labels': labels_batch}



    return DataLoader(Dataset(config.task, split), 
                      batch_size=config.batch_size, 
                      shuffle=True,
                      collate_fn=base_collate,
                      num_workers=2)