import torch, datasets



class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encoding = encodings
        self.labels = labels

    def __getitem__(self, idx):
        data = {key: val[idx] for key, val in self.encoding.items()}
        data['labels'] = torch.tensor(self.labels[idx]).long()

        return data

    def __len__(self):
        return len(self.labels)



def load_dataset(tokenizer, split):
    dataset = datasets.Dataset.from_json(f'data/{split}.json')
    encodings = tokenizer(dataset['x'], padding=True, truncation=True, return_tensors='pt')
    dataset = Dataset(encodings, dataset['y'])
    return dataset