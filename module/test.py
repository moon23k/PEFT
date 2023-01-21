import torch, math, time, evaluate
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizerFast



class Tester:
    def __init__(self, config, model, test_dataloader, tokenizer):
        super(Tester, self).__init__()
        
        self.model = model
        self.task = config.task
        self.tokenizer = tokenizer
        self.device = config.device
        self.device_type = config.device_type
        self.dataloader = test_dataloader
        self.batch_size = config.batch_size        
        self.vocab_size = model.config.vocab_size
        
        if self.task == 'nmt':
            self.metric_name = 'BLEU'
            self.metric_module = evaluate.load('bleu')

        elif self.task == 'dialog':
            self.metric_name = 'Similarity'
            self.metric_model = BertModel.from_pretrained('bert-base-uncased')
            self.metric_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        
        elif self.task == 'sum':
            self.metric_name = 'ROUGE'
            self.metric_module = load('rouge')


    
    def metric_score(self, pred, label):
        if self.task == 'nmt':
            score = self.metric_module.compute(predictions=[pred.split()], 
                                               references=[[label.split()]])['bleu']

        elif self.task == 'dialog':
            encoding = self.metric_tokenizer([pred, label], padding=True, return_tensors='pt')
            bert_out = self.metric_model(**encoding)[0]

            normalized = F.normalize(bert_out[:, 0, :], p=2, dim=-1)  # Only use of [CLS] token embedding
            dist = normalized.matmul(normalized.T)
            sim_matrix = dist.new_ones(dist.shape) - dist
            score = sim_matrix[0, 1].item()

        elif self.task == 'sum':
            score = self.metric_module.compute(predictions=[pred.split()], 
                                               references=[[label.split()]])['rouge2'].mid.fmeasure            
            
        return score * 100




    def test(self):
        self.model.eval()
        
        start_time = time.time()
        with torch.no_grad():
            for _, batch in tqdm(enumerate(self.dataloader)):   
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                labels[labels ==-100] = 0
                                
                preds = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                            max_new_tokens=300, use_cache=True)
                
                preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
                labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

                metric_module.add_batch(predictions=preds, 
                                        references=[[l] for l in labels])    

        bleu_score = metric_module.compute()['bleu'] * 100

        print('Test Results')
        print(f"  >> {self.metric_name} Score: {bleu_score:.2f}")
        print(f"  >> Spent Time: {self.measure_time(start_time, time.time())}")

