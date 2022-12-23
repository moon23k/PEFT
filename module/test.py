import torch, math, time
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_metric
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
            self.metric_module = load_metric('bleu')

        elif self.task == 'dialog':
            self.metric_name = 'Similarity'
            self.metric_model = BertModel.from_pretrained('bert-base-uncased')
            self.metric_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        
        elif self.task == 'sum':
            self.metric_name = 'ROUGE'
            self.metric_module = load_metric('rouge')


    def loss_test(self):
        tot_loss = 0.0
        
        with torch.no_grad():
            for _, batch in enumerate(self.dataloader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)                
                
                with torch.autocast(device_type=self.device_type, dtype=torch.float16):
                    loss = self.model(input_ids = input_ids, 
                                      attention_mask = attention_mask,
                                      labels = labels)[0]

                tot_loss += loss
            tot_loss /= len(self.dataloader)
        
        print(f'Loss Test Results on {self.task} Task')
        print(f">> Loss: {tot_loss:.3f} | PPL: {math.exp(tot_loss):.2f}\n")


    
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


    def metric_test(self):
        metric_results = []
        batch = next(iter(self.dataloader))
        
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        labels[labels ==-100] = 0

        greedy_pred = self.model.generate(input_ids=input_ids, 
                                          attention_mask=attention_mask, 
                                          max_length=512)
        beam_pred = self.model.generate(input_ids=input_ids, 
                                        attention_mask=attention_mask, 
                                        max_length=512, 
                                        num_beams=4)

        for idx in range(self.batch_size):
            temp_dict = dict()

            temp_dict['input_seq'] = self.tokenizer.decode(input_ids[idx], skip_special_tokens=True)
            temp_dict['label_seq'] = self.tokenizer.decode(labels[idx], skip_special_tokens=True)

            temp_dict['greedy_pred'] = self.tokenizer.decode(greedy_pred[idx], skip_special_tokens=True)
            temp_dict['beam_pred'] = self.tokenizer.decode(beam_pred[idx], skip_special_tokens=True)

            temp_dict['greedy_metric'] = self.metric_score(temp_dict['greedy_pred'], temp_dict['label_seq'])
            temp_dict['beam_metric'] = self.metric_score(temp_dict['beam_pred'], temp_dict['label_seq'])
            
            metric_results.append(temp_dict)
        
        metric_results = sorted(metric_results, key=lambda d: d['beam_metric'])
        
        #print_dicts takes only three elems from metric_results
        print_dicts = [metric_results[0]] + \
                      [metric_results[self.batch_size // 2]] + \
                      [metric_results[-1]]


        print(f'Metric Test on {self.task} model')
        for d in print_dicts:
            print(f">> Input Sequence: {d['input_seq']}")
            print(f">> Label Sequence: {d['label_seq']}")
            
            print(f">> Greedy Sequence: {d['greedy_pred']}")
            print(f">> Beam   Sequence : {d['beam_pred']}")
            
            print(f">> Greedy {self.metric_name.upper()} Score: {d['greedy_metric']:.2f}")
            print(f">> Beam   {self.metric_name.upper()} Score : {d['beam_metric']:.2f}\n")