import math, time, torch, evaluate



class Tester:
    def __init__(self, config, model, tokenizer, test_dataloader):
        super(Tester, self).__init__()
        
        self.model = model
        self.task = config.task
        self.tokenizer = tokenizer
        self.device = config.device
        self.dataloader = test_dataloader
        self.beam_size = config.beam_size
        self.max_len = 512

        if self.task == 'nmt':
            self.metric_name = 'BLEU'
            self.metric_module = evaluate.load('bleu')
        else:
            self.metric_name = 'ROUGE'
            self.metric_module = evaluate.load('rouge')


    @staticmethod
    def measure_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_min = int(elapsed_time / 60)
        elapsed_sec = int(elapsed_time - (elapsed_min * 60))
        return f"{elapsed_min}m {elapsed_sec}s"


    def test(self):
        self.model.eval()        
        tot_len = len(self.dataloader)
        greedy_score, beam_score = 0, 0

        print(f'Test Results on {self.task.upper()}')
        with torch.no_grad():
            for batch in self.dataloader:
            
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].tolist()
        
                greedy_pred = self.model.generate(
                    input_ids, do_sample=False,
                    max_new_tokens=self.max_len, 
                )
                
                beam_pred = self.model.generate(
                    input_ids, num_beams=self.beam_size, 
                    max_new_tokens=self.max_len, do_sample=False
                )
                
                greedy_score += self.metric_score(greedy_pred, labels)
                beam_score += self.metric_score(beam_pred, labels)
        
        greedy_score = round(greedy_score / tot_len, 2)
        beam_score = round(beam_score / tot_len, 2)
        
        return greedy_score, beam_score
        


    def metric_score(self, pred, label):
        pred = self.tokenizer.batch_decode(pred, skip_special_tokens=True)[0]
        label = self.tokenizer.batch_decode(label, skip_special_tokens=True)[0]

        #For Translation Task
        if self.task == 'nmt':
            score = self.metric_module.compute(
                predictions=[pred], 
                references=[[label]]
            )['bleu']

        #For Summarization Task
        elif self.task == 'sum':        
            score = self.metric_module.compute(
                predictions=[pred], 
                references=[[label]]
            )['rouge2']

        #For Dialogue Generation Task
        elif self.task == 'dialog':
            encoding = self.metric_tokenizer(
                pred, label, 
                padding=True, 
                truncation=True, 
                return_tensors='pt'
            )

            bert_out = self.metric_model(**encoding)[0]

            normalized = torch.nn.functional.normalize(bert_out[:, 0, :], p=2, dim=-1)
            dist = normalized.matmul(normalized.T)
            sim_matrix = dist.new_ones(dist.shape) - dist
            score = sim_matrix[0, 1].item()

        return (score * 100)
