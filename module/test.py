import torch, evaluate



class Tester:
    def __init__(self, config, model, tokenizer, test_dataloader):
        super(Tester, self).__init__()
        
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader = test_dataloader

        self.task = config.task    
        self.device = config.device
        self.max_len = config.max_len
        

        if self.task == 'nmt':
            self.metric_name = 'BLEU'
            self.metric_module = evaluate.load('bleu')
        else:
            self.metric_name = 'ROUGE'
            self.metric_module = evaluate.load('rouge')


    def test(self):
        self.model.eval()        
        score = 0

        print(f'Test Results on {self.task.upper()}')
        with torch.no_grad():
            for batch in self.dataloader:
            
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].tolist()
        
                pred = self.model.generate(
                    input_ids, 
                    do_sample=False,
                    max_new_tokens=self.max_len, 
                )
                
                score += self.metric_score(pred, labels)
                
        score = round(score/len(self.dataloader), 2)
        
        return score
        


    def metric_score(self, pred, label):
        pred = self.tokenizer.batch_decode(pred, skip_special_tokens=True)[0]
        label = self.tokenizer.batch_decode(label, skip_special_tokens=True)[0]

        #For Translation Task
        if self.task == 'nmt':
            score = self.metric_module.compute(
                predictions=[pred], 
                references=[[label]]
            )['bleu']

        #For Dialgue Generation and Summarization Tasks
        else:        
            score = self.metric_module.compute(
                predictions=[pred], 
                references=[[label]]
            )['rouge2']
