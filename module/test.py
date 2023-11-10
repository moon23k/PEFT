import torch, evaluate



class Tester:
    def __init__(self, config, model, tokenizer, test_dataloader):
        super(Tester, self).__init__()
        
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader = test_dataloader

        self.device = config.device        
        self.metric_module = evaluate.load('bleu')
        


    def test(self):
        score = 0.0         
        self.model.eval()

        with torch.no_grad():
            for batch in self.dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = self.tokenize(batch['labels'])

                preds = self.model.generate(input_ids=input_ids, attention_mask=attention_mask)
                score += self.evaluate(preds, labels)

        txt = f"TEST Result on {self.task.upper()} with {self.model_type.upper()} model"
        txt += f"\n-- Score: {round(score/len(self.dataloader), 2)}\n"
        print(txt)


    def evaluate(self, preds, labels):
        pred = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        score = self.metric_module.compute(
            predictions=preds, 
            references =[[l] for l in labels]
        )['bleu']

        return score * 100