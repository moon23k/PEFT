import os, yaml, argparse, torch
from transformers import set_seed, AutoTokenizer
from module import (
    load_dataloader,
    load_model,
    Trainer, 
    Tester
) 




class Config(object):
    def __init__(self, args):

        with open('config.yaml', 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
            for group in params.keys():
                for key, val in params[group].items():
                    setattr(self, key, val)

        self.task = args.task
        self.mode = args.mode
        self.search_method = args.search
        self.ckpt = f"ckpt/{self.task}.pt"

        use_cuda = torch.cuda.is_available()
        self.device_type = 'cuda' \
                           if use_cuda and self.mode != 'inference' \
                           else 'cpu'
        self.device = torch.device(self.device_type)


        if self.task == 'nmt':
            self.mname = "Helsinki-NLP/opus-mt-en-de"
        
        elif self.task == 'dialog':
            self.mname = "facebook/blenderbot_small-90M"
        
        elif self.task == 'sum':
            self.mname = "t5-small"
            self.max_len *= 2
            self.batch_size //= 2


    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")





def inference(config, model, tokenizer):
    print(f'--- Inference Process Started! ---')
    print('[ Type "quit" on user input to stop the Process ]')
    
    while True:
        input_seq = input('\nUser Input Sequence >> ').lower()

        #End Condition
        if input_seq == 'quit':
            print('\n--- Inference Process has terminated! ---')
            break        

        #convert user input_seq into model input_ids
        input_ids = tokenizer(input_seq, return_tensors='pt').to(config.device)

        #Search Output Sequence
        if config.search_method == 'greedy':
            output_seq = model.generate(input_ids)
        else:
            output_seq = model.generate(input_ids, num_beams=config.num_beams)
        print(f"Model Out Sequence >> {output_seq}")       



def main(args):
    set_seed(42)
    config = Config(args)
    model = load_model(config)
    tokenizer = AutoTokenizer.from_pretrained(
        config.mname, 
        model_max_length=config.max_len
    )
    config.update_attr(tokenizer)


    if config.mode == 'train':
        train_dataloader = load_dataloader(config, tokenizer, 'train')
        valid_dataloader = load_dataloader(config, tokenizer, 'valid')
        trainer = Trainer(config, model, train_dataloader, valid_dataloader)
        trainer.train()


    if config.mode == 'test':
        test_dataloader = load_dataloader(config, tokenizer, 'test')
        tester = Tester(config, model, test_dataloader, tokenizer)
        tester.test()
    
    elif config.mode == 'inference':
        inference(config, model, tokenizer)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', required=True)
    parser.add_argument('-mode', required=True)
    parser.add_argument('-search', default='greedy', required=False)
    
    args = parser.parse_args()
    assert args.task in ['nmt', 'dialog', 'sum']
    assert args.mode in ['train', 'test', 'inference']

    if args.task == 'inference':
        assert args.search in ['greedy', 'beam']

    main(args)