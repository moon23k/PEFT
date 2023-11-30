import os, json, argparse, torch
from transformers import set_seed, AutoTokenizer
from module import load_model, load_dataset, set_trainer




class Config(object):
    def __init__(self, peft):

        self.peft = peft
        self.mname = 'bert-base-uncased'
        
        self.lr = 1e-5
        self.n_epochs = 5
        self.batch_size = 32
        self.max_len = 512
        self.num_labels = 4

        self.num_virtual_tokens = 10
        self.encoder_hidden_size = 128

        self.ckpt = f"ckpt/{peft}"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    def update_param_info(self, model):
        all_params = 0
        trainable_params = 0
        
        for _, param in model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        setattr(self, 'all_params', all_params)
        setattr(self, 'trainable_params', trainable_params)
        setattr(self, 'trainable_perc', round(100 * trainable_params / all_params, 2))


    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")
            


def main(peft):

    #Prerequisites
    set_seed(42)
    config = Config(peft)
    model, param_dict = load_model(config)
    tokenizer = AutoTokenizer.from_pretrained(
        config.mname, model_max_length=config.max_len
    )

    #Load datasets
    train_dataset = load_dataset(tokenizer, 'train')
    valid_dataset = load_dataset(tokenizer, 'valid')
    test_dataset = load_dataset(tokenizer, 'test')

    #Load Trainer
    trainer = set_trainer(config, model, tokenizer, train_dataset, valid_dataset)    
    
    #Training
    torch.cuda.reset_max_memory_allocated()
    train_output = trainer.train()
    train_metrics = train_output.metrics
    gpu_memory = torch.cuda.memory_allocated()
    gpu_max_memory = torch.cuda.max_memory_allocated()
    
    #Evaluating
    eval_output = trainer.evaluate(test_dataset)

    report = {
        'all_params': param_dict['all_params'],
        'trainable_params': param_dict['trainable_params'],
        'trainable_perc': param_dict['trainable_perc'],
        'num_epochs': train_metrics['epoch'],
        'train_time': round(train_metrics['train_runtime'], 1),
        'train_samples_per_second': round(train_metrics['train_samples_per_second'], 1),
        'train_loss': round(train_metrics['train_loss'], 2),
        'accuracy': eval_output['eval_accuracy'],
        'gpu_memory': f"{gpu_memory / (1024 ** 3):.2f} GB",
        'gpu_max_memory': f"{gpu_max_memory / (1024 ** 3):.2f} GB",
    }

    with open(f"report/{peft}.json", 'w') as f:
        json.dump(report, f)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-peft', required=True)
    
    args = parser.parse_args()
    assert args.peft.lower() in [
        'vanilla', 'prompt_tuning', 
        'prefix_tuning', 'p_tuning', 
        'lora', 'ia3'
    ]

    os.makedirs(f'ckpt/{args.peft}', exist_ok=True)
    main(args.peft)