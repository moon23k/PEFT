from datasets import Dataset
from transformers import (
    DataCollatorForSeq2Seq, 
    TrainingArguments, 
    Trainer
)




def set_training_args(config):
    training_args_dict = {

        'do_train': config.mode == 'train'
        'do_eval': config.mode == 'test'

        'save_strategy': 'epoch',
        'logging_strategy': 'epoch',
        'evaluation_strategy': 'epoch',

        'disable_tqdm': True,

        'output_dir': config.output_dir,
        'logging_dir': config.logging_dir,

        'fp16': config.fp_16,
        'fp16_opt_level': config.fp16_opt_level,

        'optim': config.optim,

        'num_train_epochs': config.n_epochs,
        'learning_rate': config.learning_rate,

        'per_device_train_batch_size': config.batch_size,
        'per_device_train_batch_size': config.batch_size,

        'gradient_checkpointing': config.gradient_checkpointing,
        'gradient_accumulation_steps': config.gradient_accumulation_steps
    }

    return TrainingArguments(**training_args_dict)



def load_trainer(config, model, tokenizer):
    training_args = set_training_args(config)

    train_dataset = Dataset.from_json('data/train.json')
    valid_dataset = Dataset.from_json('data/valid.json')
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, 
        model=model, 
        padding=True
    )    

    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=train_dataset, 
        eval_dataset=valid_dataset,
        data_collator=data_collator
    )

    return trainer