import os, torch
from transformers import T5ForConditionalGeneration
from peft import (
    TaskType,
    PeftModel,
    get_peft_config, 
    get_peft_model, 
    LoraConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    PromptEncoderConfig
)




def set_peft_config(config):
    peft_args = {'task_type':TaskType.SEQ_2_SEQ_LM,
                 'inference_mode': False}
    
    if config.peft == 'lora':
        peft_config = LoraConfig(**peft_args)
    elif config.peft == 'prefix':
        peft_args['num_virtual_tokens'] = config.num_virtual_tokens
        peft_config = PrefixTuningConfig(**peft_args)        
    elif config.peft == 'prompt':
        peft_args['num_virtual_tokens'] = config.num_virtual_tokens
        peft_config = PromptTuningConfig(**peft_args)
    elif config.peft == 'p_tuning':
        peft_args['num_virtual_tokens'] = config.num_virtual_tokens
        peft_args['encoder_hidden_size'] = config.encoder_hidden_size
        peft_config = PromptEncoderConfig(**peft_args)

    return peft_args



def load_model(config):

    model = T5ForConditionalGeneration.from_pretrained(config.mname)

    if config.mode == 'train':
        peft_config = set_peft_config(config)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        model = PeftModel(model, config.ckpt)

    return model.to(config.device)        