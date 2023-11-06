import os, torch
from transformers import T5ForConditionalGeneration
from peft import (
    TaskType,
    get_peft_config, 
    get_peft_model, 
    get_peft_model_state_dict, 

    LoraConfig, 
    PrefixTuningConfig, 
    AutoPeftModel
)




def get_peft_config(config):

    if config.peft == 'lora':
        peft_config = LoraConfig()
    
    elif config.peft == 'prefix_tuning':
        peft_config = PrefixTuningConfig()
    
    elif config.peft == 'p_tuning':
        peft_config = pass
    
    elif config.peft == 'prompt_tuning':
        peft_config = pass    
    
    elif config.peft == 'ia3':
        peft_config = pass

    return peft_config




def load_model(config):
    mname = config.mname

    if config.task == 'train':

        model = T5ForConditionalGeneration.from_pretrained(mname) 
        print(f"Pretrained {config.mname} model has loaded")

        peft_config = get_peft_model(config)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        return model.to(config.device)

    else:
        model = AutoPeftModel(config.ckpt)
        return model.to(config.device)