import torch
from transformers import AutoModelForSequenceClassification
from peft import (
    TaskType,
    get_peft_model,
    PromptTuningInit,    
    
    PromptTuningConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,  #for P-Tuning
    
    LoraConfig,
    IA3Config
)




def set_peft_config(config):
    peft = config.peft
    task_type = TaskType.SEQ_CLS
    num_virtual_tokens = config.num_virtual_tokens
    encoder_hidden_size = config.encoder_hidden_size
    

    if peft == 'prompt_tuning':
        peft_config = PromptTuningConfig(
            task_type=task_type,
            inference_mode = False,
            num_virtual_tokens=num_virtual_tokens,
            prompt_tuning_init=PromptTuningInit.RANDOM,
            tokenizer_name_or_path=config.mname
        )

    elif peft == 'prefix_tuning':
        peft_config = PrefixTuningConfig(
            task_type=task_type,
            inference_mode=False,
            num_virtual_tokens=num_virtual_tokens
        )


    elif peft == 'p_tuning':
        peft_config = PromptEncoderConfig(
            task_type=task_type,
            inference_mode = False,
            num_virtual_tokens=num_virtual_tokens,
            encoder_hidden_size=encoder_hidden_size
        )

    elif peft == 'lora':
        peft_config = LoraConfig(
            task_type=task_type,
            inference_mode = False,
            r=16,
            lora_alpha=16,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["classifier"],
        )

    elif peft == 'ia3':
        peft_config = IA3Config(
            task_type=task_type,
            inference_mode = True,
            target_modules=["query", "value", "classifier"],
            feedforward_modules=["classifier"]
        )

    return peft_config



def set_param_dict(model):
    param_dict = {}
    all_params = 0
    trainable_params = 0
    
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    param_dict['all_params'] = all_params
    param_dict['trainable_params'] = trainable_params
    param_dict['trainable_perc'] = f"{(100 * trainable_params / all_params):.5f}%"

    return param_dict



def load_model(config):
    model = AutoModelForSequenceClassification.from_pretrained(config.mname, num_labels=config.num_labels)

    if config.peft == 'vanilla':
        return model.to(config.device), set_param_dict(model)

    peft_config = set_peft_config(config)

    model = get_peft_model(model, peft_config)
    model.config.use_cache = False     #For Gradient Checkpointing

    return model.to(config.device), set_param_dict(model)
