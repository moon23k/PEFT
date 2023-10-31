import os, torch
from transformers import (
    T5ForConditionalGeneration
)



def print_model_desc(model):
    #Number of trainerable parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"--- Model Params: {n_params:,}")

    #Model size check
    param_size, buffer_size = 0, 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f"--- Model  Size : {size_all_mb:.3f} MB\n")



def load_model(config):

    mname = config.mname

    model = T5ForConditionalGeneration.from_pretrained(mname) 

    print(f"Pretrained {config.mname} model has loaded")
    print_model_desc(model)
        


    return model.to(config.device)