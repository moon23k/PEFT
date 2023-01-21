import os, re, json 
import yaml, nltk, argparse
from datasets import load_dataset
from transformers import T5TokenizerFast



def load_data(task):
    if task == 'nmt':
        data = load_dataset('wmt14', 'de-en', split='train')['translation']
        
    elif task == 'dialog':
        data = load_dataset('daily_dialog', split='train')['dialog']

    elif task == 'sum':
        data = load_dataset('cnn_dailymail', '3.0.0', split='train')

    return data



#NMT
def process_nmt(orig_data, tokenizer, volumn=36000):
    min_len = 10 
    max_len = 300
    max_diff = 50
    prefix = 'translate English to German: '

    volumn_cnt = 0
    processed = []
    
    for elem in orig_data:
        src, trg = elem['en'].lower(), elem['de'].lower()
        src_len, trg_len = len(src), len(trg)

        #define filtering conditions
        min_condition = (src_len >= min_len) & (trg_len >= min_len)
        max_condition = (src_len <= max_len) & (trg_len <= max_len)
        dif_condition = abs(src_len - trg_len) < max_diff

        if max_condition & min_condition & dif_condition:
            temp_dict = dict()
            
            src_tokenized = tokenizer(prefix + src, max_length=512, truncation=True)
            trg_tokenized = tokenizer(trg, max_length=512, truncation=True)

            temp_dict['input_ids'] = src_tokenized['input_ids']
            temp_dict['attention_mask'] = src_tokenized['attention_mask']
            temp_dict['labels'] = trg_tokenized['input_ids']
            
            processed.append(temp_dict)
            
            #End condition
            volumn_cnt += 1
            if volumn_cnt == volumn:
                break

    return processed



#Dialog
def process_dialog(orig_data, tokenizer, volumn=36000):
    volumn_cnt = 0
    src_list, trg_list = [], []
    processed = []

    for dial in orig_data:
        dial_list = []
        dial_turns = len(dial)
        
        for uttr in dial:
            _uttr = re.sub(r"\s([?,.!’](?:\s|$))", r'\1', uttr)
            _uttr = re.sub(r'([’])\s+', r'\1', _uttr)
            dial_list.append(_uttr.strip().lower())
        
        if dial_turns < 2:
            continue

        elif dial_turns == 2:
            src_list.append(dial_list[0])
            trg_list.append(dial_list[1])
            continue  #To avoid duplicate on below condition

        #Incase of dial_turns is even
        elif dial_turns % 2 == 0:
            src_list.extend(dial_list[0::2])
            trg_list.extend(dial_list[1::2])

            src_list.extend(dial_list[1:-1:2])
            trg_list.extend(dial_list[2::2])
        
        #Incase of dial_turns is odds
        elif dial_turns % 2 == 1:
            src_list.extend(dial_list[0:-1:2])
            trg_list.extend(dial_list[1::2])
            
            src_list.extend(dial_list[1::2])
            trg_list.extend(dial_list[2::2])   

    assert len(src_list) == len(trg_list)
    
    for src, trg in zip(src_list, trg_list):
        temp_dict = dict()
        src_tokenized = tokenizer(src, max_length=512, truncation=True)
        trg_tokenized = tokenizer(trg, max_length=512, truncation=True)

        temp_dict['input_ids'] = src_tokenized['input_ids']
        temp_dict['attention_mask'] = src_tokenized['attention_mask']
        temp_dict['labels'] = trg_tokenized['input_ids']
        
        processed.append(temp_dict)

        #End Condition
        volumn_cnt += 1
        if volumn_cnt == volumn:
            break
    
    return processed



#Sum
def process_sum(orig_data, tokenizer, volumn=36000):    
    max_num=30  
    min_len=500 
    max_len=2000
    prefix = 'summarize: '

    volumn_cnt = 0
    processed = []

    for elem in orig_data:
        prefix = 'summarize: '
        src, trg = elem['article'].lower(), elem['highlights'].lower()

        #Filter too Short or too Long Context
        if not (min_len < len(src) < max_len):
            continue
        if len(trg) > min_len:
            continue
        
        #Filter too long Sentences 
        src_split = nltk.tokenize.sent_tokenize(src)
        if len(src_split) > max_num:
            continue
        for seq in src_split:
            if len(seq) > min_len:
                continue

        #Add Prefix and make it into long string obj 
        src = prefix + ' '.join(src_split)

        #remove unnecessary characters in trg sequence
        trg = re.sub(r'\n', ' ', trg.strip())         #remove \n
        trg = re.sub(r"\s([.](?:\s|$))", r'\1', trg)  #remove whitespace in front of dot
        
        temp_dict = dict()
        src_tokenized = tokenizer(prefix + src, max_length=512, truncation=True)
        trg_tokenized = tokenizer(trg, max_length=512, truncation=True)

        temp_dict['input_ids'] = src_tokenized['input_ids']
        temp_dict['attention_mask'] = src_tokenized['attention_mask']
        temp_dict['labels'] = trg_tokenized['input_ids']

        processed.append(temp_dict)

        volumn_cnt += 1
        if volumn_cnt == volumn:
            break
    
    return processed




def save_data(task, data_obj):
    #split data into train/valid/test sets
    train, valid, test = data_obj[:-6000], data_obj[-6000:-3000], data_obj[-3000:]
    data_dict = {k:v for k, v in zip(['train', 'valid', 'test'], [train, valid, test])}

    for key, val in data_dict.items():
        with open(f'data/{task}/{key}.json', 'w') as f:
            json.dump(val, f)        
        assert os.path.exists(f'data/{task}/{key}.json')
    


def main(task):
    #Prerequisite
    os.makedirs(f'data/{task}', exist_ok=True)
    if task == 'sum':
        nltk.download('punkt')

    #Load Original Data
    orig = load_data(task)
    tokenizer = T5TokenizerFast.from_pretrained('t5-small', model_max_length=512)

    #PreProcess Data
    if task == 'nmt':
        processed = process_nmt(orig, tokenizer)
    elif task == 'dialog':
        processed = process_dialog(orig, tokenizer)
    elif task == 'sum':
        processed = process_sum(orig, tokenizer)        

    #Save Data
    save_data(task, processed)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', required=True)
    
    args = parser.parse_args()
    assert args.task in ['all', 'nmt', 'dialog', 'sum']
    
    if args.task == 'all':
        for task in ['nmt', 'dialog', 'sum']:
            main(task)
    else: 
        main(args.task)