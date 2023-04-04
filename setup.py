import os, re, json, argparse
from datasets import load_dataset
from transformers import T5TokenizerFast, AutoTokenizer



def load_data(task):
    if task == 'nmt':
        data = load_dataset('wmt14', 'de-en', split='train')['translation']
        
    elif task == 'dialog':
        data = load_dataset('daily_dialog', split='train')['dialog']

    elif task == 'sum':
        data = load_dataset('cnn_dailymail', '3.0.0', split='train')

    return data



#NMT
def process_nmt(orig_data, tokenizer, volumn=32000):
    processed, volumn_cnt = [], 0
    min_len, max_len, max_diff = 10, 300, 50 
    
    for elem in orig_data:
        src, trg = elem['en'].lower(), elem['de'].lower()
        src_len, trg_len = len(src), len(trg)

        #define filtering conditions
        min_condition = (src_len >= min_len) & (trg_len >= min_len)
        max_condition = (src_len <= max_len) & (trg_len <= max_len)
        dif_condition = abs(src_len - trg_len) < max_diff

        if max_condition & min_condition & dif_condition:
            temp_dict = dict()
            
            src_tokenized = tokenizer(src)
            trg_tokenized = tokenizer(trg)

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
def process_dialog(orig_data, tokenizer, volumn=32000):
    processed, volumn_cnt = [], 0
    src_list, trg_list = [], []
    
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
        src_tokenized = tokenizer(src)
        trg_tokenized = tokenizer(trg)

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
def process_sum(orig_data, tokenizer, volumn=32000):    
    min_len, max_len=500, 2000
    processed, volumn_cnt = [], 0

    for elem in orig_data:
        src, trg = elem['article'].lower(), elem['highlights'].lower()

        #Filter too Short or too Long Context
        if not (min_len < len(src) < max_len):
            continue
        if len(trg) > min_len:
            continue

        #remove unnecessary characters in trg sequence
        trg = re.sub(r'\n', ' ', trg.strip())         #remove \n
        trg = re.sub(r"\s([.](?:\s|$))", r'\1', trg)  #remove whitespace in front of dot
        
        temp_dict = dict()
        src_tokenized = tokenizer(src)
        trg_tokenized = tokenizer(trg)

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
    train, valid, test = data_obj[:-2000], data_obj[-2000:-1000], data_obj[-1000:]
    data_dict = {k:v for k, v in zip(['train', 'valid', 'test'], [train, valid, test])}

    for key, val in data_dict.items():
        with open(f'data/{task}/{key}.json', 'w') as f:
            json.dump(val, f)        
        assert os.path.exists(f'data/{task}/{key}.json')
    


def main(task):
    #Prerequisite
    os.makedirs(f'data/{task}', exist_ok=True)

    #Load Original Data
    orig = load_data(task)

    if task == 'sum':
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-LongT5ForConditionalGeneration")
    else:
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