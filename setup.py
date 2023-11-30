import os, re, json, random
from datasets import load_dataset




def fetch_agnews(orig_data):
    fetched = []
    max_len = 300
    tot_volumn = 1200
    class_volumn = 1200 // 4
    class1_cnt, class2_cnt, class3_cnt, class4_cnt = 0, 0, 0, 0
    class1_data, class2_data, class3_data, class4_data = [], [], [], []

    for elem in orig_data['train']:
        curr_volumn = class1_cnt + class2_cnt + class3_cnt + class4_cnt
        
        if curr_volumn == tot_volumn:
            break

        text = elem['text'].lower()
        if len(text) > max_len:
            continue
            
        text = re.sub(r'\\+', ' ', text)
        text = re.sub(r'\s{2,}', ' ', text)
        text = re.sub(r'&lt;b&gt;', ' ', text)
        label = elem['label']

        if label == 0 and class1_cnt < class_volumn:
            class1_cnt += 1
            class1_data.append({'x': text, 'y': label})
            continue
        
        elif label == 1 and class2_cnt < class_volumn:
            class2_cnt += 1
            class2_data.append({'x': text, 'y': label})
            continue
        
        elif label == 2 and class3_cnt < class_volumn:
            class3_cnt += 1
            class3_data.append({'x': text, 'y': label})
            continue
        
        elif label == 3 and class4_cnt < class_volumn:
            class4_cnt += 1
            class4_data.append({'x': text, 'y': label})

    fetched = [elem for elem1, elem2, elem3, elem4 \
               in zip(class1_data, class2_data, class3_data, class4_data) \
               for elem in (elem1, elem2, elem3, elem4)]

    return fetched



def split_shuffle(fetched):
    train_data = fetched[:-200]
    valid_data = fetched[-200:-100]
    test_data = fetched[-100:]

    random.shuffle(train_data)
    random.shuffle(valid_data)
    random.shuffle(test_data)

    return train_data, valid_data, test_data



def save_data(train_data, valid_data, test_data):
    for key, val in zip(['train', 'valid', 'test'], [train_data, valid_data, test_data]):
        
        f_name = f'data/{key}.json'

        with open(f_name, 'w') as f:
            json.dump(val, f)
        
        assert os.path.exists(f_name)



def main():
    orig_data = load_dataset('ag_news')
    fetched_data = fetch_agnews(orig_data)
    train_data, valid_data, test_data = split_shuffle(fetched_data)
    save_data(train_data, valid_data, test_data)



if __name__ == '__main__':
    main()