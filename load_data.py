# load_data.py  just for loading the data 
def load_dataset(filepath):
    input_texts, target_texts = [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if '|' in line:
                input_text, target_text = line.strip().split('|', 1)
            elif '\t' in line:
                input_text, target_text = line.strip().split('\t', 1)
            else:
                continue
            input_texts.append(input_text.strip())
            target_texts.append(target_text.strip())
    return input_texts, target_texts

