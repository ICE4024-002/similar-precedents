import pandas as pd
import re
from tqdm import tqdm
from hanspell import spell_checker

total_qa = pd.read_csv('./total_qa.csv')

def clean_text(text):
    text = re.sub(r'[\n\r\x0b\t]', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

def spell_check_text(text):
    max_length = 300
    if len(text) > max_length:
        parts = [text[i:i+max_length] for i in range(0, len(text), max_length)]
        return ''.join([spell_checker.check(part).as_dict()['checked'] for part in parts])
    else:
        return spell_checker.check(text).as_dict()['checked']

total_qa['question'] = total_qa['question'].apply(clean_text)
total_qa['answer'] = total_qa['answer'].apply(clean_text)

# progress_apply를 사용하여 tqdm 적용
tqdm.pandas()
total_qa['question'] = total_qa['question'].progress_apply(spell_check_text)
total_qa['answer'] = total_qa['answer'].progress_apply(spell_check_text)

total_qa.to_csv('./total_qa_spell_checked.csv', index=False)

print('>>> Question 및 Answer 교정 완료!')