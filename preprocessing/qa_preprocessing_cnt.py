import pandas as pd
import re

from tqdm import tqdm
from hanspell import spell_checker

total_qa = pd.read_csv('./total_qa.csv')

questions = total_qa['question']
answers = total_qa['answer']

for i in tqdm( range(len(questions))):
    question = questions[i]
    question = question.replace("\n", "").replace("\t", " ")
    pattern_punctuation = re.compile(r'[^\w\s]')
    questions[i] = pattern_punctuation.sub('', question)

for i in tqdm(range(len(answers))):
    answer = answers[i]
    answer = answer.replace("\n", "").replace("\t", " ")
    pattern_punctuation = re.compile(r'[^\w\s]')
    answers[i] = pattern_punctuation.sub('', answer)

print('>>> questions, answers 개행문자 및 특수문자 제거 완료')

# 맞춤법과 띄어쓰기가 잘못된 QA 데이터 개수 확인
max_length = 300

question_errors = 0
for question in tqdm(questions):
    # question이 300자를 초과하는지 확인
    if len(question) > max_length:
        
        # question을 max_length에 맞춰 여러 조각으로 나누기
        question_parts = [question[i:i+max_length] for i in range(0, len(question), max_length)]
        
        # 각 조각을 spell checker로 체크
        for part in question_parts:
            if spell_checker.check(part).as_dict()['errors'] > 0:
                question_errors += 1
                break
                
    else:
        # question이 300자 이하일 경우 바로 체크
        if spell_checker.check(question).as_dict()['errors'] > 0:
            question_errors += 1

print('>>> 질문 맞춤법/띄어쓰기 오류 개수 검출 완료')

answer_errors = 0
for answer in tqdm(answers):
    if len(answer) > max_length:
        answer_parts = [answer[i:i+max_length] for i in range(0, len(answer), max_length)]
        for part in answer_parts:
            if spell_checker.check(part).as_dict()['errors'] > 0:
                answer_errors += 1
                break
    else:
        if spell_checker.check(answer).as_dict()['errors'] > 0:
            answer_errors += 1

print('>>> 답변 맞춤법/띄어쓰기 오류 개수 검출 완료')


print(f'>>> 맞춤법/띄어쓰기가 올바르지 않은 질문: {question_errors}')
print(f'>>> 맞춤법/띄어쓰기가 올바르지 않은 답변: {answer_errors}')