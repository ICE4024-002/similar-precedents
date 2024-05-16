import similar_precedent
import os
import pandas as pd
import csv
import torch
import concurrent.futures
from openai import OpenAI
from sqlalchemy import create_engine
from sqlalchemy import text
from tqdm import tqdm
from datasets import load_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

client = OpenAI(
    # This is the default and can be omitted
    api_key="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",
)

dataset_id ="joonhok-exo-ai/korean_law_open_data_precedents"
dataset = load_dataset(dataset_id)
data = dataset['train']

engine = create_engine('postgresql://leeeeeyeon:1234@localhost:5432/postgres')
connection = engine.connect()
print(">>> Connection established successfully!")

query = text(f"""
SELECT
    embedding
FROM
    ko_sbert_not_processed_precedents;
""")
result = connection.execute(query).fetchall()

result_embeddings = torch.tensor([
    [float(num_str) for num_str in row[0][1:-1].split(',')]
    for row in result
])
print(">>> Precedent Embeddings Loaded successfully!")

similarity_threshold = 65

total_datas = pd.read_csv('./result-csv/total_similarities.csv')

total_answers = []

question_vectors = pd.read_csv('./qa-csv/embedded_question_data_spell_ko_sbert_multitask.csv')
question_vectors_list = []
for i in tqdm(range(0, len(question_vectors))):
    qa_vector = []
    for elem in question_vectors.iloc[i]:
        qa_vector.append(elem)
    question_vectors_list.append(torch.tensor(qa_vector))
print('>>> Qusetion vectors created!')

def process_data(i):
    question = total_datas.iloc[i]["Question"]
    question_vector = question_vectors_list[i]
    similarData, similarity = similar_precedent.get_similar_precedent_total(data, result_embeddings, question_vector)

    prompt = {}

    if similarity < similarity_threshold:
        prompt = {
            "role" : "system",
            "content" : "You are a Korean law expert tasked with providing clear and precise answers to various legal questions."
                        "Create a universal solution for user query and provide guidance in the last line of your answers to contact legal experts for details."
                        "Ensure your explanation is both comprehensive and accessible to non-expert users."
                        "You must answer in Korean."
        }
    else:
        prompt = {
            "role" : "system",
            "content" : "You are a Korean law expert tasked with summarizing similar precedents and providing conclusions based on them."
                        "First, briefly summarize the relevant details from similar case law, focusing on judgementSummary and fullText."
                        "Conclude your response by explaining how these cases apply to the user's query, emphasizing the outcome of the precedents."
                        "Your responses should always end with a reference to specific articles or sections of the law that directly apply to the user's query, ensuring your advice is grounded in relevant legal principles."
                        "For each legal query, carefully analyze any given context or case law to extract pertinent legal precedents and principles."
                        "In responding to the user's query, consider both the general principles of law and any relevant case law or statutes that specifically address the issue at hand."
                        "Structure your response to start with a summary of the case, followed by a conclusion that outlines the legal basis for the advice, as follows: 'Based on the precedents, your situation can be concluded as follows... In accordance with Article [number] of [Law Name]'."
                        "Ensure your explanation is both comprehensive and accessible to non-expert users."
                        "You must answer in Korean."
        }

    similar_data = {
                    "role" : "system",
                    "content" : f"""
                    Here is a info of relevant case law:
                        'caseName': {similarData['사건명']},
                        'sentence': {similarData['선고']},
                        'caseNumber': {similarData['사건번호']},
                        'judgementType': {similarData['판결유형']},
                        'decision': {similarData['판시사항']},
                        'judgementSummary': {similarData['판결요지']},
                        'referenceArticles': {similarData['참조조문']},
                        'referencePrecedents': {similarData['참조판례']},
                        'fullText': {similarData['전문']}
                        """
    }

    userInput = {
                    "role": "user",
                    "content": f"{question}",
    }

    try:
        messages_list = []
        if similarity < similarity_threshold:
            messages_list = [prompt, userInput]
        else:
            messages_list = [prompt, similar_data, userInput]

        # print(">>> GPT generating...")
        chat_completion = client.chat.completions.create(
            messages=messages_list,
            model="gpt-3.5-turbo",
            temperature=0
        )

        return [i, similarity, question, chat_completion.choices[0].message.content]
    except:
        global cnt
        cnt += 1

total_answers = []
cnt = 0 # 답변 생성 불가 횟수

# 병렬 처리를 위해 ThreadPoolExecutor를 사용
with concurrent.futures.ThreadPoolExecutor() as executor:
    results = list(tqdm(executor.map(process_data, range(0, len(total_datas))), total=len(total_datas)))

# 결과를 total_answers 리스트에 추가
total_answers.extend(results)
qa_datas = pd.read_csv('./qa-csv/total_qa_spell_checked.csv')

total_csv_file_path = "./result-csv/total_answers.csv"
with open(total_csv_file_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # 헤더 쓰기
    writer.writerow(['Similarity', 'Question', 'GPT Answer', 'Original Answer'])
    # 결과 쓰기
    for idx, similarity, question, gpt_answer in total_answers:
        writer.writerow([similarity, question, gpt_answer, qa_datas.iloc[total_datas.iloc[idx]["Question Index"]]['answer']])
print(f">>> GPT-3.5 Turbo generated answers saved in {total_csv_file_path}!")