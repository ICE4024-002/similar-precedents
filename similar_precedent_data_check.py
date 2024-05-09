import torch
import pandas as pd
import heapq
from tqdm import tqdm
from sqlalchemy import create_engine
from sqlalchemy import text
from datasets import load_dataset
import csv

dataset_id ="joonhok-exo-ai/korean_law_open_data_precedents"
dataset = load_dataset(dataset_id)
data = dataset['train']
summary = data['판결요지']
full_text = data['전문']

texts = []
for i in tqdm(range(len(data))):
  texts.append(summary[i] if summary[i] is not None else full_text[i])

print('>>> Precedent Texts loaded!')

engine = create_engine('postgresql://leeeeeyeon:1234@localhost:5432/postgres')
connection = engine.connect()
print(">>> Connection established successfully!")

def cal_score(a, b):
    if len(a.shape) == 1: a = a.unsqueeze(0)
    if len(b.shape) == 1: b = b.unsqueeze(0)

    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]

    return torch.mm(a_norm, b_norm.transpose(0, 1)) * 100

qa_questions = pd.read_csv('./qa-csv/total_qa_spell_checked.csv')

qa_vectors = pd.read_csv('./qa-csv/embedded_question_data_spell_ko_sbert_multitask.csv')

qa_vectors_list = []
for i in tqdm(range(0, len(qa_vectors))):
    qa_vector = []
    for elem in qa_vectors.iloc[i]:
        qa_vector.append(elem)
    qa_vectors_list.append(torch.tensor(qa_vector))

print('>>> QA vectors created!')

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

bottom_similarity_heap = []
top_similarity_heap = []
similarity_heap = []
heap_size = 10

for i in tqdm(range(0, len(qa_vectors_list))):
    qa_vector = torch.tensor(qa_vectors_list[i])

    similarities = cal_score(qa_vector.unsqueeze(0), result_embeddings)

    max_similarity = similarities.max().item()

    if max_similarity >= 65:  # 유사도가 65 이상인 결과에 대해서만 처리
        max_similarity_idx = similarities.argmax().item()

        if len(bottom_similarity_heap) < heap_size:
            heapq.heappush(bottom_similarity_heap, (-max_similarity, qa_questions.iloc[i]["question"], texts[max_similarity_idx]))
        else:
            heapq.heappushpop(bottom_similarity_heap, (-max_similarity, qa_questions.iloc[i]["question"], texts[max_similarity_idx]))

        if len(top_similarity_heap) < heap_size:
            heapq.heappush(top_similarity_heap, (-max_similarity, qa_questions.iloc[i]["question"], texts[i]))
        else:
            heapq.heappushpop(top_similarity_heap, (-max_similarity, qa_questions.iloc[i]["question"], texts[i]))

        if len(similarity_heap) < heap_size:
            heapq.heappush(similarity_heap, (-max_similarity, qa_questions.iloc[i]["question"], texts[i]))
        else:
            heapq.heappushpop(similarity_heap, (-max_similarity, qa_questions.iloc[i]["question"], texts[i]))

bottom_similarities = sorted(bottom_similarity_heap, key=lambda x: -x[0])  # 하위 5개에 대한 결과 정렬
top_similarities = sorted(top_similarity_heap, key=lambda x: -x[0])

# CSV 파일 경로 설정
bottom_csv_file_path = "./bottom_similarities.csv"
top_csv_file_path = "./top_similarities.csv"
csv_file_path = "./similarities.csv"

# CSV 파일에 쓰기
with open(bottom_csv_file_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # 헤더 쓰기
    writer.writerow(['Similarity', 'Question', 'Text'])
    # 결과 쓰기
    for similarity, question, text in bottom_similarities:
        writer.writerow([-similarity, question, text])
print(f">>> 65 이상 유사도를 가진 하위 10개 결과 저장 완료")

with open(top_csv_file_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # 헤더 쓰기
    writer.writerow(['Similarity', 'Question', 'Text'])
    # 결과 쓰기
    for similarity, question, text in top_similarities:
        writer.writerow([-similarity, question, text])
print(f">>> 65 이상 유사도를 가진 상위 10개 결과 저장 완료")

with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # 헤더 쓰기
    writer.writerow(['Similarity', 'Question', 'Text'])
    # 결과 쓰기
    for similarity, question, text in similarity_heap:
        writer.writerow([-similarity, question, text])
print(f">>> 65 이상 유사도를 가진 결과 저장 완료")