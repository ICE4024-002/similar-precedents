import torch
import pandas as pd
from tqdm import tqdm
from sqlalchemy import create_engine
from sqlalchemy import text
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset

engine = create_engine('postgresql://leeeeeyeon:1234@localhost:5432/postgres')
connection = engine.connect()
print(">>> Connection established successfully!")

def cal_score(a, b):
    if len(a.shape) == 1: a = a.unsqueeze(0)
    if len(b.shape) == 1: b = b.unsqueeze(0)

    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]

    return torch.mm(a_norm, b_norm.transpose(0, 1)) * 100

model = AutoModel.from_pretrained('BM-K/KoSimCSE-roberta')
tokenizer = AutoTokenizer.from_pretrained('BM-K/KoSimCSE-roberta')
print('>>> Model & Tokenizer loaded!')

qa_vectors = pd.read_csv('./embedded_question_data_spell.csv')

qa_vectors_list = []
for i in tqdm(range(0, len(qa_vectors))):
    qa_vector = []
    for elem in qa_vectors.iloc[i]:
        qa_vector.append(elem)
    qa_vectors_list.append(torch.tensor(qa_vector))

print('>>> QA vectors created!')

vectors = pd.read_csv('./embedded_data.csv')

vectors_list = []
for i in tqdm(range(0, len(vectors))):
    arr = []
    for elem in vectors.iloc[i]:
        arr.append(elem)
    vectors_list.append(torch.tensor(arr))

print('>>> Precedent vectors created!')

max_similarities = []
query = text(f"""
SELECT
    embedding
FROM
    items;
""")
result = connection.execute(query).fetchall()

result_embeddings = torch.tensor([
    [float(num_str) for num_str in row[0][1:-1].split(',')]
    for row in result
])

print('>>> Precedent tensor vectors created!')

for i in tqdm(range(0, len(qa_vectors_list))):
    qa_vector = torch.tensor(qa_vectors_list[i])

    similarities = cal_score(qa_vector.unsqueeze(0), result_embeddings)

    max_similarity = similarities.max().item()

    max_similarities.append(max_similarity)

max_similarities = torch.tensor(max_similarities)
# 평균, 최댓값, 최솟값 계산
mean_value = torch.mean(max_similarities).item()
max_value = torch.max(max_similarities).item()
min_value = torch.min(max_similarities).item()

# 결과 출력
print("평균: ", mean_value)
print("최댓값: ", max_value)
print("최솟값: ", min_value)

def get_similar_precedent(question):
    inputs = tokenizer([question], padding=True, truncation=True, return_tensors="pt")
    embeddings, _ = model(**inputs, return_dict=False)

    embedding = embeddings[0][0].tolist()
    str(embedding).replace("'", "''")

    query = text(f"""
        SELECT
            id, embedding
        FROM
            items;
        """)
    result = connection.execute(query).fetchall()

    max_similarity = -100
    precedent_idx = -1
    for i in range(0, len(result)):
        row = result[i]
        str_data = row[1]
        num_list = [float(num_str) for num_str in str_data[1:-1].split(',')]

        tensor_data = torch.tensor(num_list)
        if cal_score(torch.tensor(embedding), tensor_data).item() > max_similarity:
            max_similarity = cal_score(torch.tensor(embedding), tensor_data).item()
            precedent_idx = i
    print('>>> Similar precedent index: ', precedent_idx, ' / Similarity: ', max_similarity)

    dataset_id ="joonhok-exo-ai/korean_law_open_data_precedents"
    dataset = load_dataset(dataset_id)
    data = dataset['train']

    return data[precedent_idx]