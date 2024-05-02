import torch
from tqdm import tqdm
from sqlalchemy import create_engine
from sqlalchemy import text
from transformers import AutoModel, AutoTokenizer 
from datasets import load_dataset


dataset_id ="joonhok-exo-ai/korean_law_open_data_precedents"
dataset = load_dataset(dataset_id)
data = dataset['train']
summary = data['판결요지']
full_text = data['전문']

texts = [] 
for i in tqdm(range(len(data))):
  texts.append(summary[i] if summary[i] is not None else full_text[i])

print('>>> Precedent Texts loaded!')

engine = create_engine('postgresql://song-yeonghyun:1234@localhost:5432/postgres')
connection = engine.connect()
print(">>> Connection established successfully!")

tokenizer = AutoTokenizer.from_pretrained('jhgan/ko-sbert-multitask')
model = AutoModel.from_pretrained('jhgan/ko-sbert-multitask')
print('>>> Model & Tokenizer loaded!')

def cal_score(a, b):
    if len(a.shape) == 1: a = a.unsqueeze(0)
    if len(b.shape) == 1: b = b.unsqueeze(0)

    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]

    return torch.mm(a_norm, b_norm.transpose(0, 1)) * 100

max_similarities = []
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

print('>>> Precedent tensor vectors created!')

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_similar_precedent(question):
    encoded_input = tokenizer([question], padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        model_output = model(**encoded_input)
    
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    question_vector = embeddings[0]

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

    similarities = cal_score(question_vector.unsqueeze(0), result_embeddings)
    max_similarity = similarities.max().item()
    max_similarity_idx = similarities.argmax().item()

    dataset_id ="joonhok-exo-ai/korean_law_open_data_precedents"
    dataset = load_dataset(dataset_id)
    data = dataset['train']

    print(f'>>> Max similarity: {max_similarity}')
    print(f'>>> Similar Text: {texts[max_similarity_idx]}')

    return data[max_similarity_idx], max_similarity