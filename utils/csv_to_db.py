import pandas as pd
from tqdm import tqdm
from sqlalchemy import create_engine
from sqlalchemy import text
import os
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '../.env.local')
load_dotenv(dotenv_path)

vectors = pd.read_csv('./csv/qa-csv/embedded_data_ko_sbert_multitask.csv')
print('>>> CSV read complete!')

vectors_list = []
for i in tqdm(range(0, len(vectors))):
    arr = []
    for elem in vectors.iloc[i]:
        arr.append(elem)
    vectors_list.append(arr)

print('>>> CSV to array complete!')

engine = create_engine('postgresql://postgres:6341@localhost:5432/postgres')
connection = engine.connect()
print('>>> Connection established successfully!')


for elem in tqdm(vectors_list):
    connection.execute(text(f"INSERT INTO ko_sbert_not_processed_precedents (embedding) VALUES ('{elem}');"))
connection.commit()
print('>>> Vector inserted successfully!')