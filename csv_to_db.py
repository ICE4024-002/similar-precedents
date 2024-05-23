import pandas as pd
from tqdm import tqdm
from sqlalchemy import create_engine
from sqlalchemy import text

vectors = pd.read_csv('./embedded_data.csv')
print('>>> CSV read complete!')

vectors_list = []
for i in tqdm(range(0, len(vectors))):
    arr = []
    for elem in vectors.iloc[i]:
        arr.append(elem)
    vectors_list.append(arr)

print('>>> CSV to array complete!')

engine = create_engine('postgresql://song-yeonghyun:1234@localhost:5432/postgres')
connection = engine.connect()
print('>>> Connection established successfully!')


for elem in tqdm(vectors_list):
    connection.execute(text(f"INSERT INTO items (embedding) VALUES ('{elem}');"))
connection.commit()
print('>>> Vector inserted successfully!')