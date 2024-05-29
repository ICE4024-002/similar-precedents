import os
import torch
from datasets import load_dataset
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '../.env.local')
load_dotenv(dotenv_path)

db_url = os.getenv('DB_URL')
engine = create_engine(db_url)
connection = engine.connect()

def load_precedents():
    dataset_id ="joonhok-exo-ai/korean_law_open_data_precedents"
    dataset = load_dataset(dataset_id)
    data = dataset['train']

    print('>>> Precedents Text loaded!')
    return data

def load_precedents_embeddings():
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

    print('>>> Precedents Embeddings loaded!')
    return result_embeddings

def load_question_embeddings():
    query = text(f"""
    SELECT
        question_id, embedding
    FROM
        question_vector;
    """)
    result = connection.execute(query).fetchall()

    result_embeddings = [
        (row[0], [float(num_str) for num_str in row[1][1:-1].split(',')])
        for row in result
    ]

    print('>>> Question Embeddings loaded!')
    return result_embeddings