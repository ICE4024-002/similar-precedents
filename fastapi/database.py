from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import os
from dotenv import load_dotenv
from typing import Any
from sqlalchemy import text

dotenv_path = os.path.join(os.path.dirname(__file__), '../.env.local')
load_dotenv(dotenv_path)
db_url = os.getenv('DB_URL')

engine = create_engine(db_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def add_and_commit(db: Session, instance: Any) -> Any:
    db.add(instance)
    db.commit()
    db.refresh(instance)

    return instance

def add_embedding_to_db(question_id, embedding, table_name, db: Session):
    db.execute(text(f"INSERT INTO {table_name} (question_id, embedding) VALUES ({question_id}, '{embedding}')"))
    db.commit()
