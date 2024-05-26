from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

leeeeeyeon = 'leeeeeyeon'
yeonghyun = 'song-yeonghyun'
engine = create_engine(f'postgresql://{yeonghyun}:1234@localhost:5432/postgres')
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()