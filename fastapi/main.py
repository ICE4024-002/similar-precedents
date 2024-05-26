from fastapi import FastAPI, Depends, Query
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager
import asyncio

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from init import load_precedents, load_precedents_embeddings
from similar_precedent import find_similar_precedent
from qa_system import get_gpt_answer_by_precedent

from dto.feedback import models, schemas
from dto.precedent.schemas import Precedent
from api import crud
from api.database import SessionLocal, engine

models.Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    precedents = await loop.run_in_executor(None, load_precedents)
    precedents_embeddings = await loop.run_in_executor(None, load_precedents_embeddings)

    app.state.precedents = precedents
    app.state.precedents_embeddings = precedents_embeddings
    yield

app = FastAPI(lifespan=lifespan)

# 질문에 대한 유사 판례를 반환하는 API
@app.get("/similar-precedent/")
def get_similar_precedent(question: str = Query(...)):
    similar_precedent, similarity = find_similar_precedent(app.state.precedents, app.state.precedents_embeddings, question)

    precedent = Precedent(**similar_precedent)

    return {"similarity": similarity, "precedent": precedent}

# 질문이 Input으로 들어오면 유사 판례 정보와 GPT 답변을 Output으로 반환하는 API
@app.get("/answer/")
def get_gpt_answer(question: str = Query(...)):
    similar_precedent, similarity = find_similar_precedent(question, app.state.precedents, app.state.precedents_embeddings)
    answer = get_gpt_answer_by_precedent(question, similar_precedent, similarity)
    return { "answer": answer, "similarity": similarity, "precedent": similar_precedent }

# 전문가의 평가를 DB에 저장하는 API
@app.post("/feedback/", response_model=schemas.FeedBack)
def create_feedback(feedback: schemas.FeedbackCreate, db: Session = Depends(get_db)):
    return crud.create_feedback(db=db, feedback=feedback)