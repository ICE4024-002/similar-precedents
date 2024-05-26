from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager
import asyncio
from dto.feedback import models, schemas
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

# 전문가의 평가를 DB에 저장하는 API@app.post("/feedback/", response_model=schemas.FeedBack)def create_feedback(feedback: schemas.FeedbackCreate, db: Session = Depends(get_db)):    return crud.create_feedback(db=db, feedback=feedback)