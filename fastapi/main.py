from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from dto.feedback import models, schemas
from api import crud
from api.database import SessionLocal, engine
from fastapi import FastAPI

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# 전문가의 평가를 DB에 저장하는 API@app.post("/feedback/", response_model=schemas.FeedBack)def create_feedback(feedback: schemas.FeedbackCreate, db: Session = Depends(get_db)):    return crud.create_feedback(db=db, feedback=feedback)