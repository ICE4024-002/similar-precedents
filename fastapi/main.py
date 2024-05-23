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

@app.post("/qaf/", response_model=schemas.QAF)
def create_qaf(qaf: schemas.QAFCreate, db: Session = Depends(get_db)):
    return crud.create_qaf(db=db, qaf=qaf)

@app.get("/test")
def test():
    return {"test": "test"}