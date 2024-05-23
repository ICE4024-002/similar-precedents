from sqlalchemy.orm import Session
from dto.feedback import models, schemas

def create_qaf(db: Session, qaf: schemas.QAFCreate):
    db_qaf = models.QAF(**qaf.model_dump())
    db.add(db_qaf)
    db.commit()
    db.refresh(db_qaf)
    return db_qaf
