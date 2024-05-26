from sqlalchemy.orm import Session
from dto.feedback import models, schemas

def create_feeback(db: Session, feedback: schemas.FeedbackCreate):
    db_feedback = models.feeback(**feedback.model_dump())
    db.add(db_feedback)
    db.commit()
    db.refresh(db_feedback)
    return db_feedback
