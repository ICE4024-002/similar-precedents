from sqlalchemy import Column, Integer, ForeignKey
from sqlalchemy.orm import relationship
from database import Base

class QuestionerFeedback(Base):
    __tablename__ = "questioner_feedback"
    id = Column(Integer, primary_key=True, index=True)
    qna_id = Column(Integer, ForeignKey('qna.id'))
    feedback = Column(Integer, nullable=False)