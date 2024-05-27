from sqlalchemy import Column, Integer, Text, ForeignKey
from sqlalchemy.orm import relationship
from database import Base

class ExpertFeedback(Base):
    __tablename__ = "expert_feedback"
    id = Column(Integer, primary_key=True, index=True)
    qna_id = Column(Integer, ForeignKey('qna.id'))
    feedback = Column(Text, nullable=False)