from sqlalchemy import Column, Integer, Text
from sqlalchemy.orm import relationship
from database import Base

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

class QnA(Base):
    __tablename__ = "qna"
    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)

    expert_feedback = relationship('ExpertFeedback', backref='qna')
    questioner_feedback = relationship('QuestionerFeedback', backref='qna')