from sqlalchemy import Column, Integer, String
from api.database import Base

class FeedBack(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    question = Column(String, index=True)
    answer = Column(String, index=True)
    feedback = Column(String, index=True)