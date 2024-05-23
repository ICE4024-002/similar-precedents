from sqlalchemy import Column, Integer, String
from api.database import Base

class QAF(Base):
    __tablename__ = "qaf"

    id = Column(Integer, primary_key=True, index=True)
    question = Column(String, index=True)
    answer = Column(String, index=True)
    feedback = Column(String, index=True)