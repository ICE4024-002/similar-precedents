from pydantic import BaseModel

class QAFBase(BaseModel):
    question: str
    answer: str
    feedback: str

class QAFCreate(QAFBase):
    pass

class QAF(QAFBase):
    id: int

    class Config:
        orm_mode = True