from pydantic import BaseModel

class QnABase(BaseModel):
    question: str
    answer: str

class QnACreate(QnABase):
    pass

class QnA(QnABase):
    id: int

    class Config:
        from_attributes = True
