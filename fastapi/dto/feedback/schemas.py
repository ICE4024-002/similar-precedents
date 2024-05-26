from pydantic import BaseModel

class FeedBackBase(BaseModel):
    question: str
    answer: str
    feedback: str

class FeedbackCreate(FeedBackBase):
    pass

class FeedBack(FeedBackBase):
    id: int

    class Config:
        from_attributes = True