from pydantic import BaseModel

class QuestionerFeedbackBase(BaseModel):
    qna_id: int
    feedback: int

class QuestionerFeedbackCreate(QuestionerFeedbackBase):
    pass

class QuestionerFeedback(QuestionerFeedbackBase):
    id: int

    class Config:
        from_attributes = True