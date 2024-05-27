from pydantic import BaseModel

class ExpertFeedbackBase(BaseModel):
    qna_id: int
    feedback: str

class ExpertFeedbackCreate(ExpertFeedbackBase):
    class Config:
        exclude = ["id"]

class ExpertFeedback(ExpertFeedbackBase):
    id: int

    class Config:
        from_attributes = True
