from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager
import asyncio

import sys, os

import dto.feedback
import dto.feedback.expert
import dto.feedback.expert.models
import dto.feedback.expert.schemas
import dto.feedback.questioner
import dto.feedback.questioner.models
import dto.feedback.questioner.schemas
import dto.precedent
import dto.precedent.schemas
import dto.qna
import dto.qna.models
import dto.qna.schemas
import dto.question
import dto.question.schemas

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from database import engine, Base, get_db

from init import load_precedents, load_precedents_embeddings
from similar_precedent import find_similar_precedent
from qa_system import get_gpt_answer_by_precedent
from g_eval import calculate_g_eval_score

from config.cors_config import add_cors_middleware

# 데이터베이스 테이블 생성
Base.metadata.create_all(bind=engine)

# TODO: init.py로 분리
@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    precedents = await loop.run_in_executor(None, load_precedents)
    precedents_embeddings = await loop.run_in_executor(None, load_precedents_embeddings)

    app.state.precedents = precedents
    app.state.precedents_embeddings = precedents_embeddings

    print(">> Server started!")
    yield

app = FastAPI(lifespan=lifespan)

add_cors_middleware(app)

# 질문에 대한 유사 판례를 반환하는 API
@app.post("/similar-precedent")
def get_similar_precedent(Question: dto.question.schemas.Question):
    print(Question.question)
    similar_precedent, similarity = find_similar_precedent(Question.question, app.state.precedents, app.state.precedents_embeddings)

    precedent = dto.precedent.schemas.Precedent(**similar_precedent)

    return {"similarity": similarity, "precedent": precedent}

# 질문에 대한 유사 판례 정보와 GPT 답변을 반환하는 API
@app.post("/answer")
def get_gpt_answer(Question: dto.question.schemas.Question, db: Session = Depends(get_db)):
    # 유사 판례 검색
    similar_precedent, similarity = find_similar_precedent(Question.question, app.state.precedents, app.state.precedents_embeddings)

    # GPT 답변 생성
    answer = get_gpt_answer_by_precedent(Question.question, similar_precedent, similarity)

    # DB에 QA 저장
    qna = dto.qna.models.QnA(question=Question.question, answer=answer)
    db.add(qna)
    db.commit()

    return { "answer": answer, "similarity": similarity, "precedent": similar_precedent }

# 질문과 답변에 대해 G-EVAL 점수를 반환하는 API
@app.post("/g-eval")
def get_g_eval_score(qna: dto.qna.schemas.QnABase):
    g_eval_score = calculate_g_eval_score(qna.question, qna.answer)

    return { "g-eval": g_eval_score }

# 전문가의 피드백이 없는 QnA 목록을 반환하는 API
@app.get("/waiting-questions")
def get_waiting_questions(db: Session = Depends(get_db)):
    QnA = dto.qna.models.QnA
    questions = db.query(QnA).filter(~QnA.expert_feedback.any()).all()

    result = [{"id": qna.id, "question": qna.question} for qna in questions]

    return result

# 특정 QnA에 대한 질문과 답변을 반환하는 API
@app.get("/waiting-questions/{id}")
def get_waiting_question(id: int, db: Session = Depends(get_db)):
    QnA = dto.qna.models.QnA
    qna = db.query(QnA).filter(QnA.id == id).first()

    if qna is None:
        return HTTPException(status_code=404, detail="QnA not found")
    
    return { "question": qna.question, "answer": qna.answer }  

# 전문가의 피드백을 저장하는 API
@app.post("/expert-feedback")
def create_expert_feedback(feedback: dto.feedback.expert.schemas.ExpertFeedbackCreate, db: Session = Depends(get_db)):
    expert_feedback = dto.feedback.expert.models.ExpertFeedback(qna_id=feedback.qna_id, feedback=feedback.feedback)

    db.add(expert_feedback)
    db.commit()

    return expert_feedback

# 질문자의 피드백을 저장하는 API
@app.post("/questioner-feedback")
def create_questioner_feedback(feedback: dto.feedback.questioner.schemas.QuestionerFeedbackCreate, db: Session = Depends(get_db)):
    questioner_feedback = dto.feedback.questioner.models.QuestionerFeedback(qna_id=feedback.qna_id, feedback=feedback.feedback)

    db.add(questioner_feedback)
    db.commit()

    return questioner_feedback