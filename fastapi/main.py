from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager
import asyncio

import sys, os
import time

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

from database import engine, Base, get_db, add_and_commit, add_embedding_to_db

from init import load_precedents, load_precedents_embeddings, load_question_embeddings
from similar_precedent import find_similar_precedent
from qa_system import get_gpt_answer_by_precedent, regenerate_gpt_answer
from g_eval import calculate_g_eval_score, evaluate_scores
from embedding import create_embeddings
from similar_question import get_similar_question

from config.cors_config import add_cors_middleware

# 데이터베이스 테이블 생성
Base.metadata.create_all(bind=engine)

# TODO: init.py로 분리
@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    precedents = await loop.run_in_executor(None, load_precedents)
    precedents_embeddings = await loop.run_in_executor(None, load_precedents_embeddings)
    question_embeddings = await loop.run_in_executor(None, load_question_embeddings)

    app.state.precedents = precedents
    app.state.precedents_embeddings = precedents_embeddings
    app.state.question_embeddings = question_embeddings

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
    start_time = time.time()
    
    # 질문 임베딩
    question_vector = create_embeddings(Question.question)
    embedding_time = time.time()
    print("질문 임베딩 시간:", embedding_time - start_time)
    
    # 유사 판례 검색
    similar_precedent, similarity = find_similar_precedent(question_vector, app.state.precedents, app.state.precedents_embeddings)
    search_time = time.time()
    print("유사 판례 검색 시간:", search_time - embedding_time)
    
    similar_qna = None
    expert_feedback = None
    questioner_feedback = None

    # 유사 질문 검색
    if app.state.question_embeddings:
        similar_question_id, question_similarity = get_similar_question(question_vector, app.state.question_embeddings)
        print(">>> similar_question_id: ", similar_question_id)

        if question_similarity >= 65:
            similar_qna = db.query(dto.qna.models.QnA).filter(dto.qna.models.QnA.id == similar_question_id).first()
            print(">>> similar_question: ", similar_qna.question if similar_qna else "None")
            print(">>> similar_answer: ", similar_qna.answer if similar_qna else "None")

            # 유사 질문에 대한 피드백이 없을 경우 None 반환
            expert_feedback = db.query(dto.feedback.expert.models.ExpertFeedback).filter(dto.feedback.expert.models.ExpertFeedback.qna_id == similar_question_id).first()
            questioner_feedback = db.query(dto.feedback.questioner.models.QuestionerFeedback).filter(dto.feedback.questioner.models.QuestionerFeedback.qna_id == similar_question_id).first()
        
            print(">>> expert_feedback: ", expert_feedback.feedback if expert_feedback else "None")
            print(">>> questioner_feedback: ", questioner_feedback.feedback if questioner_feedback else "None")
        else:
            print(">>> 유사 질문 threshold 불만족")
    
    similar_answer = None
    expertFeedback = None
    
    if questioner_feedback and questioner_feedback.feedback > 3:
        similar_answer = similar_qna.answer
    
    if expert_feedback:
        expertFeedback = expert_feedback.feedback

    # GPT 답변 생성
    answer, prompt = get_gpt_answer_by_precedent(Question.question, similar_precedent, similarity, expertFeedback, similar_answer)
    gpt_time = time.time()
    print("GPT 답변 생성 시간:", gpt_time - search_time)
    
    # 답변에 대한 G-EVAL 점수 계산
    g_eval_score = calculate_g_eval_score(Question.question, answer)
    g_eval_score_not_satisfied = evaluate_scores(g_eval_score)
    print(">>> G-EVAL 점수: ", g_eval_score)
    if g_eval_score_not_satisfied:
        print(">>> G-EVAL 점수 충족 X !!!")
        answer, prompt = regenerate_gpt_answer(Question.question)
    g_eval_time = time.time()
    print("G-EVAL 점수 계산 및 미충족 시 재생성 시간:", g_eval_time - gpt_time)
    
    # DB에 QA 저장
    # 판례 유사도 65 이상, G-EVAL 점수 4점 이상
    qna = dto.qna.models.QnA(question=Question.question, answer=answer)
    if similarity >= 65 and not g_eval_score_not_satisfied:
        add_and_commit(db, qna)
    db_save_time = time.time()
    print("DB 저장 시간:", db_save_time - g_eval_time)
    
    # DB에 질문 벡터 저장
    question_vector_float = [tensor.item() for tensor in question_vector]
    if similarity >= 65:
        add_embedding_to_db(qna.id, question_vector_float, "question_vector", db)
    
    app.state.question_embeddings = load_question_embeddings()

    db_embedding_time = time.time()
    print("질문 벡터 DB 저장 시간:", db_embedding_time - db_save_time)
    
    total_time = db_embedding_time - start_time
    print("전체 실행 시간:", total_time)

    similar_qna_dto = {"question": similar_qna.question, "answer": similar_qna.answer} if similar_qna else None
    
    return { "id": qna.id, "answer": answer, "similarity": similarity, "precedent": similar_precedent, "prompt": prompt, 
            "qna": similar_qna_dto }


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

    add_and_commit(db, expert_feedback)

    return expert_feedback

# 질문자의 피드백을 저장하는 API
@app.post("/questioner-feedback")
def create_questioner_feedback(feedback: dto.feedback.questioner.schemas.QuestionerFeedbackCreate, db: Session = Depends(get_db)):
    questioner_feedback = dto.feedback.questioner.models.QuestionerFeedback(qna_id=feedback.qna_id, feedback=feedback.feedback)

    add_and_commit(db, questioner_feedback)

    return questioner_feedback