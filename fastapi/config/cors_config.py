from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI

origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:3000"
    # 필요한 경우 다른 도메인도 추가 가능
]

def add_cors_middleware(app: FastAPI):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
