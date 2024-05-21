from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello World"}

@app.get("/home")
def home():
    return {"message": "Welcome to the home page"}