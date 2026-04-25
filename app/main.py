from fastapi import FastAPI
from pydantic import BaseModel
from app.model_loader import model_loader
from app.inference import generate_response

app = FastAPI()


# Request schema
class PromptRequest(BaseModel):
    prompt: str


@app.on_event("startup")
def startup_event():
    print("Starting server and loading model...")
    model_loader.load()


@app.get("/")
def root():
    return {"status": "LLM LoRA server running"}


@app.post("/infer")
def infer(request: PromptRequest):
    response = generate_response(request.prompt)
    return {"response": response}