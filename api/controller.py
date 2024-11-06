from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from service.translation_service import query
from fastapi.responses import JSONResponse
import modal
import math
import logging
import os
from dotenv import load_dotenv
from util.modal_image import get_image
from util.chatgpt import chatgpt_translation

load_dotenv()

web_app = FastAPI()
app = modal.App("baleegh")

log_level = os.getenv("LOG_LEVEL", "ERROR").upper()
logging.basicConfig(level=log_level)

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@web_app.get("/")
def get_translation(text: str):
    if not text or not isinstance(text, str):
        return JSONResponse(content={"error": "Invalid input. Text must be a non-empty string."}, status_code=400)
    
    tokens = math.ceil(0.75 * len(text))
    if tokens > 128:
        return JSONResponse(content={"error": f"Input text exceeds the 128 token limit."}, status_code=400)
    
    result = query(text.lower())
    return JSONResponse(content={"translation": result})

@web_app.get("/test")
def test(text: str):
    return chatgpt_translation(text)

@app.function(image=get_image(), secrets=[modal.Secret.from_name("env-variables")], keep_warm=7)
@modal.asgi_app()
def fastapi_app():
    return web_app