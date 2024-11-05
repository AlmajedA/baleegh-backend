from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from service.translation_service import query
from fastapi.responses import JSONResponse
import modal
import math
import logging
import os
from dotenv import load_dotenv
from util import huggingface_model

load_dotenv()

web_app = FastAPI()
app = modal.App("baleegh")

log_level = os.getenv("LOG_LEVEL", "ERROR").upper()
logging.basicConfig(level=log_level)

image = modal.Image.debian_slim().pip_install(
    "fastapi==0.115.2",
    "ibm_watsonx_ai==1.1.15",  
    "python-dotenv==1.0.1", 
    "requests==2.32.2",       
)

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
    
    result = query(text)
    return JSONResponse(content={"translation": result})

@web_app.get("/test")
def test():
    return huggingface_model.query("test")

@app.function(image=image, secrets=[modal.Secret.from_name("env-variables")])
@modal.asgi_app()
def fastapi_app():
    return web_app