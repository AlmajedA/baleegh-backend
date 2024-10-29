from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from service.translation_service import query
from fastapi.responses import JSONResponse
import modal

web_app = FastAPI()
app = modal.App("baleegh")

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
    result = query(text)
    return JSONResponse(content={"translation": result})

@app.function(image=image, secrets=[modal.Secret.from_name("env-variables")])
@modal.asgi_app()
def fastapi_app():
    return web_app