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
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
from util.allam_model import AllamModel

load_dotenv()
MODEL_DIR = "BALEEGH"
MINUTES = 60
HOURS = 60 * MINUTES

app = modal.App("baleegh", image=get_image(), secrets=[modal.Secret.from_name("env-variables")])

log_level = os.getenv("LOG_LEVEL", "ERROR").upper()
logging.basicConfig(level=log_level)

@app.cls(container_idle_timeout=5 * MINUTES, timeout=24 * HOURS, cpu=3, keep_warm=3)
class WebApp:
    def __init__(self):
        # Initialize FastAPI app
        self.web_app = FastAPI()
        
        # Load any resources here (e.g., Hugging Face model)
        # self.model = load_your_model()
        
        # Set up CORS
        self.web_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Adjust as needed
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self.web_app.add_api_route("/", self.query)

    @modal.build()  # add another step to the image build
    def download_model_to_folder(self):
        from huggingface_hub import snapshot_download

        os.makedirs(MODEL_DIR, exist_ok=True)
        snapshot_download("Abdulmohsena/Faseeh", local_dir=MODEL_DIR)
    
    @modal.enter()
    def setup(self):
        self.tokenizer = AutoTokenizer.from_pretrained("Abdulmohsena/Faseeh")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Abdulmohsena/Faseeh")
        self.generation_config = GenerationConfig.from_pretrained("Abdulmohsena/Faseeh")

    def query(self, text: str):

        encoded_ar = self.tokenizer(text, return_tensors="pt")
        generated_tokens = self.model.generate(**encoded_ar, generation_config=self.generation_config)

        model_response = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        gpt_response = chatgpt_translation(text)

        return self.allam(model_response, gpt_response)
    
    def allam(self, query1, query2):
        model = AllamModel(
            model_id=os.environ["IBM_MODEL_ID"], 
            project_id=os.environ["IBM_PROJECT_ID"]
        )
        return model.generate_text(query1, query2)
    
    @modal.asgi_app()
    def fastapi_app(self):
        return self.web_app

