import os
import re
from dotenv import load_dotenv

import modal
import weave
import wandb
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig

load_dotenv()

# 1) Define App and volume for HF weights
HF_CACHE_PATH = "/root/.cache/huggingface"
app = modal.App(
    "baleegh",
    secrets=[modal.Secret.from_name("env-variables")],
    volumes={HF_CACHE_PATH: modal.Volume.from_name("hf-cache-vol", create_if_missing=True)},
)

# 2) Build image
image = (
    modal.Image.debian_slim()
    .pip_install(
        "fastapi==0.115.2",
        "python-dotenv==1.0.1",
        "weave==0.51.17",
        "wandb==0.18.5",
        "huggingface-hub",
        "transformers",
        "torch",
    )
)

MINUTES = 60
HOURS = 60 * MINUTES

# 3) Expose FastAPI app via Modal
@app.function(
    image=image,
    container_idle_timeout=5 * MINUTES,
    timeout=24 * HOURS,
    keep_warm=1,
)

@modal.asgi_app(label="baleegh-webapp-fastapi-app")
def fastapi_app():
    # Preload model at container startup:
    os.makedirs(HF_CACHE_PATH, exist_ok=True)
    wandb.login(key=os.environ["WANDB_KEY"])
    weave.init("Baleegh")

    tokenizer = AutoTokenizer.from_pretrained(
        "Abdulmohsena/Faseeh", cache_dir=HF_CACHE_PATH
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "Abdulmohsena/Faseeh", cache_dir=HF_CACHE_PATH
    )
    generation_config = GenerationConfig.from_pretrained(
        "Abdulmohsena/Faseeh", cache_dir=HF_CACHE_PATH
    )

    web_app = FastAPI()
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @web_app.get("/")
    @weave.op()
    def translate(text: str):
        inputs = tokenizer(text, return_tensors="pt")
        with torch.inference_mode():
            tokens = model.generate(**inputs, generation_config=generation_config)
        raw = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
        output = re.sub(r"[A-Za-z:]", "", raw)
        return JSONResponse(content={"translation": output})

    return web_app

if __name__ == "__main__":
    from modal import enable_output

    with enable_output():
        app.run()
