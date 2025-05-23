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

# 1) Define Stub and volume for HF weights
stub = modal.Stub(
    "baleegh",
    secrets=[modal.Secret.from_name("env-variables")],
)
HF_CACHE_PATH = "/root/.cache/huggingface"
hf_cache = modal.Volume().persist("hf-cache-vol")

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

# 3) One “setup” function to download & load model & config
@stub.function(
    image=image,
    mounts={hf_cache: HF_CACHE_PATH},
    container_idle_timeout=5 * MINUTES,
    keep_warm=3,
)
def setup_models():
    os.makedirs(HF_CACHE_PATH, exist_ok=True)
    wandb.login(key=os.environ["WANDB_KEY"])
    weave.init("Baleegh")
    tokenizer = AutoTokenizer.from_pretrained(
        "Abdulmohsena/Faseeh", cache_dir=HF_CACHE_PATH
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "Abdulmohsena/Faseeh", cache_dir=HF_CACHE_PATH
    )
    gen_cfg = GenerationConfig.from_pretrained(
        "Abdulmohsena/Faseeh", cache_dir=HF_CACHE_PATH
    )
    return tokenizer, model, gen_cfg

tokenizer, model, generation_config = setup_models.call()

# 4) Expose FastAPI app
@stub.asgi_app(
    image=image,
    mounts={hf_cache: HF_CACHE_PATH},
    container_idle_timeout=5 * MINUTES,
    timeout=24 * HOURS,
    keep_warm=3,
)
def fastapi_app():
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    def translate(text: str):
        inputs = tokenizer(text, return_tensors="pt")
        with torch.inference_mode():
            tokens = model.generate(**inputs, generation_config=generation_config)
        raw = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
        output = re.sub(r"[A-Za-z:]", "", raw)
        return JSONResponse(content={"translation": output})

    return app


if __name__ == "__main__":
    stub.run()
