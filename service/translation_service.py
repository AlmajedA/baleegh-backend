import os
import requests
from dotenv import load_dotenv
from util.allam_model import AllamModel
import time

load_dotenv()

MODEL_URL = os.environ['MODEL_URL']
HEADERS = {
    "Authorization": f"Bearer {os.environ['HF_TOKEN']}",
}

def query(text):
    data = {
        "inputs": text,
        "parameters": {
            "src_lang": "eng_Latn",
            "tgt_lang": "arb_Arab"
        }
    }
    
    max_retries = 30
    retry_delay = 2  # seconds

    for attempt in range(max_retries):
        response = requests.post(MODEL_URL, headers=HEADERS, json=data)
        if response.status_code == 200:
            response_text = response.json()[0]['translation_text']
            return allam(response_text)
        else:
            time.sleep(retry_delay)
    
    response.raise_for_status()

def allam(prompt):
    model = AllamModel(
        model_id=os.environ["IBM_MODEL_ID"], 
        project_id=os.environ["IBM_PROJECT_ID"]
    )
    return model.generate_text(prompt)
