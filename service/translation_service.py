import os
from dotenv import load_dotenv
from util.allam_model import AllamModel
# import weave
# import wandb
import requests
from util.chatgpt import chatgpt_translation


load_dotenv()

# wandb.login(key=os.environ['WANDB_KEY'])
# weave.init("Baleegh")

MODEL_URL = os.environ['MODEL_URL']
HEADERS = {
    "Authorization": f"Bearer {os.environ['HF_TOKEN']}",
}

# @weave.op
def query(text):
    data = {
        "inputs": text,
        "parameters": {
            "src_lang": "eng_Latn",
            "tgt_lang": "arb_Arab"
        }
    }
        
    response = requests.post(MODEL_URL, headers=HEADERS, json=data)
    response_text = response.json()[0]['translation_text']
            
        
    gpt_response = chatgpt_translation(text)
    return allam(response_text, gpt_response)

def allam(query1, query2):
    model = AllamModel(
        model_id=os.environ["IBM_MODEL_ID"], 
        project_id=os.environ["IBM_PROJECT_ID"]
    )
    return model.generate_text(query1, query2)
