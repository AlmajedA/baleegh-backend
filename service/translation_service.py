import os
from dotenv import load_dotenv
from util.allam_model import AllamModel
# import weave
# import wandb
from util.chatgpt import chatgpt_translation


load_dotenv()

# wandb.login(key=os.environ['WANDB_KEY'])
# weave.init("Baleegh")


# @weave.op
def query(text):
    return "testtes"  
    # gpt_response = chatgpt_translation(text)
    # return allam(response_text, gpt_response)

def allam(query1, query2):
    model = AllamModel(
        model_id=os.environ["IBM_MODEL_ID"], 
        project_id=os.environ["IBM_PROJECT_ID"]
    )
    return model.generate_text(query1, query2)
