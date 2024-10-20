from fastapi import FastAPI
from translation_model import query

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/tranlsations")
def get_translation(text: str):
    data = {
        "inputs": text,
        "parameters": {
            "src_lang": "eng_Latn",
            "tgt_lang": "arb_Arab"
        }
    }

    return query(data)