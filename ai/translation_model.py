import os
import requests
from dotenv import load_dotenv
from allam_model import AllamModel

load_dotenv()

MODEL_URL = os.getenv('MODEL_URL')
HEADERS = {
    "Authorization": f"Bearer {os.getenv('HF_TOKEN')}",
}

SYSTEM_PROMPT = "\n".join([
    "أنت مختص في اللغة العربية، ودورك هو تصحيح الأخطاء الإملائية والنحوية في الجمل العربية التي تتلقاها.",
    "يجب أن تقوم بالتغييرات الضرورية فقط، دون تغيير الكلمات أو المفردات إلا إذا كان ضروريًا للنحو.",
    "أجب فقط بالجملة المصححة دون إضافة أي شرح للتعديلات أو الأخطاء.",
    "مثال: إذا أعطيتك 'في نفس الوقت'، يجب أن ترد بـ'في الوقت نفسه' فقط، أو إذا أعطيتك 'لم أرى' ترد بـ'لم أرَ' فقط دون أي تعليق."
])

def query(payload):
    response = requests.post(MODEL_URL, headers=HEADERS, json=payload)
    response_text = response.json()[0]['translation_text']
    prompt = construct_prompt(response_text)
    return allam(prompt)

def allam(prompt):
    model = AllamModel(
        model_id=os.getenv("IBM_MODEL_ID"), 
        project_id=os.getenv("IBM_PROJECT_ID")
    )
    return model.generate_text(prompt)

def construct_prompt(query):
    return f"{SYSTEM_PROMPT}\n\nالجملة المراد تصحيحها: {query}\nالجملة الصحيحة:"
