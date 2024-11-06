import os
import requests
from dotenv import load_dotenv
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai import Credentials

load_dotenv()

def get_ibm_access_token(api_key):
    url = 'https://iam.cloud.ibm.com/identity/token'
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    data = {
        'grant_type': 'urn:ibm:params:oauth:grant-type:apikey',
        'apikey': api_key
    }

    response = requests.post(url, headers=headers, data=data)
    response.raise_for_status()  # Ensure we raise an error for bad responses
    return response.json().get('access_token')

class AllamModel:
    def __init__(self, model_id, project_id, url="https://eu-de.ml.cloud.ibm.com"):
        api_key = os.environ['IBM_API_KEY']
        if not api_key:
            raise ValueError("IBM_API_KEY environment variable not set")

        self.credentials = Credentials(
            url=url,
            token=get_ibm_access_token(api_key)
        )
        self.parameters = {
            "decoding_method": "greedy",
            "max_new_tokens": 100,
            "repetition_penalty": 1.0,
            "temperature": 0.0,
        }
        self.model = Model(
            model_id=model_id,
            credentials=self.credentials,
            params=self.parameters,
            project_id=project_id
        )
        self.system_prompt = "\n".join([
            "أنت مختص في اللغة العربية الكلاسيكية، ودورك هو تصحيح الأخطاء الإملائية والنحوية في الجمل العربية التي تتلقاها مع مراعاة تفضيل الجملة الأقرب إلى أساليب اللغة العربية الفصحى القديمة وأساليب القرآن الكريم.",
            "سوف تتلقى جملتين عربيتين، ويجب عليك اختيار الجملة الأولى دائمًا كخيار أساسي، ولا تحيد عن اختيارها إلا في حالات استثنائية.",
            "اختر الجملة الثانية فقط إذا كانت الجملة الأولى غير منطقية أو غير مفهومة إطلاقًا، ومع ذلك، حاول استخدام كلمات الجملة الأولى بقدر الإمكان في التصحيح.",
            "إذا كانت الجملتان متشابهتين تمامًا من حيث الصحة اللغوية والنحوية، فاعتبر الجملة الأولى الخيار الافتراضي دائمًا دون استثناء.",
            "مهم جدًا: أجب بجملة واحدة فقط، وهي الجملة المختارة بعد التصحيح، دون إضافة أي نص آخر، مثل: 'الجملة الأولى' أو 'الجملة الثانية' أو أي شروحات أخرى. لا تضف شيئًا سوى الجملة المصححة النهائية.",
            "قم بإجراء التغييرات الضرورية فقط على الجملة المختارة، وتجنب تغيير الكلمات أو المفردات إلا إذا كان ذلك ضروريًا لسلامة النحو والتعبير.",
            "مثال: إذا كانت الجملة 'في نفس الوقت'، يجب أن ترد بـ'في الوقت نفسه' فقط، أو إذا كانت الجملة 'لم أرى'، ترد بـ'لم أرَ' فقط دون أي تعليق."
        ])
    
    def construct_prompt(self, query1, query2):
         return f"{self.system_prompt}\nالجملة الأولى: {query1}\nالجملة الثانية: {query2}\nالجملة المختارة والمصححة:"
    
    def generate_text(self, query1, query2):
        prompt = self.construct_prompt(query1, query2)
        return self.model.generate_text(prompt)
