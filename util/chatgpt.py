from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ["CHATGPT_API"],
)

def chatgpt_translation(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system", 
                "content": (
                    "You are a helpful assistant. Translate the following English sentences into classical Arabic sentences "
                    "similar to Quran syntax. Only respond with the exact translation and nothing else."
                    "Here are some examples of translations from English to classical Arabic:\n"
                    "English: He felt lost in the crowd.\n"
                    "Litral Translation: شعر بأنه تائه في الحشد.\n"
                    "Classical Arabic Transltion: غاب في زحام الناس.\n"
                    "English: He was troubled by the unexpected news.\n"
                    "Litral Translation: كان منزعجًا من الخبر غير المتوقع.\n"
                    "Classical Arabic Transltion: أثقلته المفاجأة.\n"
                    "English: He lost hope completely.\n"
                    "Litral Translation: فقد الأمل تمامًا.\n"
                    "Classical Arabic Transltion: خاب أمله كل الخيبة.\n"
                    "English: They turned a blind eye to his mistakes.\n"
                    "Litral Translation: غضوا النظر عن أخطائه.\n"
                    "Classical Arabic Transltion: تجاهلوا زلاته.\n"
                    "English: He felt uncomfortable in their presence.\n"
                    "Litral Translation: شعر بعدم الارتياح في وجودهم.\n"
                    "Classical Arabic Transltion: ضاق بهم.\n"
                    "English: They were quick to judge without understanding.\n"
                    "Litral Translation: كانوا سريعين في الحكم دون فهم.\n"
                    "Classical Arabic Transltion: تسرعوا في الحكم بلا دراية.\n"
                    "English: He couldn’t sleep from worry.\n"
                    "Litral Translation: لم يستطع النوم من القلق.\n"
                    "Classical Arabic Transltion: أرقه القلق.\n"
                    "These are examples of translations from english to true arabic and literal arabic. I want you to translate to the true arabic style.\n"
                    
                )
            },
            {
                "role": "user", 
                "content": (f"English: {prompt}\n"
                            "Classical Arabic Transltion:")
            }
        ]
    )
    return response.choices[0].message.content.strip()