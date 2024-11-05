from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("Abdulmohsena/Faseeh")
model = AutoModelForSeq2SeqLM.from_pretrained("Abdulmohsena/Faseeh")

model_name = "Abdulmohsena/Faseeh"

tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn", tgt_lang="arb_Arab")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
generation_config = GenerationConfig.from_pretrained(model_name)

def query(text):
    encoded_ar = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(**encoded_ar, generation_config=generation_config)
    return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)