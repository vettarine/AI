import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

MODEL_DIR = "models/student"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR).to(device)
model.eval()

prompt = "Люблю тебя, мой край родной"

inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    output = model.generate(
    **inputs,
    max_length=120,
    do_sample=True,
    temperature=1.0,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.2,
    eos_token_id=tokenizer.eos_token_id
)   

print(tokenizer.decode(output[0], skip_special_tokens=True))
