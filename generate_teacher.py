import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm

MODEL_NAME = "ai-forever/rugpt3small_based_on_gpt2"
PROMPTS_FILE = "data/prompts.txt"
OUT_FILE = "data/teacher_poems.jsonl"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

with open(PROMPTS_FILE, encoding="utf-8") as f:
    prompts = [line.strip() for line in f if line.strip()]

with open(OUT_FILE, "w", encoding="utf-8") as out:
    for prompt in tqdm(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            eos_token_id=tokenizer.eos_token_id
        )

        text = tokenizer.decode(output[0], skip_special_tokens=True)

        lines = text.split("\n")
        lines = [l for l in lines if l.strip()]

        text = "\n".join(lines[:6])
        json.dump({"prompt": prompt, "text": text}, out, ensure_ascii=False)
        out.write("\n")
