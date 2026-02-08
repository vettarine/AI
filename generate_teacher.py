"""
Шаг 1: Генерация стихов моделью-учителем.

Учитель — rugpt3medium (~350M параметров).
Генерируем несколько продолжений на каждый промпт,
фильтруем мусор, сохраняем в JSONL.
"""

import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm

# ─── Настройки ───────────────────────────────────────────────
TEACHER_MODEL = "ai-forever/rugpt3medium_based_on_gpt2"   # 350M — учитель
PROMPTS_FILE  = "data/prompts.txt"
OUT_FILE      = "data/teacher_poems.jsonl"
SAMPLES_PER_PROMPT = 5        # вариантов на каждый промпт
MAX_NEW_TOKENS     = 100
MAX_LINES          = 6        # берём не более 6 строк из генерации
# ──────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Устройство: {device}")
print(f"Загрузка учителя: {TEACHER_MODEL}")

tokenizer = GPT2Tokenizer.from_pretrained(TEACHER_MODEL)
model = GPT2LMHeadModel.from_pretrained(TEACHER_MODEL).to(device)
model.eval()

with open(PROMPTS_FILE, encoding="utf-8") as f:
    prompts = [line.strip() for line in f if line.strip()]

print(f"Промптов: {len(prompts)}, вариантов на промпт: {SAMPLES_PER_PROMPT}")
print(f"Ожидаем до {len(prompts) * SAMPLES_PER_PROMPT} примеров (до фильтрации)")


# ─── Фильтрация и очистка ────────────────────────────────────
def is_good_poem(text: str) -> bool:
    """Отбрасываем прозу, мусор, URL-ы и слишком короткие тексты."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < 2:
        return False
    if len(text) < 30:
        return False
    # URL и технический мусор
    bad_markers = [
        "http", "www.", ".ru/", ".com", "©", "(C)", "(c)",
        ">>", "<<", "====", "----", "****",
        "@", "#", "Оригинал", "original", "vk.com",
        "\t", "img", "<", ">",
    ]
    text_lower = text.lower()
    for m in bad_markers:
        if m.lower() in text_lower:
            return False
    # Строки не должны быть слишком длинными (признак прозы)
    for line in lines:
        if len(line) > 80:
            return False
    return True


def clean_poem(text: str) -> str:
    """Берём первые MAX_LINES непустых строк."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return "\n".join(lines[:MAX_LINES])


# ─── Генерация ───────────────────────────────────────────────
count = 0
with open(OUT_FILE, "w", encoding="utf-8") as out:
    for prompt in tqdm(prompts, desc="Промпты"):
        for _ in range(SAMPLES_PER_PROMPT):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.92,
                    repetition_penalty=1.15,
                    no_repeat_ngram_size=3,
                    eos_token_id=tokenizer.eos_token_id,
                )

            text = tokenizer.decode(output[0], skip_special_tokens=True)
            poem = clean_poem(text)

            if not is_good_poem(poem):
                continue

            json.dump({"prompt": prompt, "poem": poem},
                      out, ensure_ascii=False)
            out.write("\n")
            count += 1

print(f"\nГотово! Сохранено {count} стихов в {OUT_FILE}")
