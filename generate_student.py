"""
Шаг 3: Сравнение генерации ДО и ПОСЛЕ дистилляции.

Для каждого тестового промпта генерируем:
  1) Базовая модель (rugpt3small без обучения) — "до"
  2) Модель-ученик (rugpt3small после дистилляции) — "после"

Результат: наглядное сравнение для отчёта.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ─── Настройки ───────────────────────────────────────────────
BASE_MODEL  = "ai-forever/rugpt3small_based_on_gpt2"
STUDENT_DIR = "models/student"

# Тестовые промпты — НЕ совпадают с обучающими,
# чтобы проверить обобщение модели
TEST_PROMPTS = [
    "Люблю тебя, мой край родной",
    "Тихая ночь, звёзды горят",
    "Осень пришла, листья кружат",
    "Дорога длинная зовёт",
    "Снег идёт, и всё бело",
    "О, как прекрасен этот мир",
    "Река течёт среди полей",
]

VARIANTS_PER_PROMPT = 2
# ──────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"


def generate_poem(model, tokenizer, prompt, n_variants=2):
    """Генерирует n_variants продолжений для заданного промпта."""
    results = []
    for _ in range(n_variants):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=True,
                temperature=0.85,
                top_k=50,
                top_p=0.92,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                eos_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        results.append("\n".join(lines[:5]))
    return results


# ─── Загрузка моделей ────────────────────────────────────────
print("=" * 60)
print("Загрузка БАЗОВОЙ модели (до дистилляции)...")
base_tok = GPT2Tokenizer.from_pretrained(BASE_MODEL)
base_model = GPT2LMHeadModel.from_pretrained(BASE_MODEL).to(device)
base_model.eval()
print(f"  Параметров: {sum(p.numel() for p in base_model.parameters()):,}")

print("Загрузка модели-УЧЕНИКА (после дистилляции)...")
student_tok = GPT2Tokenizer.from_pretrained(STUDENT_DIR)
student_model = GPT2LMHeadModel.from_pretrained(STUDENT_DIR).to(device)
student_model.eval()
print(f"  Параметров: {sum(p.numel() for p in student_model.parameters()):,}")
print("=" * 60)

# ─── Генерация и сравнение ───────────────────────────────────
for prompt in TEST_PROMPTS:
    print(f"\n{'=' * 60}")
    print(f"  ПРОМПТ: {prompt}")
    print(f"{'=' * 60}")

    print("\n--- БАЗОВАЯ МОДЕЛЬ (до дистилляции) ---")
    for i, poem in enumerate(generate_poem(base_model, base_tok, prompt, VARIANTS_PER_PROMPT)):
        print(f"\n  Вариант {i + 1}:")
        for line in poem.split("\n"):
            print(f"    {line}")

    print("\n--- МОДЕЛЬ-УЧЕНИК (после дистилляции) ---")
    for i, poem in enumerate(generate_poem(student_model, student_tok, prompt, VARIANTS_PER_PROMPT)):
        print(f"\n  Вариант {i + 1}:")
        for line in poem.split("\n"):
            print(f"    {line}")

print(f"\n{'=' * 60}")
print("Сравнение завершено.")
print("Для отчёта: скопируйте вывод выше и сравните качество генерации.")
