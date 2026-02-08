"""
Шаг 2: Дистилляция — обучение модели-ученика на данных учителя.

Ученик — rugpt3small (~125M параметров).
Учитель — rugpt3medium (~350M параметров).

Подход: data distillation (обучение на сгенерированных учителем текстах).
Ученик учится воспроизводить стиль стихов, которые сгенерировал учитель.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm

# ─── Настройки ───────────────────────────────────────────────
STUDENT_MODEL = "ai-forever/rugpt3small_based_on_gpt2"    # 125M — ученик
DATA_FILE     = "data/teacher_poems.jsonl"
OUT_DIR       = "models/student"

EPOCHS     = 6
BATCH_SIZE = 4
LR         = 5e-5
MAX_LENGTH = 128
# ──────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Устройство: {device}")
print(f"Модель ученика: {STUDENT_MODEL}")

tokenizer = GPT2Tokenizer.from_pretrained(STUDENT_MODEL)
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(STUDENT_MODEL).to(device)


class PoetryDataset(Dataset):
    """Датасет стихов, сгенерированных учителем."""

    def __init__(self, path, tokenizer, max_length):
        self.samples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                # Обучаем на полном тексте стиха (промпт + продолжение)
                self.samples.append(obj["poem"])
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.samples[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze()
        attention_mask = enc["attention_mask"].squeeze()

        # labels = input_ids, но на позициях паддинга ставим -100
        # чтобы loss не считался на паддинге
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


dataset = PoetryDataset(DATA_FILE, tokenizer, MAX_LENGTH)
print(f"Размер датасета: {len(dataset)} примеров")

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

total_steps = len(loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=total_steps // 10,
    num_training_steps=total_steps,
)

# ─── Обучение ────────────────────────────────────────────────
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch in tqdm(loader, desc=f"Эпоха {epoch + 1}/{EPOCHS}"):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        loss.backward()

        # Gradient clipping для стабильности
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Эпоха {epoch + 1}: loss = {avg_loss:.4f}")

# ─── Сохранение ──────────────────────────────────────────────
model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
print(f"\nМодель ученика сохранена в {OUT_DIR}")
