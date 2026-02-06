import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
from tqdm import tqdm

MODEL_NAME = "distilgpt2"
DATA_FILE = "data/teacher_poems.jsonl"
OUT_DIR = "models/student"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)

class PoetryDataset(Dataset):
    def __init__(self, path):
        self.texts = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.texts.append(obj["text"])

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=128,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
        }

dataset = PoetryDataset(DATA_FILE)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

optimizer = AdamW(model.parameters(), lr=5e-5)

model.train()
for epoch in range(3):
    total_loss = 0
    for batch in tqdm(loader, desc=f"Epoch {epoch}"):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}: loss = {total_loss / len(loader):.4f}")

model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
