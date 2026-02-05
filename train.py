import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import get_dataloaders
from model import get_model
from optimizer import AdaSmooth
from evaluate import evaluate

DATA_DIR="stanford-dogs-dataset/versions/2/images/Images"
BATCH_SIZE=32
EPOCHS=5
LR=1e-4

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader,val_loader,test_loader,num_classes=get_dataloaders(DATA_DIR,BATCH_SIZE)

criterion=nn.CrossEntropyLoss()

def train_model(model,optimizer,name):

    print(f"\n===== {name} =====")

    model.to(device)

    for epoch in range(EPOCHS):

        model.train()

        loop=tqdm(train_loader)

        for x,y in loop:

            x=x.to(device)
            y=y.to(device)

            optimizer.zero_grad()

            loss=criterion(model(x),y)

            loss.backward()

            optimizer.step()

            loop.set_description(f"Epoch {epoch}")
            loop.set_postfix(loss=loss.item())

    p,r=evaluate(model,test_loader,device)

    print("Precision:",p)
    print("Recall:",r)

    return p,r

# === Эксперимент 1: pretrained + Adam ===
model=get_model(num_classes,True)
optimizer=torch.optim.Adam(model.parameters(),lr=LR)
train_model(model,optimizer,"Pretrained + Adam")

# === Эксперимент 2: pretrained + AdaSmooth ===
model=get_model(num_classes,True)
optimizer=AdaSmooth(model.parameters(),lr=LR)
train_model(model,optimizer,"Pretrained + AdaSmooth")

# === Эксперимент 3: from scratch + Adam ===
model=get_model(num_classes,False)
optimizer=torch.optim.Adam(model.parameters(),lr=LR)
train_model(model,optimizer,"Scratch + Adam")
