import torch
from sklearn.metrics import precision_score, recall_score

def evaluate(model, loader, device):

    model.eval()

    y_true=[]
    y_pred=[]

    with torch.no_grad():
        for x,y in loader:

            x=x.to(device)
            y=y.to(device)

            out=model(x)
            preds=torch.argmax(out,dim=1)

            y_true+=y.cpu().tolist()
            y_pred+=preds.cpu().tolist()

    precision=precision_score(y_true,y_pred,average="macro")
    recall=recall_score(y_true,y_pred,average="macro")

    return precision,recall
