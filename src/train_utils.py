from tqdm import tqdm
import torch
from config import *



def train(model, dataloader, optimizer):
    model.train()
    final_loss = 0
    tracker = tqdm(dataloader, total=len(dataloader))
    for data in tracker:
        for key, value in data.items():
            data[key] = value.to(DEVICE)
        optimizer.zero_grad()
        _, loss = model(**data)
        loss.backward()
        optimizer.step()
        final_loss += loss.item()
    return final_loss / len(dataloader)
    

def evaluate(model, dataloader):
    model.eval()
    final_loss = 0
    final_predictions = []
    tracker = tqdm(dataloader, total=len(dataloader))
    # if you have out of memory error put torch.no_grad()
    with torch.no_grad():
        for data in tracker:
            for key, value in data.items():
                data[key] = value.to(DEVICE)
            batch_predictions, loss = model(**data)
            final_predictions.append(batch_predictions)
            final_loss += loss.item()
        return final_predictions, final_loss / len(dataloader)
        
            