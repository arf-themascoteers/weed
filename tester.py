import time
import copy
import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test(model, dataloader):
    print('Testing')
    print('-' * 10)
    model.eval()
    running_corrects = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
    percent = round(running_corrects.item() / len(dataloader.dataset) * 100, 2)
    print(f"Correct {running_corrects} among {len(dataloader.dataset)} - {percent}%")
