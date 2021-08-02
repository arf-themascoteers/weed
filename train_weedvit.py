import torch
import trainer
from torch.utils.data import DataLoader
import weed_dataset_maker
import weed_vit
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

image_datasets = weed_dataset_maker.create_weed_dataset(is_train=True)
class_names = image_datasets.classes
num_classes = len(class_names)

model = weed_vit.WeedVit(num_classes).to(device)
model.train()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
num_epochs = 10
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)
    running_loss = 0.0
    running_corrects = 0
    BATCH_SIZE = 10
    TOTAL_BATCH = len(image_datasets) / BATCH_SIZE
    if len(image_datasets) % BATCH_SIZE != 0:
        TOTAL_BATCH += 1
    TOTAL_BATCH = int(TOTAL_BATCH)
    for batch_number in range(0,TOTAL_BATCH,BATCH_SIZE):
        START_INDEX = batch_number * BATCH_SIZE
        END_INDEX = START_INDEX + BATCH_SIZE
        inputs_labels = [(image_datasets[i][0], image_datasets[i][1]) for i in range(START_INDEX, END_INDEX)]
        inputs = [ip[0] for ip in inputs_labels]
        labels = [ip[1] for ip in inputs_labels]
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        labels = torch.tensor(labels).to(device)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data)
        print(f"Epoch #{epoch}, Batch #{batch_number}, Loss = {loss.item()}")
    scheduler.step()

    print(f"Epoch #{epoch}")
    print(f"Loss {running_loss / len(image_datasets)}")
    print(f"Correct {running_corrects} among {len(image_datasets)}")

torch.save(model, 'models/best.pth')