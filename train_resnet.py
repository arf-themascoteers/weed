import torch
import torchvision
import torch.nn as nn
import trainer
from torch.utils.data import DataLoader
import weed_dataset_maker
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

image_datasets = weed_dataset_maker.create_weed_dataset(is_train=True)
dataloaders = DataLoader(image_datasets, batch_size=4, shuffle=True)
class_names = image_datasets.classes
num_classes = len(class_names)

model_conv = torchvision.models.resnet18(pretrained=True)
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, num_classes)
model_conv = model_conv.to(device)

trainer.train(model_conv, dataloaders)