import torch
import torchvision
import torch.nn as nn
import trainer
from torchvision import datasets
from torchvision import transforms
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = 'data/train'
image_datasets = datasets.ImageFolder(data_dir,data_transforms)
dataloaders = torch.utils.data.DataLoader(image_datasets,
                                          batch_size=4, shuffle=True)

dataset_sizes = len(image_datasets)
class_names = image_datasets.classes

model_conv = torchvision.models.resnet18(pretrained=True)
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)
model_conv = model_conv.to(device)
trainer.train(model_conv, dataloaders)