import torch
import torchvision
import torch.nn as nn
import trainer
from torch.utils.data import DataLoader
import weed_dataset_maker
import timm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

image_datasets = weed_dataset_maker.create_weed_dataset(is_train=True)
dataloaders = DataLoader(image_datasets, batch_size=4, shuffle=True)
class_names = image_datasets.classes
num_classes = len(class_names)

model = timm.create_model('efficientnet_b3a', pretrained=True)
num_ftrs = model.classifier.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model_conv = model.to(device)

trainer.train(model_conv, dataloaders)