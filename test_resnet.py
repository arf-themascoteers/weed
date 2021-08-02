import torch
from torchvision import datasets
from torchvision import transforms
import tester
import weed_dataset_maker
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image_datasets = weed_dataset_maker.create_weed_dataset(is_train=False)
dataloaders = DataLoader(image_datasets, batch_size=10, shuffle=True)

model_conv = torch.load("models/best.pth")
tester.test(model_conv, dataloaders)