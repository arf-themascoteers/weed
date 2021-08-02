import torch
from torchvision import datasets
from torchvision import transforms
import tester

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = 'data/val'
image_datasets = datasets.ImageFolder(data_dir,data_transforms)
dataloaders = torch.utils.data.DataLoader(image_datasets,
                                          batch_size=4, shuffle=True)

dataset_sizes = len(image_datasets)
class_names = image_datasets.classes

model_conv = torch.load("models/best.pth")
tester.test(model_conv, dataloaders)