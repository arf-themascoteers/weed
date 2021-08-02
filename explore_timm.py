import torch
from PIL import Image
import torch
import timm
import requests
import torchvision.transforms as transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = timm.create_model('vit_base_patch16_224', pretrained=True).to(device)
model.eval()
config = resolve_data_config({}, model=model)
transform = create_transform(**config)

img = Image.open("data/train/lantana/50.jpg").convert('RGB')
img = transform(img).unsqueeze(0).to(device)
out = model(img)
clsidx = torch.argmax(out)
print(clsidx.item())

print(model)
exit(0)