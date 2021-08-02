import torch
from PIL import Image
import torch
import timm
import requests
import torchvision.transforms as transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

model = timm.create_model('deit_base_patch16_224', pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256, interpolation=3),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])

img = Image.open("data/train/lantana/50.jpg")
img = transform(img)[None,]
out = model(img)
clsidx = torch.argmax(out)
print(clsidx.item())
print("\n".join(torch.hub.list(github=True)))
exit(0)