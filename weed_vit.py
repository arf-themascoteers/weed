import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTFeatureExtractor, ViTForImageClassification
import torch

class WeedVit(nn.Module):
    def __init__(self, n_classes):
        super(WeedVit, self).__init__()
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        self.vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        self.linear1 = nn.Linear(1000, 500)
        self.lrelu = nn.LeakyReLU(0.2)
        self.linear2 = nn.Linear(500, n_classes)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        x = self.feature_extractor(images=x, return_tensors="pt")
        x = x.to(self.device)
        x = self.vit(**x)
        x = self.linear1(x.logits)
        x = self.lrelu(x)
        x = self.linear2(x)
        return x