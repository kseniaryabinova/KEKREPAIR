import torch
from torch import nn
from torchvision import models


class ApartmentRepairRecognizer(nn.Module):
    def __init__(self, pretrained_backbne: bool) -> None:
        super().__init__()
        self.classifier = models.resnet18(pretrained=pretrained_backbne)
        self.classifier.fc = nn.Linear(512, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)
