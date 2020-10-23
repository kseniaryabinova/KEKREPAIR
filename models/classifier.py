import torch
from torch import nn
from torchvision import models
from torch.cuda.amp import autocast


class ApartmentRepairmentRecognizer(nn.Module):
    def __init__(
            self,
            pretrained_backbone: bool,
            mixed_precision: bool
    ) -> None:
        super().__init__()
        self.amp = mixed_precision
        self.classifier = models.resnet18(pretrained=pretrained_backbone)
        self.classifier.fc = nn.Linear(512, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.amp:
            with autocast():
                x = self.classifier(x)
        else:
            x = self.classifier(x)
        return x
