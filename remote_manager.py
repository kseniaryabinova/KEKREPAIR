from multiprocessing.managers import BaseManager

import torch
from torch import nn
from torchvision import models

import cv2
import numpy as np


device = torch.device('cuda:0')


class ApartmentRepairmentRecognizer(nn.Module):
    def __init__(
            self,
            pretrained_backbone: bool,
            mixed_precision: bool
    ) -> None:
        super().__init__()
        self.amp = mixed_precision
        self.classifier = models.resnet18(pretrained=pretrained_backbone)
        self.classifier.fc = nn.Linear(512, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.classifier(x)
        return x


model = ApartmentRepairmentRecognizer(
        pretrained_backbone=True,
        mixed_precision=False)

path = 'model_0.619.pth'

model.load_state_dict(torch.load(path))
model.to(device)


def preprocess(frame):
    prep_frame = cv2.resize(frame, (224, 224))
    prep_frame = np.transpose(prep_frame, (2, 0, 1))
    prep_frame = prep_frame / 255
    prep_frame = torch.tensor(prep_frame)
    prep_frame = torch.unsqueeze(prep_frame, 0)
    return prep_frame


def predict(frame):
    prep_frame = preprocess(frame)
    print(prep_frame.shape)
    predictions = model(prep_frame.to(device, dtype=torch.float))
    print(predictions.cpu())
    predictions =  torch.squeeze(predictions.cpu())

    label_id = torch.argmax(predictions.detach())
    return label_id.item()


BaseManager.register('predict', callable=predict)
manager = BaseManager(address=('0.0.0.0', 1448),
                      authkey='ksenia'.encode('utf-8'))

server = manager.get_server()
print('start to serving...')
server.serve_forever()
