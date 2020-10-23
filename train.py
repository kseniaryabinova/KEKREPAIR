import os
import yaml
import argparse

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from models.classifier import ApartmentRepairRecognizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apartment repair recognizer')
    parser.add_argument(
        '--c',
        '-config',
        type=str,
        required=True,
        help='Path for training configuration .yaml file.'
    )
    args = parser.parse_args()
    with open(args.c, 'r') as fp:
        config = yaml.load(fp, Loader=yaml.SafeLoader)

    T = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = ImageFolder(root=config['dataset_path'], transform=T)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config['batch_size'],
        num_workers=config['dataloader_num_workers'],
        pin_memory=True
    )

    criterion = torch.nn.CrossEntropyLoss()

    model = ApartmentRepairRecognizer(True)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    model = model.cuda()

    for epoch in range(config['n_epochs']):
        running_loss = 0.0
        for i, batch in enumerate(dataloader):
            images, labels = batch[0].cuda(), batch[1].cuda()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 5 == 4:
                print(running_loss / 5)
                running_loss = 0.0
