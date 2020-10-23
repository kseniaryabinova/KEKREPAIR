import os
import yaml
import argparse

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.cuda.amp import GradScaler, autocast

from models.classifier import ApartmentRepairmentRecognizer


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

    if config['mixed_precision']:
        scaler = GradScaler()
    else:
        scaler = None

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

    model = ApartmentRepairmentRecognizer(
        pretrained_backbone=True,
        mixed_precision=config['mixed_precision']
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    if config['use_gpu']:
        device = torch.device('cuda:0')
        if config['n_gpus'] > 1:
            model = torch.nn.DataParallel(model)
    else:
        device = torch.device('cpu')
    model = model.to(device)
    model.train()

    for epoch in range(config['n_epochs']):
        running_loss = 0.0
        for i, batch in enumerate(dataloader):
            images, labels = batch[0].cuda(), batch[1].cuda()

            optimizer.zero_grad()
            if scaler is not None:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

            if i % 5 == 4:
                print(running_loss / 5)
                running_loss = 0.0
