import os
import yaml

import torch
from torch_optimizer import Ranger
from torchvision import transforms
from torch.cuda.amp import GradScaler

from dl_utils.training import train_model
from dl_utils.general import parse_arguments
from dl_utils.training import get_data_loaders
from dl_utils.training import compute_class_weights
from models.classifier import ApartmentRepairmentRecognizer


if __name__ == '__main__':
    args = parse_arguments()

    with open(args.c, 'r') as fp:
        config = yaml.load(fp, Loader=yaml.SafeLoader)

    torch.manual_seed(config['seed'])
    torch.backends.cudnn.benchmark = True

    if not os.path.exists(config['experiment_result_path']):
        os.mkdir(config['experiment_result_path'])
    weights_path = os.path.join(config['experiment_result_path'], 'weights')
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)

    if config['mixed_precision']:
        scaler = GradScaler()
    else:
        scaler = None

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_loader, val_loader = get_data_loaders(
        config,
        train_transforms,
        val_transforms
    )

    model = ApartmentRepairmentRecognizer(
        config['n_classes'],
        pretrained_backbone=True,
        mixed_precision=config['mixed_precision']
    )

    optimizer = Ranger(
        model.parameters(),
        lr=config['learning_rate'] / 3.,
        weight_decay=config['weight_decay']
    )

    scheduler = None
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=config['learning_rate'],
    #     total_steps=config['n_epochs'] * config['steps_per_epoch'],
    #     epochs=config['n_epochs'],
    #     steps_per_epoch=config['steps_per_epoch']
    # )

    if config['use_gpu']:
        device = torch.device('cuda:0')
        if config['n_gpus'] > 1:
            model = torch.nn.DataParallel(model)
    else:
        device = torch.device('cpu')

    class_weights = compute_class_weights(
        config['unique_classes'],
        config['n_samples_per_class'],
        device
    )
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    model = model.to(device)
    model.train()
    train_model(
        model,
        optimizer,
        scheduler,
        criterion,
        config['n_epochs'],
        train_loader,
        val_loader,
        device,
        scaler,
        config['display_step'],
        weights_path,
        config['experiment_result_path']
    )
