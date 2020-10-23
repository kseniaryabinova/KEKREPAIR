import os
import math
from copy import deepcopy
from typing import Union, Tuple, Dict

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score

from dl_utils.general import is_parallel

def get_data_loaders(
        confg_file: dict,
        train_transforms: transforms.Compose,
        val_transforms: transforms.Compose
) -> Tuple[DataLoader, Union[DataLoader, None]]:
    """
    Build DataLoader's instances for training and
    validation procedure.

    :param confg_file: Configuration .yaml file;
    :param train_transforms: Augmentation's composition for training;
    :param val_transforms: Augmentation's composition for validation;
    """
    train_set = ImageFolder(
        root=confg_file['train_path'],
        transform=train_transforms
    )
    train_loader = DataLoader(
        train_set,
        batch_size=confg_file['train_batch_size'],
        num_workers=confg_file['dataloader_num_workers'],
        pin_memory=True
    )

    if confg_file['val_path']:
        val_set = ImageFolder(
            root=confg_file['val_path'],
            transform=val_transforms
        )
        val_loader = DataLoader(
            val_set,
            batch_size=confg_file['val_batch_size']
        )
        return train_loader, val_loader
    else:
        return train_loader, None


def train_model(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.OneCycleLR,
        criterion: torch.nn.CrossEntropyLoss,
        n_epochs: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        grad_scaler: GradScaler,
        display_step: int,
        save_path: str,
        log_path: str
) -> None:
    """
    Training loop.

    :param model: Model for training.
    :param optimizer: Model optimizer. Could be any
            subclass of torch.optim.Optimizer class;
    :param criterion: Cross-entropy loss function;
    :param n_epochs: Number of epochs;
    :param train_loader: Dataset loader for training;
    :param val_loader: Dataset loader for validation;
    :param device: Torch device type;
    :param grad_scaler: AMP gradient scaler;
    :param display_step: Iteration factor for logging;
    :param save_path: Path for saving model weights;
    :param log_path: Path for tensorboard logging.
    """
    tb_writer = SummaryWriter(log_dir=log_path, flush_secs=10)
    step = 0
    best_f1 = 0.
    ema = EMA(model)
    for epoch in range(n_epochs):
        running_loss = 0.0
        for batch in train_loader:
            images, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            if grad_scaler is not None:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            ema.update(model)
            scheduler.step()
            running_loss += loss.item()

            if step % display_step == display_step - 1:
                print(f'[INFO] Epoch {epoch + 1}, Iteration {step + 1} '
                      f'Loss={running_loss / display_step}')
                tb_writer.add_scalar('Loss', running_loss/display_step, step)
                running_loss = 0.0
            step += 1
        metrics = eval_model(ema.ema, val_loader, device)
        tb_writer.add_scalars(
            'Metrics',
            {
                'Accuracy': metrics['accuracy'],
                'F1-score': metrics['F1'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall']
            },
            epoch
        )
        f1 = metrics['F1']
        print(f'[METRICS] After epoch {epoch + 1}: '
              f'Accuracy={round(metrics["accuracy"], 3)}, '
              f'F1-score={round(metrics["F1"], 3)}, '
              f'Precision={round(metrics["precision"], 3)}, '
              f'Recall={round(metrics["recall"], 3)}')
        if f1 > best_f1:
            print(f'[NEW BEST MODEL] Obtained new best model with '
                  f'F1-score={round(f1, 3)}')
            best_f1 = f1
            save_name = 'model_' + str(round(f1, 3)) + '.pth'
            save = os.path.join(save_path, save_name)
            torch.save(ema.ema.state_dict(), save)


def eval_model(
        model: torch.nn.Module,
        val_loader: DataLoader,
        device: torch.device
) -> Dict[str, float]:
    """
    Evaluate model in terms of accuracy, F1-score,
    precision and recall metrics.

    :param model: Model for evaluation;
    :param val_loader: Dataset loader for validation;
    :param device: Torch device type.

    :return: Dictionary with metrics.
    """
    model.eval()
    predicted = []
    ground_truth = []

    with torch.no_grad():
        for batch in val_loader:
            images, labels = batch[0].to(device), batch[1].to(device)
            predictions = model(images)
            _, predictions = torch.max(predictions, dim=1)
            ground_truth.extend(labels.detach().cpu().numpy())
            predicted.extend(predictions.detach().cpu().numpy())
    model.train()

    accuracy = accuracy_score(ground_truth, predicted)
    f1 = f1_score(ground_truth, predicted, average='macro', zero_division=0)
    precision = precision_score(
        ground_truth,
        predicted,
        average='macro',
        zero_division=0
    )
    recall = recall_score(
        ground_truth,
        predicted,
        average='macro',
        zero_division=0
    )
    return {
        'F1': f1,
        'recall': recall,
        'accuracy': accuracy,
        'precision': precision,
    }


class EMA:
    """
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """
    def __init__(
            self,
            model: torch.nn.Module,
            decay: float = 0.9999,
            updates: int = 0
    ) -> None:
        """
        :param model: Model for averaging;
        :param decay: Decay coefficient;
        :param updates: Counter of EMA updates.
        """
        self.ema = deepcopy(
            model.module if is_parallel(model) else model
        ).eval()
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model: torch.nn.Module) -> None:
        """
        Update model weights according to exponential
        moving average rule.

        :param model: Model which weights should be updated.
        """
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)
            if is_parallel(model):
                msd = model.module.state_dict()
            else:
                model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()
