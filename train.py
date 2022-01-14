import torch
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar

from model import LightningModel
from dataset import TargetDataset

import hydra
from omegaconf import DictConfig
from loguru import logger
import os

import albumentations as A

@hydra.main(config_path=".", config_name="config")
def main(config: DictConfig) -> None:
    
    wandb_logger = WandbLogger(
        name=config.wandb.name,
        project=config.wandb.project
    )
    
    transform = A.Compose([
        A.Flip(p=0.5), # HorizontalFlip + VerticalFlip
        A.RandomBrightnessContrast(p=0.2),
        A.Rotate(limit=(-30, 30), p=0.3),
        A.Resize(width=224, height=224),
    ])
    test_transform = A.Compose([
        A.Resize(width=224, height=224)
    ])
    
    dataset = TargetDataset(
        dataset_path=config.dataloader.dataset_path,
        transform=transform, test_transform=test_transform
    )
    
    total_items = len(dataset)
    trainset_items = int(total_items * (1.0 - config.dataloader.validset_portion))
    valset_items = total_items - trainset_items
    train_dataset, val_dataset = random_split(dataset, [trainset_items, valset_items])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.dataloader.batch_size,
        shuffle=True,
        num_workers=config.dataloader.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.dataloader.batch_size,
        shuffle=False,
        num_workers=config.dataloader.num_workers,
        pin_memory=True
    )
    
    model = LightningModel(
        pretrained_path=config.model.pretrained_path,
        lr=config.train.lr,
        epoch_milestones=config.train.optimizer_params.lr_scheduler.milestones,
        model_args=config.model.args,
        num_classes=config.dataloader.num_classes,
        pretrained=config.model.pretrained
    )
    
    if config.train.gpus == 0:
        logger.warning("Training with CPU, falling back to BF16 format")
    
    trainer = pl.Trainer(
        logger=wandb_logger,
        default_root_dir=os.path.join(os.getcwd(), config.train.checkpoint_base_dir),
        accelerator='gpu' if config.train.gpus > 0 else 'cpu',
        gpus=config.train.gpus,
        precision=config.train.precision if config.train.gpus > 0 else 'bf16',
        max_epochs=config.train.epochs,
        callbacks=[
            TQDMProgressBar(refresh_rate=1),
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="valid_acc"),
            LearningRateMonitor("epoch")
        ]
    )
    
    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None
    
    logger.info("Start training from epoch 0")
    trainer.fit(model, train_loader, val_loader)
    

if __name__ == '__main__':
    main()
