from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar, EarlyStopping
from pytorch_lightning.plugins import DDPPlugin

from model import LightningModel

import hydra
from omegaconf import DictConfig
from loguru import logger
import os

# import albumentations as A
# from albumentations.pytorch import ToTensorV2
import torchvision.transforms as T
from torchvision.datasets import ImageNet

@hydra.main(config_path=".", config_name="config")
def main(config: DictConfig) -> None:
    base_dir = config.train.base_dir
    checkpoint_dir = os.path.join(base_dir, 'checkpoints')
    
    # directory internally used by pytorch lightning
    trainer_root_dir = os.path.join(base_dir, 'trainer')
    os.makedirs(trainer_root_dir, exist_ok=True)
    
    wandb_enabled = bool(config.wandb.enabled)
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info(f"Created checkpoint directory {checkpoint_dir}")
    
    if wandb_enabled:    
        # directory internally used by wandb
        wandb_log_dir = os.path.join(base_dir, 'wandb')
        os.makedirs(wandb_log_dir, exist_ok=True)
    
        if config.wandb.login_key != 'None' and config.wandb.login_key is not None:
            import wandb
            wandb.login(key=config.wandb.login_key)
        
        lightning_logger = WandbLogger(
            name=config.expr_name,
            project=config.wandb.project,
            save_dir=base_dir
        )
    else:
        lightning_logger = True
    
    
    if config.dataloader.use_dali:
        pass  # Initializing inside model
    else:
        train_transform = T.Compose([
            T.Resize([224, 224]),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomAutocontrast(p=0.5),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        val_transform = T.Compose([
            T.Resize([224, 224]),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        logger.info("Loading trainset")
        train_dataset = ImageNet(
            root=config.dataloader.basepath,
            split='train',
            transform=train_transform,
            # target_transform=target_transform
        )
        logger.info("Done loading trainset")
        
        logger.info("Loading validset")
        val_dataset = ImageNet(
            root=config.dataloader.basepath,
            split='val',
            transform=val_transform,
            # target_transform=target_transform
        )
        logger.info("Done loading validset")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.dataloader.batch_size,
            shuffle=True,
            num_workers=config.dataloader.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )

    model = LightningModel(config)
    
    if wandb_enabled:
        lightning_logger.watch(model)

    if config.train.gpus == 0:
        logger.warning("Training with CPU, falling back to BF16 format")

    strategy = None
    if config.train.strategy and config.train.strategy.lower() != 'none':
        strategy = config.train.strategy
        if strategy.lower() == 'ddp':
            strategy = DDPPlugin(find_unused_parameters=False)
        
    trainer = pl.Trainer(
        logger=lightning_logger,
        default_root_dir=trainer_root_dir,
        accelerator='gpu' if config.train.gpus > 0 else 'cpu',
        gpus=config.train.gpus,
        strategy=strategy,
        precision=config.train.precision if config.train.gpus > 0 else 'bf16',
        max_epochs=config.train.epochs,
        limit_train_batches=config.train.limit_train_batches,
        limit_val_batches=config.train.limit_val_batches,
        enable_progress_bar=not config.headless,
        callbacks=[
            # TQDMProgressBar(refresh_rate=1),
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename='%s-epoch{epoch:04d}-val_acc{validation/accuracy:.2f}' % (config.expr_name),
                mode="max", monitor="validation/f1"
                ),
            LearningRateMonitor("epoch"),
            EarlyStopping(
                monitor="validation/accuracy",
                patience=5,
                min_delta=0.005,
                mode="max",
                )
        ]
    )

    if wandb_enabled:
        trainer.logger._log_graph = True
        trainer.logger._default_hp_metric = None

    if config.dataloader.use_dali:
        trainer.fit(model)  # Dataloader is initialized inside model
    else:            
        trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main()
