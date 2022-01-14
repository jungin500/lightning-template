from torchvision.models import mobilenet_v2
from loguru import logger

import torch
from torch import optim

import pytorch_lightning as pl

# Implement your own model!
TargetModel = mobilenet_v2

class LightningModel(pl.LightningModule):
    def __init__(self, pretrained_path, lr, epoch_milestones, model_args, num_classes, pretrained=True,):
        super().__init__()
        self.model = TargetModel(**model_args)
        self.lr = lr
        self.epoch_milestones = epoch_milestones
        self.loss_module = torch.nn.MSELoss()
        
        if pretrained:
            logger.info("Loading pretrained model")
            state_dict = torch.load(pretrained_path)
            self.model.load_state_dict(state_dict)
        else:
            logger.warning("Training from scratch without pretrained model")
        
    def forward(self, images):
        return self.model(images)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.epoch_milestones, gamma=0.1)
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        output = self.model(images)
        output = torch.sigmoid(output)
        accuracy = ((output > 0.5) == labels).float().mean()
        loss = self.loss_module(output, labels)
        
        self.log("train_acc", accuracy, on_step=False, on_epoch=True)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        output = self.model(images)
        output = torch.sigmoid(output)
        accuracy = ((output > 0.5) == labels).float().mean()
        self.log("valid_acc", accuracy)
        
    def test_step(self, batch, batch_idx):
        images, labels = batch
        output = self.model(images)
        output = torch.sigmoid(output)
        accuracy = ((output > 0.5) == labels).float().mean()
        self.log("test_acc", accuracy)
        
        