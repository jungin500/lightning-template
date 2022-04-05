from torchvision.models.mobilenet import mobilenet_v3_small, mobilenet_v3_large
from loguru import logger

import torch
from torch import optim
from torch import nn

import numpy as np
from torchinfo import summary
from sklearn.metrics import precision_score, recall_score, f1_score

import pytorch_lightning as pl

class LightningModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        
        if config.model.type == 'small':
            self.model = mobilenet_v3_small(num_classes=config.model.num_classes, **config.model.args)
        elif config.model.type == 'large':
            self.model = mobilenet_v3_large(num_classes=config.model.num_classes, **config.model.args)
        else:
            raise RuntimeError("Invalid model type: " + config.model.type)
        
        summary(self.model, (1, 3, 224, 224), device='cpu')

        logger.info("Initializing weight (Kaiming for Conv2d, Xavier for Linear")
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

        self.lr = config.train.optim.lr
        self.weight_decay = float(config.train.optim.weight_decay)
        self.momentum = float(config.train.optim.momentum)
        self.loss_module = torch.nn.CrossEntropyLoss()
        self.use_ema = bool(config.train.ema.enabled)
        if self.use_ema:
            logger.error("Required to setup EMA on model.py manually. also, install torch_ema package!")
            # self.ema = ExponentialMovingAverage(self.model.parameters(), decay=config.train.ema.decay)
        self.forward_idx = 0

        if bool(config.model.pretrained.enabled):
            logger.info("Loading pretrained model")
            state_dict = torch.load(config.model.pretrained.path)
            self.load_state_dict(state_dict['state_dict'])
        else:
            logger.warning("Training from scratch without pretrained model")
            
        self.use_dali = config.dataloader.use_dali
        self.dataloader_cfg = config.dataloader  # bunch of key-value storages
        self.headless = config.headless
            
        self.automatic_optimization = False
        self.save_hyperparameters(config)
        
    def prepare_data(self, *args, **kwargs):
        if self.use_dali:
            return
        return super().prepare_data(*args, **kwargs)
        
    def setup(self, *args, **kwargs):
        if not self.use_dali:
            return super().setup(*args, **kwargs)
        
        # Start initilize DALI dataloader
        import nvidia.dali as dali
        from nvidia.dali import pipeline_def
        import nvidia.dali.fn as fn
        import nvidia.dali.types as types

        from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
        import torch
        import os

        @pipeline_def
        def GetImageNetPipeline(device, data_path, shuffle, shard_id=0, num_shards=1):
            jpegs, labels = fn.readers.file(
                file_root=data_path,
                # random_shuffle=False,  # (shuffles inside a initial_fill)
                shuffle_after_epoch=shuffle, # (shuffles entire datasets)
                name="Reader",
                shard_id=shard_id, num_shards=num_shards
            )
            images = fn.decoders.image(jpegs,
                                    device='mixed' if device == 'gpu' else 'cpu',
                                    output_type=types.DALIImageType.RGB)
            
            images = fn.resize(images, size=[224, 224])  # HWC
            images = fn.crop_mirror_normalize(images,
                                            dtype=types.FLOAT,
                                            scale = 1 / 255.,
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225],
                                            output_layout="CHW") # CHW
            if device == "gpu":
                labels = labels.gpu()
            # PyTorch expects labels as INT64
            labels = fn.cast(labels, dtype=types.INT64)
            
            return images, labels
        
        class LightningWrapper(DALIClassificationIterator):
            def __init__(self, *kargs, **kvargs):
                super().__init__(*kargs, **kvargs)

            def __next__(self):
                out = super().__next__()
                # DDP is used so only one pipeline per process
                # also we need to transform dict returned by DALIClassificationIterator to iterable
                # and squeeze the lables
                out = out[0]
                return [out[k] if k != "label" else torch.squeeze(out[k]) for k in self.output_map]

        device_id = self.local_rank
        shard_id = self.global_rank
        num_shards = self.trainer.world_size
        
        trainset_pipeline = GetImageNetPipeline(
            data_path=os.path.join(self.dataloader_cfg.basepath, 'train'),
            batch_size=self.dataloader_cfg.batch_size, device='gpu',
            shuffle=True,
            device_id=device_id, shard_id=shard_id,
            num_shards=num_shards, num_threads=self.dataloader_cfg.num_workers
        )
        
        validset_pipeline = GetImageNetPipeline(
            data_path=os.path.join(self.dataloader_cfg.basepath, 'val'),
            shuffle=False, device='gpu',
            device_id=device_id, shard_id=shard_id,
            # lot number of threads require lot of GPU memory
            batch_size=64, num_threads=2,
            num_shards=num_shards
        )
        
        logger.info("Creating train_loader")
        self.train_loader = LightningWrapper(trainset_pipeline, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=True)
        logger.info("Creating valid_loader")
        self.valid_loader = LightningWrapper(validset_pipeline, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=True)
        
    def train_dataloader(self, *args, **kwargs):
        if not self.use_dali:
            return super().train_dataloader(*args, **kwargs)
        return self.train_loader
    
    def val_dataloader(self, *args, **kwargs):
        if not self.use_dali:
            return super().val_dataloader(*args, **kwargs)
        return self.valid_loader
        
    def forward(self, images):
        print("Batch %05d -> Batches: %d" % (self.forward_idx, images.shape[0]))
        self.forward_idx += 1
        return self.model(images)

    def configure_optimizers(self):
        optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=self.momentum)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True, threshold=3e-3, threshold_mode='abs')
        return [optimizer], [scheduler]
    
    def on_fit_start(self):
        if self.use_ema:
            model_device = next(self.model.parameters()).data.device
            if self.ema.shadow_params[0].data.device != model_device:
                self.ema.to(torch.device(model_device))
        if self.headless:
            logger.info("Begin training in headless mode")
        
    def on_epoch_start(self) -> None:
        if not self.trainer.sanity_checking and self.headless:
            logger.info(f"[State={self.trainer.state.status}] Epoch {self.trainer.current_epoch} begin")
        return super().on_epoch_start()
    
    def on_epoch_end(self) -> None:
        if not self.trainer.sanity_checking and self.headless:
            logger.info(f"[State={self.trainer.state.status}] Epoch {self.trainer.current_epoch} end")
        return super().on_epoch_end()
    
    def training_epoch_end(self, outputs):
        self.forward_idx += 1
        return super().training_epoch_end(outputs)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        output = self.model(images)
        # print('output:', output.shape, ', body->', output)
        # print('labels:', labels.shape, ', body->', labels)
        # Manual optimization
        optimizer = self.optimizers()
        
        optimizer.zero_grad()
        loss = self.loss_module(output, labels)
        self.manual_backward(loss)
        optimizer.step()
        
        with torch.no_grad():
            output = torch.argmax(torch.log_softmax(output, -1), -1)
            accuracy = (output == labels).float().mean()
            
            labels = labels.cpu().numpy()
            output = output.cpu().numpy()
            
        self.log("train/accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss", loss, on_step=True, prog_bar=True)
        # return loss
    
    def training_step_end(self, batch_parts):
        if self.use_ema:
            self.ema.update()
    
    def on_validation_epoch_start(self):
        self.labels = []
        self.outputs = []

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        
        if self.use_ema:
            with self.ema.average_parameters():
                output = self.model(images)
        else:
            output = self.model(images)
            
        output = torch.argmax(torch.log_softmax(output, -1), -1)
        
        labels = labels.float().cpu().numpy()
        output = output.float().cpu().numpy()
        
        self.labels.append(labels)
        self.outputs.append(output)
        
        
    def on_validation_epoch_end(self):
        labels = np.concatenate(self.labels, axis=0)
        outputs = np.concatenate(self.outputs, axis=0)
        
        accuracy = np.mean(np.equal(labels, outputs).astype(np.float32))
        precision = precision_score(labels, outputs, average='weighted', zero_division=0)
        recall = recall_score(labels, outputs, average='weighted', zero_division=0)
        f1 = f1_score(labels, outputs, average='weighted', zero_division=0)
        
        self.log("validation/accuracy", accuracy, prog_bar=True)
        self.log("validation/precision", precision, prog_bar=True)
        self.log("validation/recall", recall, prog_bar=True)
        self.log("validation/f1", f1, prog_bar=True)
        
        # Do not update scheduler while initializing
        if not self.trainer.sanity_checking:
            # Manually update lr_scheduler
            scheduler = self.lr_schedulers()
            # Manual scheduler step
            logger.info("Scheduler - accuracy step %.4f%%" % accuracy)
            scheduler.step(accuracy)
            
