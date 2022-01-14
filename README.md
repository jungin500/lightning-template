Easy-to-use PyTorch training template

## Details
- [PyTorch Lightning](https://www.pytorchlightning.ai/) for easy-to-use NVIDIA AMP(Automatic Mixed Precision) and Tensorflow-like callback functions (`pytorch_lightning.callbacks.ModelCheckpoint`, `pytorch_lightning.callbacks.LearningRateMonitor`, so on)
- [Wandb](https://wandb.ai) for hyperparameters, loss, accuracy, model parameters and GPU usage logging
- [Hydra](https://github.com/facebookresearch/hydra) for configuration management
- [Loguru](https://github.com/Delgan/loguru) for VERY simple-to-use logging module

## Example pipeline
- Model: [torchvision.models.mobilenet_v2](https://pytorch.org/vision/stable/_modules/torchvision/models/mobilenetv2.html)
- Dataset: BYOD (Bring Your Own Dataset) - simply put `annotations.json` on `/dataset/` folder, see `dataset.py`
- Loss/Criterion: `torch.nn.MSELoss`
- Optimizers: `torch.optim.Adam` with custom learning rate (Specifically on `config.yaml`)

## TODO
- [ ] Customize loss function by yaml config parameters
- [ ] Customize multiple schedulers using yaml config