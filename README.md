Easy-to-use PyTorch Lightning CNN model training template with **MobileNetV2** and **ImageNet** dataloader example

# Usage (`conda` recommended)
```bash
# Change "torch-trainer" to your new environment name
export CONDA_ENV=torch-trainer

# Create new environment along with PyTorch (with GPU support compatible up to RTX 3xxx series)
conda create -n $CONDA_ENV -c pytorch python=3.9 pytorch torchvision cudatoolkit=11.3
conda activate $CONDA_ENV
pip3 install -r requirements.txt

# ... Edit model, configuration

# Go ahead training your model!
python3 train.py wandb.login_key=123abc model.type=large ...
```

## Details
- [PyTorch Lightning](https://www.pytorchlightning.ai/) for easy-to-use NVIDIA AMP(Automatic Mixed Precision) and Tensorflow-like callback functions (`pytorch_lightning.callbacks.ModelCheckpoint`, `pytorch_lightning.callbacks.LearningRateMonitor`, so on)
- [Wandb](https://wandb.ai) for hyperparameters, loss, accuracy, model parameters and GPU usage logging
- [Hydra](https://github.com/facebookresearch/hydra) for configuration management
- [Loguru](https://github.com/Delgan/loguru) for simple and powerful logging
- [DALI](https://github.com/NVIDIA/DALI) for state-of-the-art dataloading framework including NVJPEG (GPU-based JPEG decoding) and on-GPU data augmentation

## Example pipeline
- Model: [torchvision.models.mobilenet](https://pytorch.org/vision/stable/_modules/torchvision/models/mobilenetv2.html) Large/Small
- Dataset: ImageNet with torchvision dataset OR via NVIDIA DALI (See `model.py` and `config.yaml`)
- Loss/Criterion: `torch.nn.CrossEntropyLoss`
- Optimizers: `torch.optim.RMSprop` (See `config.yaml`)

## TODO
- [ ] Customize loss function by yaml config parameters
- [ ] Customize multiple schedulers using yaml config

## Done so far
- [X] Integrate NVIDIA DALI pipeline (Linux only! enable via `dataloader.use_dali=True`)
- [X] Moved training artifacts into single directory
- [X] Remove sample dataloader and integrate into model.py (inside `LightningModule`)
- [X] Support for Multi-GPU training pipeline (as well as Kubernetes)
- [X] Easy Wandb login by specifying API key into `config.yaml` (Be careful not to expose API key!)