


# Zeroshot Lightning
![Structure](https://drive.google.com/uc?export=view&id=1KjeNm8MlGKfpDG5TYNINK0jrJTUoHIp0)  
  
ZSL using Pytorch Lightning. Based on CLIP.
## Installation
Create a [wandb](https://wandb.ai/site) account.

    git clone https://github.com/luca-medeiros/zeroshot_lightning
    # Install pytorch (1.7.1 or above) and torchvision
    cd zeroshot_lightning
    pip install -r requirements
    wandb login

## Training
First, the dataset should be folders of images of shape 224x224.
The name of the folders will be used to train the text encoder - On this repo should be in korean (hangul).
Then the config.yml file:

    train_path: FOLDER_PATH for the train split.
    valid_path: FOLDER_PATH for the valid split.
    opt: sgd - Choose the optimizer [sgd, adamp, sgdp, madgrad]
    module: pretrained - Image backbone encoder [regnet, deit, efficientnet_b3a, pretrained]
    instance: WANDB_PROJECT/WANDB_EXPERIMENT_NAME - Also path for storing the weights locally.
    oversample: true - Oversample minority classes to match majority class.
    logit_scale: 100.0 - Scale to multiply logits.
    epochs: 150
    num_workers: 16
    b: 128 - Batch size.
    low_dim: 768 - Dimension of image encoder vector.
    lr: 0.00052
    resume: '' - Leave empty if training from scratch.

Get weights for pretrained [model] - Highly recommended train with pretrained module (set module to pretrained on config).
Running the train script with multi-gpu support:

    python train.py --config config.yml --gpus 0 1 2 --lr_finder
Track the training metrics on the generated wandb link.

## Eval
    python train.py --config config.yml --gpus 0 --eval

## Test
WIP
