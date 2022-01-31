#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 14:21:18 2021


@author: Luca Medeiros, lucamedeiros@outlook.com
"""

import os
import yaml
import wandb
import warnings
import pytorch_lightning as pl

import callbacks
from argparse import ArgumentParser, Namespace
from pytorch_lightning.loggers import WandbLogger

from model import ZSLModel
from dataloader import DataModule


warnings.filterwarnings("ignore")


def read_config(config):
    with open(config, 'r') as stream:
        cfg_dict = yaml.safe_load(stream)
    cfg = Namespace(**cfg_dict)

    return cfg


def batch_size_tuner(trainer, model):
    new_batch_size = trainer.tuner.scale_batch_size(model, mode="binsearch")

    return new_batch_size


def main(args):
    cfg = read_config(args.config)
    pl.utilities.seed.seed_everything(seed=42)
    data_module = DataModule(cfg)
    data_module.setup()
    project, instance = cfg.instance.split('/')
    cfg.directory = f'./data/{project}--{instance}'
    if not os.path.isdir(cfg.directory):
        os.makedirs(cfg.directory)

    if cfg.resume != '':
        print('Resuming...')
        model = ZSLModel.load_from_checkpoint(cfg.resume)
    else:
        model = ZSLModel(cfg, data_module.classes, data_module.val_classes, data_module.ndata)

    # ------------
    # training
    # ------------
    job_type = 'train' if not args.eval else 'eval'
    wandb_logger = WandbLogger(project=project,
                               name=instance,
                               job_type=job_type,
                               offline=args.wandb_offline)
    accelerator = None
    replace_sampler_ddp = False
    if len(args.gpus) > 1:
        accelerator = 'dp'
        replace_sampler_ddp = True
    callbacks_list = [callbacks.progress_bar,
                      callbacks.model_checkpoint(cfg),
                      callbacks.lr_monitor,
                      callbacks.early_stopping,
                      callbacks.MetricsLogCallback(cfg, data_module.classes),
                      callbacks.DistributionLogCallback(wandb_logger, data_module),
                      callbacks.CodeSnapshot(root='./',
                                             output_file=os.path.join(cfg.directory, 'source_code.zip'),
                                             filetype=['.py', '.yml']
                                             )
                      ]
    trainer = pl.Trainer(gpus=args.gpus,
                         accelerator=accelerator,
                         # replace_sampler_ddp=replace_sampler_ddp,
                         val_check_interval=1.0,
                         #limit_train_batches=0.01,
                         benchmark=True,
                         max_epochs=cfg.epochs,
                         default_root_dir=cfg.directory,
                         logger=wandb_logger,
                         callbacks=callbacks_list,
                         )

    if args.eval:
        print('Eval mode')
        trainer.validate(model, val_dataloaders=data_module.val_dataloader())
        return

    if args.lr_finder:
        # Run learning rate finder
        lr_finder = trainer.tuner.lr_find(model, data_module, max_lr=0.1)
        lr_finder.results

        # Plot with
        fig = lr_finder.plot(suggest=True)
        new_lr = lr_finder.suggestion()
        trainer.logger.experiment.log({"Lr_FINDER": wandb.Image(fig, caption=f"{new_lr}")}, commit=True)
        model.lr = new_lr

    trainer.fit(model, data_module)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help='YML config file')
    parser.add_argument('--gpus', type=int, nargs='+', help='gpus to use')
    parser.add_argument('--eval', action='store_true', help='run evaluation')
    parser.add_argument('--lr_finder', action='store_true', help='run optimal lr finder')
    parser.add_argument('--wandb_offline', action='store_true', help='Dont push logging to wandb ui')
    args = parser.parse_args()
    main(args)
