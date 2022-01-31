#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 14:22:24 2021


@author: Luca Medeiros, lucamedeiros@outlook.com
"""

import torch
import timm
import numpy as np
import pytorch_lightning as pl
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch_optimizer as optim_jet
import torchvision.transforms as transforms

from kobart.model_kobart import Kobart


class ZSLModel(pl.LightningModule):
    def __init__(self, args, classes, val_classes, ndata):
        super().__init__()
        print('==> Building model..')
        self.classes = classes
        self.val_classes = val_classes
        self.logit_scale = args.logit_scale
        self.args = args
        self.head = None
        self.lemniscate = None
        self.lr = args.lr
        self.ndata = ndata
        self.normalizer = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                               (0.2023, 0.1994, 0.2010))
        self.t = transforms.Compose([
                        transforms.Resize(224),             # resize shortest side to 224 pixels
                        transforms.CenterCrop(224),         # crop longest side to 224 pixels at center,
            ])
        self.module = self.create_module()
        self.text_encoder = Kobart(args, self.classes)
        self.img_criterion = self.create_criterion()
        self.txt_criterion = self.create_criterion()
        self.save_hyperparameters()

    def create_module(self):
        """
        Generate the backbone embedding module.

        """
        if self.args.module == 'regnet':
            module = timm.create_model('regnety_040', pretrained=True)
            num_ftrs = module.head.fc.in_features
            module.head.fc = nn.Linear(num_ftrs, self.args.low_dim)
        elif self.args.module == 'deit':
            module = timm.create_model('deit_base_patch16_224', pretrained=True)
            num_ftrs = module.head.in_features
            module.head = nn.Linear(num_ftrs, self.args.low_dim)
        elif self.args.module == 'pretrained':
            print('** Pretrained module')
            module = torch.load('./models/image_encoder_b3.pth')
            # Freeze all the layers
            for param in module.parameters():
                param.requires_grad = False
            # Generate unfreezed classifier
            num_ftrs = module.classifier.in_features
            module.classifier = nn.Linear(num_ftrs, self.args.low_dim)
            return module
        else:
            module = timm.create_model('efficientnet_b3a', pretrained=True)
            num_ftrs = module.classifier.in_features
            module.classifier = nn.Linear(num_ftrs, self.args.low_dim)
        
        return module

    def create_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def get_parameters(self):
        params = [{'params': self.text_encoder.parameters()}, {'params': self.module.parameters()}]
        return params

    def configure_optimizers(self):
        pars = self.get_parameters()
        if self.args.opt == 'sgd':
            optimizer = optim.SGD(pars, lr=self.lr, momentum=0.9, weight_decay=5e-4)
        elif self.args.opt == 'adamp':
            optimizer = optim_jet.AdamP(pars, lr=self.lr, betas=(0.9, 0.999), weight_decay=1e-2)
        elif self.args.opt == 'sgdp':
            optimizer = optim_jet.SGDP(pars, lr=self.lr, momentum=0.9, weight_decay=1e-5, nesterov=True)
        elif self.args.opt == 'madgrad':
            optimizer = optim_jet.MADGRAD(pars, lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.1)
        return {"optimizer": optimizer,
                "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        inputs, targets, indexes, filenames, class_names = batch
        # print(targets, class_names)
        # print(len(targets.cpu().tolist()))
        # if len(targets.cpu().tolist())!=len(list(set(targets.cpu().tolist()))):
        #     print(targets.cpu().tolist())
        results = self(inputs, class_names)
        ground_truth = torch.arange(results['outputs'][0].size(0), dtype=torch.long).type_as(targets)
        img_loss = self.img_criterion(results['outputs'][0], ground_truth)
        txt_loss = self.txt_criterion(results['outputs'][1], ground_truth)
        total_loss = (img_loss + txt_loss)/2
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, logger=True)
        
        acc_i = (torch.argmax(results['outputs'][0], 1) == ground_truth).sum()
        acc_t = (torch.argmax(results['outputs'][0], 0) == ground_truth).sum()

        self.log('train_acc_i', acc_i / results['outputs'][0].size(0),
                 on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.log('train_acc_t', acc_t / results['outputs'][0].size(0),
                 on_step=True, on_epoch=True, logger=True, prog_bar=True)

        return {'loss': total_loss, 'gt': ground_truth, **results}

    def training_step_end(self, outs):
        outs['loss'] = outs['loss'].mean()
        return outs

    def validation_step_end(self, outs):
        return outs

    def validation_step(self, batch, batch_idx):
        inputs, targets, indexes, filenames, class_names = batch
        results = self(inputs, self.val_classes)
        ground_truth = torch.arange(inputs.size(0), dtype=torch.long).type_as(targets)
        print(ground_truth)
        # loss = F.cross_entropy(results['outputs'][0], ground_truth)
        # self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)

        return results

    def forward(self, x, targets) -> dict:
        """
        Parameters
        ----------
        x : tensor
            samples.
        targets : list/tuple
            labels texts, needed for text encoder.
        """
        image_features = self.module(x)
        text_features = self.text_encoder(targets)
        
        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_image = self.logit_scale * image_features @ text_features.t()
        logits_per_text = self.logit_scale * text_features @ image_features.t()

        scores_img, preds_img_logits = torch.max(F.softmax(logits_per_image, 1), 1)

        return {
                'embeds': [image_features, text_features],
                'outputs': [logits_per_image, logits_per_text],
                'preds': preds_img_logits,
                'scores': scores_img,
                }

    def preprocess(self, sample):
        sample = transforms.ToPILImage()(sample)
        sample = self.t(sample)
        sample = np.array(sample, dtype=np.uint8)
        sample = self.normalizer(self.ToTensor(sample))

        return sample.float()

    def predict(self, imgs, texts=None, pred_classes=True):
        '''
        imgs should be RGB
        texts list
        '''
        if texts is None:
            texts = self.classes
        data = torch.stack([self.preprocess(k) for k in imgs])
        with torch.no_grad():
            output = self(data.to(self.device), texts.to(self.device))
        if pred_classes:
            output['pred_classes'] = [texts[k] for k in output['preds'].cpu().numpy().flatten()]
        return output
