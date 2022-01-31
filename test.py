# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 13:25:02 2021

@author: Luca Medeiros
"""
import glob
import yaml
import torch
import wandb
import random
import json
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from sklearn.metrics import f1_score

from model import EmbedderWrapper
from utils import ImageInstance
from tqdm import tqdm


def read_config(config):
    with open(config, 'r') as stream:
        cfg_dict = yaml.safe_load(stream)
    cfg = Namespace(**cfg_dict)

    return cfg


def make_dataloader(cfg, test_path):
    test = ImageInstance(test_path,
                         transform=None)

    print('Testset length: ', len(test))
    return torch.utils.data.DataLoader(test,
                                       batch_size=1,
                                       shuffle=True,
                                       num_workers=16)


def calculate_porcentage(counter):
    topk = defaultdict(list)
    trans = dict()
    for target, preds in counter.items():
        total = sum(preds.values())
        top1 = max(preds, key=preds.get)
        topk[target].append([top1, preds[top1]/total])
        trans[top1] = target
        # for pred, count in preds.items():
        #     porc = count/total
        #     if [pred, porc] in topk[target]:
        #         continue
        #     if porc >= 0.15:
        #         topk[target].append([pred, porc])

    return topk, trans


def inference(model, dataloader, name, type_, constraint=None):
    const_flag = 'not_constraint'
    classes = dataloader.dataset.classes
    path = dataloader.dataset.path
    classes_in = classes
    table = wandb.Table(columns=['img', 'target', 'pred', 'score', 'basename'])
    counter = defaultdict(lambda: defaultdict(int))
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            if batch_idx > 1000:
                break
            inputs, targets, indexes, filenames, class_names = batch
            # classes_in = random.sample(classes, 8)
            # classes_in.append(class_names[0])
            results = model(inputs.cuda(), classes_in)
            targets = [classes[k] for k in targets]
            for basename, img, target, pred, score in zip(filenames,
                                                          inputs,
                                                          targets,
                                                          results['preds'].cpu(),
                                                          results['scores'].cpu()):
                pred_class = classes_in[pred]
                # if score < 0.3:
                #     pred_class = 'unknown'
                counter[target][pred_class] += 1
                table.add_data(wandb.Image(img),
                               target,
                               pred_class,
                               float((score * 100)),
                               str(basename))

    topk, trans = calculate_porcentage(counter)
    with open(path + 'topk.json', 'w') as f:
        json.dump(dict(topk), f, ensure_ascii=False, indent=2)
    with open(path + 'trans.json', 'w') as f:
        json.dump(dict(trans), f, ensure_ascii=False, indent=2)
    y_true = [row[1] for row in table.data]
    y_pred = [row[2] for row in table.data]
    if not model.constraint:
        y_pred = [trans[k] if k in trans else k for k in y_pred]
    f1 = f1_score(y_true, y_pred, average='macro')
    wandb.log({f'{name}/{const_flag}_{type_}_f1': f1}, commit=False)
    wandb.log({f'{name}/{const_flag}_{type_}': table}, commit=True)


def main(args):
    cfg = read_config(args.config)
    project, instance = cfg.instance.split('/')
    wandb.init(project=project, name=instance, job_type='eval')

    if cfg.resume == '':
        raise 'Resume model not set'

    model = EmbedderWrapper.load_from_checkpoint(cfg.resume)
    model.eval()
    model.cuda()
    folders = glob.glob(args.data_path + '/*/')
    for path in folders:
        name, type_, _ = path.split('/')[-3:]
        test_dataloader = make_dataloader(cfg, path)
        inference(model, test_dataloader, name, type_)

    return
    classes = test_dataloader.dataset.classes
    table = wandb.Table(columns=['img', 'target', 'pred', 'score', 'basename'])
    counter = defaultdict(lambda: defaultdict(int))
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_dataloader)):
            inputs, targets, indexes, filenames = batch
            results = model(inputs.cuda())
            targets = [classes[k] for k in targets]
            for basename, img, target, pred, score in zip(filenames,
                                                          inputs,
                                                          targets,
                                                          results['preds'].cpu(),
                                                          results['scores'].cpu()):
                pred_class = model.classes[pred]
                counter[target][pred_class] += 1
                table.add_data(wandb.Image(img,
                                           caption=f'GT: {target} | pred: {pred_class}'),
                               target,
                               pred_class,
                               float((score * 100)),
                               str(basename))

        wandb.log({'val_table': table}, commit=True)

    topk, trans = calculate_porcentage(counter)
    constrain_classes = list()
    for k, v in topk.items():
        for class_ in v:
            constrain_classes.append(class_[0])

    constrain_classes_unique = list(set(constrain_classes))
    # constrain_classes_unique = ['쌀밥', '보리밥', '시루떡', '어묵볶음', '미소된장국', '호박죽', '된장찌개', '수수팥떡', '홍어무침', '오징어국', '현미밥', '고등어조림', '깻잎나물볶음', '시금치나물', '배추김치']
    model.class_constraint(constrain_classes_unique)
    counter2 = defaultdict(lambda: defaultdict(int))
    tableconst = wandb.Table(columns=['img', 'target', 'pred', 'pred_trans', 'score', 'basename'])
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_dataloader)):
            inputs, targets, indexes, filenames = batch
            results = model(inputs.cuda())
            targets = [classes[k] for k in targets]
            for basename, img, target, pred, score in zip(filenames,
                                                          inputs,
                                                          targets,
                                                          results['preds'].cpu(),
                                                          results['scores'].cpu()):
                pred_class = model.const_classes[pred]
                pred_class_trans = trans[pred_class]
                counter2[target][pred_class] += 1
                tableconst.add_data(wandb.Image(img,
                                           caption=f'GT: {target} | pred: {pred_class}'),
                                    target,
                                    pred_class,
                                    pred_class_trans,
                                    float((score * 100)),
                                    str(basename))

        wandb.log({'val_table_constraint_max': tableconst}, commit=True)

    topk2, trans = calculate_porcentage(counter2)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help='YML config file')
    parser.add_argument('--data_path', type=str, help='path to input data')
    args = parser.parse_args()
    main(args)
