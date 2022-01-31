#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 14:07:08 2021

@author: nuvilabs
"""


import sys


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def report(epoch, batch_idx, batch_time, data_time, loss, acc, loader, type_='train'):
    if batch_idx % 10 == 0:
        print('\n'+'Epoch {type_}: [{}][{}/{}]'
              'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
              'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
              'Acc: {acc.val:.4f} '
              'Loss: {loss.val:.4f} ({loss.avg:.4f})'.format(
                  epoch, batch_idx, len(loader), batch_time=batch_time, data_time=data_time, acc=acc, loss=loss, type_=type_))
    else:
        print('\r'+'Epoch {type_}: [{}][{}/{}]'
              'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
              'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
              'Acc: {acc.val:.4f} '
              'Loss: {loss.val:.4f} ({loss.avg:.4f})'.format(
                  epoch, batch_idx, len(loader), batch_time=batch_time, data_time=data_time, acc=acc, loss=loss, type_=type_), end="")
        sys.stdout.flush()
        