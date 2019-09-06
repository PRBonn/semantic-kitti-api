#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import sys
import torch
import numpy as np


class iouEval:
  def __init__(self, n_classes, ignore=None):
    # classes
    self.n_classes = n_classes

    # What to include and ignore from the means
    self.ignore = torch.tensor(ignore).long()
    self.include = torch.tensor(
        [n for n in range(self.n_classes) if n not in self.ignore]).long()
    print("[IOU EVAL] IGNORE: ", self.ignore)
    print("[IOU EVAL] INCLUDE: ", self.include)

    # get device
    self.device = torch.device('cpu')
    if torch.cuda.is_available():
      self.device = torch.device('cuda')

    # reset the class counters
    self.reset()

  def num_classes(self):
    return self.n_classes

  def reset(self):
    self.conf_matrix = torch.zeros(
        (self.n_classes, self.n_classes), device=self.device).long()

  def addBatch(self, x, y):  # x=preds, y=targets
    # to tensor
    x_row = torch.from_numpy(x).to(self.device).long()
    y_row = torch.from_numpy(y).to(self.device).long()

    # sizes should be matching
    x_row = x_row.reshape(-1)  # de-batchify
    y_row = y_row.reshape(-1)  # de-batchify

    # check
    assert(x_row.shape == x_row.shape)

    # idxs are labels and predictions
    idxs = torch.stack([x_row, y_row], dim=0)

    # ones is what I want to add to conf when I
    ones = torch.ones((idxs.shape[-1]), device=self.device).long()

    # make confusion matrix (cols = gt, rows = pred)
    self.conf_matrix = self.conf_matrix.index_put_(
        tuple(idxs), ones, accumulate=True)

  def getStats(self):
    # remove fp from confusion on the ignore classes cols
    conf = self.conf_matrix.clone().double()
    conf[:, self.ignore] = 0

    # get the clean stats
    tp = conf.diag()
    fp = conf.sum(dim=1) - tp
    fn = conf.sum(dim=0) - tp
    return tp, fp, fn

  def getIoU(self):
    tp, fp, fn = self.getStats()
    intersection = tp
    union = tp + fp + fn + 1e-15
    iou = intersection / union
    iou_mean = (intersection[self.include] / union[self.include]).mean()
    return iou_mean, iou  # returns "iou mean", "iou per class" ALL CLASSES

  def getacc(self):
    tp, fp, fn = self.getStats()
    total_tp = tp.sum()
    total = tp[self.include].sum() + fp[self.include].sum() + 1e-15
    acc_mean = total_tp / total
    return acc_mean  # returns "acc mean"


if __name__ == "__main__":
  # mock problem
  nclasses = 2
  ignore = []

  # test with 2 squares and a known IOU
  lbl = np.zeros((7, 7), dtype=np.int64)
  argmax = np.zeros((7, 7), dtype=np.int64)

  # put squares
  lbl[2:4, 2:4] = 1
  argmax[3:5, 3:5] = 1

  # make evaluator
  eval = iouEval(nclasses, ignore)

  # run
  eval.addBatch(argmax, lbl)
  m_iou, iou = eval.getIoU()
  print("IoU: ", m_iou)
  print("IoU class: ", iou)
  m_acc = eval.getacc()
  print("Acc: ", m_acc)
