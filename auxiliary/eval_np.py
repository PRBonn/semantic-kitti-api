#!/usr/bin/env python3

# This file is covered by the LICENSE file in the root of this project.

import numpy as np
import time


class PanopticEval:
  """ Panoptic evaluation using numpy
  
  authors: Andres Milioto and Jens Behley

  """

  def __init__(self, n_classes, device=None, ignore=None, offset=2**32, min_points=30):
    self.n_classes = n_classes
    assert (device == None)
    self.ignore = np.array(ignore, dtype=np.int64)
    self.include = np.array([n for n in range(self.n_classes) if n not in self.ignore], dtype=np.int64)

    print("[PANOPTIC EVAL] IGNORE: ", self.ignore)
    print("[PANOPTIC EVAL] INCLUDE: ", self.include)

    self.reset()
    self.offset = offset  # largest number of instances in a given scan
    self.min_points = min_points  # smallest number of points to consider instances in gt
    self.eps = 1e-15

  def num_classes(self):
    return self.n_classes

  def reset(self):
    # general things
    # iou stuff
    self.px_iou_conf_matrix = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)
    # panoptic stuff
    self.pan_tp = np.zeros(self.n_classes, dtype=np.int64)
    self.pan_iou = np.zeros(self.n_classes, dtype=np.double)
    self.pan_fp = np.zeros(self.n_classes, dtype=np.int64)
    self.pan_fn = np.zeros(self.n_classes, dtype=np.int64)

  ################################# IoU STUFF ##################################
  def addBatchSemIoU(self, x_sem, y_sem):
    # idxs are labels and predictions
    idxs = np.stack([x_sem, y_sem], axis=0)

    # make confusion matrix (cols = gt, rows = pred)
    np.add.at(self.px_iou_conf_matrix, tuple(idxs), 1)

  def getSemIoUStats(self):
    # clone to avoid modifying the real deal
    conf = self.px_iou_conf_matrix.copy().astype(np.double)
    # remove fp from confusion on the ignore classes predictions
    # points that were predicted of another class, but were ignore
    # (corresponds to zeroing the cols of those classes, since the predictions
    # go on the rows)
    conf[:, self.ignore] = 0

    # get the clean stats
    tp = conf.diagonal()
    fp = conf.sum(axis=1) - tp
    fn = conf.sum(axis=0) - tp
    return tp, fp, fn

  def getSemIoU(self):
    tp, fp, fn = self.getSemIoUStats()
    # print(f"tp={tp}")
    # print(f"fp={fp}")
    # print(f"fn={fn}")
    intersection = tp
    union = tp + fp + fn
    union = np.maximum(union, self.eps)
    iou = intersection.astype(np.double) / union.astype(np.double)
    iou_mean = (intersection[self.include].astype(np.double) / union[self.include].astype(np.double)).mean()

    return iou_mean, iou  # returns "iou mean", "iou per class" ALL CLASSES

  def getSemAcc(self):
    tp, fp, fn = self.getSemIoUStats()
    total_tp = tp.sum()
    total = tp[self.include].sum() + fp[self.include].sum()
    total = np.maximum(total, self.eps)
    acc_mean = total_tp.astype(np.double) / total.astype(np.double)

    return acc_mean  # returns "acc mean"

  ################################# IoU STUFF ##################################
  ##############################################################################

  #############################  Panoptic STUFF ################################
  def addBatchPanoptic(self, x_sem_row, x_inst_row, y_sem_row, y_inst_row):
    # make sure instances are not zeros (it messes with my approach)
    x_inst_row = x_inst_row + 1
    y_inst_row = y_inst_row + 1

    # only interested in points that are outside the void area (not in excluded classes)
    for cl in self.ignore:
      # make a mask for this class
      gt_not_in_excl_mask = y_sem_row != cl
      # remove all other points
      x_sem_row = x_sem_row[gt_not_in_excl_mask]
      y_sem_row = y_sem_row[gt_not_in_excl_mask]
      x_inst_row = x_inst_row[gt_not_in_excl_mask]
      y_inst_row = y_inst_row[gt_not_in_excl_mask]

    # first step is to count intersections > 0.5 IoU for each class (except the ignored ones)
    for cl in self.include:
      # print("*"*80)
      # print("CLASS", cl.item())
      # get a class mask
      x_inst_in_cl_mask = x_sem_row == cl
      y_inst_in_cl_mask = y_sem_row == cl

      # get instance points in class (makes outside stuff 0)
      x_inst_in_cl = x_inst_row * x_inst_in_cl_mask.astype(np.int64)
      y_inst_in_cl = y_inst_row * y_inst_in_cl_mask.astype(np.int64)

      # generate the areas for each unique instance prediction
      unique_pred, counts_pred = np.unique(x_inst_in_cl[x_inst_in_cl > 0], return_counts=True)
      id2idx_pred = {id: idx for idx, id in enumerate(unique_pred)}
      matched_pred = np.array([False] * unique_pred.shape[0])
      # print("Unique predictions:", unique_pred)

      # generate the areas for each unique instance gt_np
      unique_gt, counts_gt = np.unique(y_inst_in_cl[y_inst_in_cl > 0], return_counts=True)
      id2idx_gt = {id: idx for idx, id in enumerate(unique_gt)}
      matched_gt = np.array([False] * unique_gt.shape[0])
      # print("Unique ground truth:", unique_gt)

      # generate intersection using offset
      valid_combos = np.logical_and(x_inst_in_cl > 0, y_inst_in_cl > 0)
      offset_combo = x_inst_in_cl[valid_combos] + self.offset * y_inst_in_cl[valid_combos]
      unique_combo, counts_combo = np.unique(offset_combo, return_counts=True)

      # generate an intersection map
      # count the intersections with over 0.5 IoU as TP
      gt_labels = unique_combo // self.offset
      pred_labels = unique_combo % self.offset
      gt_areas = np.array([counts_gt[id2idx_gt[id]] for id in gt_labels])
      pred_areas = np.array([counts_pred[id2idx_pred[id]] for id in pred_labels])
      intersections = counts_combo
      unions = gt_areas + pred_areas - intersections
      ious = intersections.astype(float) / unions.astype(float)


      tp_indexes = ious > 0.5
      self.pan_tp[cl] += np.sum(tp_indexes)
      self.pan_iou[cl] += np.sum(ious[tp_indexes])

      matched_gt[[id2idx_gt[id] for id in gt_labels[tp_indexes]]] = True
      matched_pred[[id2idx_pred[id] for id in pred_labels[tp_indexes]]] = True

      # count the FN
      self.pan_fn[cl] += np.sum(np.logical_and(counts_gt >= self.min_points, matched_gt == False))

      # count the FP
      self.pan_fp[cl] += np.sum(np.logical_and(counts_pred >= self.min_points, matched_pred == False))

  def getPQ(self):
    # first calculate for all classes
    sq_all = self.pan_iou.astype(np.double) / np.maximum(self.pan_tp.astype(np.double), self.eps)
    rq_all = self.pan_tp.astype(np.double) / np.maximum(
        self.pan_tp.astype(np.double) + 0.5 * self.pan_fp.astype(np.double) + 0.5 * self.pan_fn.astype(np.double),
        self.eps)
    pq_all = sq_all * rq_all

    # then do the REAL mean (no ignored classes)
    SQ = sq_all[self.include].mean()
    RQ = rq_all[self.include].mean()
    PQ = pq_all[self.include].mean()

    return PQ, SQ, RQ, pq_all, sq_all, rq_all

  #############################  Panoptic STUFF ################################
  ##############################################################################

  def addBatch(self, x_sem, x_inst, y_sem, y_inst):  # x=preds, y=targets
    ''' IMPORTANT: Inputs must be batched. Either [N,H,W], or [N, P]
    '''
    # add to IoU calculation (for checking purposes)
    self.addBatchSemIoU(x_sem, y_sem)

    # now do the panoptic stuff
    self.addBatchPanoptic(x_sem, x_inst, y_sem, y_inst)


if __name__ == "__main__":
  # generate problem from He paper (https://arxiv.org/pdf/1801.00868.pdf)
  classes = 5  # ignore, grass, sky, person, dog
  cl_strings = ["ignore", "grass", "sky", "person", "dog"]
  ignore = [0]  # only ignore ignore class
  min_points = 1  # for this example we care about all points

  # generate ground truth and prediction
  sem_pred = []
  inst_pred = []
  sem_gt = []
  inst_gt = []

  # some ignore stuff
  N_ignore = 50
  sem_pred.extend([0 for i in range(N_ignore)])
  inst_pred.extend([0 for i in range(N_ignore)])
  sem_gt.extend([0 for i in range(N_ignore)])
  inst_gt.extend([0 for i in range(N_ignore)])

  # grass segment
  N_grass = 50
  N_grass_pred = 40  # rest is sky
  sem_pred.extend([1 for i in range(N_grass_pred)])  # grass
  sem_pred.extend([2 for i in range(N_grass - N_grass_pred)])  # sky
  inst_pred.extend([0 for i in range(N_grass)])
  sem_gt.extend([1 for i in range(N_grass)])  # grass
  inst_gt.extend([0 for i in range(N_grass)])

  # sky segment
  N_sky = 50
  N_sky_pred = 40  # rest is grass
  sem_pred.extend([2 for i in range(N_sky_pred)])  # sky
  sem_pred.extend([1 for i in range(N_sky - N_sky_pred)])  # grass
  inst_pred.extend([0 for i in range(N_sky)])  # first instance
  sem_gt.extend([2 for i in range(N_sky)])  # sky
  inst_gt.extend([0 for i in range(N_sky)])  # first instance

  # wrong dog as person prediction
  N_dog = 50
  N_person = N_dog
  sem_pred.extend([3 for i in range(N_person)])
  inst_pred.extend([35 for i in range(N_person)])
  sem_gt.extend([4 for i in range(N_dog)])
  inst_gt.extend([22 for i in range(N_dog)])

  # two persons in prediction, but three in gt
  N_person = 50
  sem_pred.extend([3 for i in range(6 * N_person)])
  inst_pred.extend([8 for i in range(4 * N_person)])
  inst_pred.extend([95 for i in range(2 * N_person)])
  sem_gt.extend([3 for i in range(6 * N_person)])
  inst_gt.extend([33 for i in range(3 * N_person)])
  inst_gt.extend([42 for i in range(N_person)])
  inst_gt.extend([11 for i in range(2 * N_person)])

  # gt and pred to numpy
  sem_pred = np.array(sem_pred, dtype=np.int64).reshape(1, -1)
  inst_pred = np.array(inst_pred, dtype=np.int64).reshape(1, -1)
  sem_gt = np.array(sem_gt, dtype=np.int64).reshape(1, -1)
  inst_gt = np.array(inst_gt, dtype=np.int64).reshape(1, -1)

  # evaluator
  evaluator = PanopticEval(classes, ignore=ignore, min_points=1)
  evaluator.addBatch(sem_pred, inst_pred, sem_gt, inst_gt)
  pq, sq, rq, all_pq, all_sq, all_rq = evaluator.getPQ()
  iou, all_iou = evaluator.getSemIoU()

  # [PANOPTIC EVAL] IGNORE:  [0]
  # [PANOPTIC EVAL] INCLUDE:  [1 2 3 4]
  # TOTALS
  # PQ: 0.47916666666666663
  # SQ: 0.5520833333333333
  # RQ: 0.6666666666666666
  # IoU: 0.5476190476190476
  # Class ignore 	 PQ: 0.0 SQ: 0.0 RQ: 0.0 IoU: 0.0
  # Class grass 	 PQ: 0.6666666666666666 SQ: 0.6666666666666666 RQ: 1.0 IoU: 0.6666666666666666
  # Class sky 	 PQ: 0.6666666666666666 SQ: 0.6666666666666666 RQ: 1.0 IoU: 0.6666666666666666
  # Class person 	 PQ: 0.5833333333333333 SQ: 0.875 RQ: 0.6666666666666666 IoU: 0.8571428571428571
  # Class dog 	 PQ: 0.0 SQ: 0.0 RQ: 0.0 IoU: 0.0

  print("TOTALS")
  print("PQ:", pq.item(), pq.item() == 0.47916666666666663)
  print("SQ:", sq.item(), sq.item() == 0.5520833333333333)
  print("RQ:", rq.item(), rq.item() == 0.6666666666666666)
  print("IoU:", iou.item(), iou.item() == 0.5476190476190476)
  for i, (pq, sq, rq, iou) in enumerate(zip(all_pq, all_sq, all_rq, all_iou)):
    print("Class", cl_strings[i], "\t", "PQ:", pq.item(), "SQ:", sq.item(), "RQ:", rq.item(), "IoU:", iou.item())
