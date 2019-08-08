#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import os
import yaml
import sys
import numpy as np

# possible splits
splits = ["train", "valid", "test"]

# possible backends
backends = ["numpy", "torch"]

if __name__ == '__main__':
  parser = argparse.ArgumentParser("./evaluate_semantics.py")
  parser.add_argument(
      '--dataset', '-d',
      type=str,
      required=True,
      help='Dataset dir. No Default',
  )
  parser.add_argument(
      '--predictions', '-p',
      type=str,
      required=None,
      help='Prediction dir. Same organization as dataset, but predictions in'
      'each sequences "prediction" directory. No Default. If no option is set'
      ' we look for the labels in the same directory as dataset'
  )
  parser.add_argument(
      '--split', '-s',
      type=str,
      required=False,
      choices=["train", "valid", "test"],
      default="valid",
      help='Split to evaluate on. One of ' +
      str(splits) + '. Defaults to %(default)s',
  )
  parser.add_argument(
      '--backend', '-b',
      type=str,
      required=False,
      choices= ["numpy", "torch"],
      default="numpy",
      help='Backend for evaluation. One of ' +
      str(backends) + ' Defaults to %(default)s',
  )
  parser.add_argument(
      '--datacfg', '-dc',
      type=str,
      required=False,
      default="config/semantic-kitti.yaml",
      help='Dataset config file. Defaults to %(default)s',
  )
  parser.add_argument(
      '--limit', '-l',
      type=int,
      required=False,
      default=None,
      help='Limit to the first "--limit" points of each scan. Useful for'
      ' evaluating single scan from aggregated pointcloud.'
      ' Defaults to %(default)s',
  )
  parser.add_argument(
      '--codalab',
      dest='codalab',
      type=str,
      default=None,
      help='Exports "scores.txt" to given output directory for codalab'
      'Defaults to %(default)s',
  )

  FLAGS, unparsed = parser.parse_known_args()

  # fill in real predictions dir
  if FLAGS.predictions is None:
    FLAGS.predictions = FLAGS.dataset

  # print summary of what we will do
  print("*" * 80)
  print("INTERFACE:")
  print("Data: ", FLAGS.dataset)
  print("Predictions: ", FLAGS.predictions)
  print("Backend: ", FLAGS.backend)
  print("Split: ", FLAGS.split)
  print("Config: ", FLAGS.datacfg)
  print("Limit: ", FLAGS.limit)
  print("Codalab: ", FLAGS.codalab)
  print("*" * 80)

  # assert split
  assert(FLAGS.split in splits)

  # assert backend
  assert(FLAGS.backend in backends)

  print("Opening data config file %s" % FLAGS.datacfg)
  DATA = yaml.safe_load(open(FLAGS.datacfg, 'r'))

  # get number of interest classes, and the label mappings
  class_strings = DATA["labels"]
  class_remap = DATA["learning_map"]
  class_inv_remap = DATA["learning_map_inv"]
  class_ignore = DATA["learning_ignore"]
  nr_classes = len(class_inv_remap)

  # make lookup table for mapping
  maxkey = max(class_remap.keys())
  
  # +100 hack making lut bigger just in case there are unknown labels
  remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
  remap_lut[list(class_remap.keys())] = list(class_remap.values())
  # print(remap_lut)

  # create evaluator
  ignore = []
  for cl, ign in class_ignore.items():
    if ign:
      x_cl = int(cl)
      ignore.append(x_cl)
      print("Ignoring xentropy class ", x_cl, " in IoU evaluation")

  # create evaluator
  if FLAGS.backend == "torch":
    from auxiliary.torch_ioueval import iouEval
    evaluator = iouEval(nr_classes, ignore)
  elif FLAGS.backend == "numpy":
    from auxiliary.np_ioueval import iouEval
    evaluator = iouEval(nr_classes, ignore)
  else:
    print("Backend for evaluator should be one of ", str(backends))
    quit()
  evaluator.reset()

  # get test set
  test_sequences = DATA["split"][FLAGS.split]

  # get label paths
  label_names = []
  for sequence in test_sequences:
    sequence = '{0:02d}'.format(int(sequence))
    label_paths = os.path.join(FLAGS.dataset, "sequences",
                               str(sequence), "labels")
    # populate the label names
    seq_label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(label_paths)) for f in fn if ".label" in f]
    seq_label_names.sort()
    label_names.extend(seq_label_names)
  # print(label_names)

  # get predictions paths
  pred_names = []
  for sequence in test_sequences:
    sequence = '{0:02d}'.format(int(sequence))
    pred_paths = os.path.join(FLAGS.predictions, "sequences",
                              sequence, "predictions")
    # populate the label names
    seq_pred_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(pred_paths)) for f in fn if ".label" in f]
    seq_pred_names.sort()
    pred_names.extend(seq_pred_names)
  # print(pred_names)

  # check that I have the same number of files
  # print("labels: ", len(label_names))
  # print("predictions: ", len(pred_names))
  assert(len(label_names) == len(pred_names))

  progress = 10
  count = 0
  print("Evaluating sequences: ", end="", flush=True)
  # open each file, get the tensor, and make the iou comparison
  for label_file, pred_file in zip(label_names, pred_names):
    count += 1
    if 100 * count / len(label_names) > progress:
      print("{:d}% ".format(progress), end="", flush=True)
      progress += 10

    # print("evaluating label ", label_file)
    # open label
    label = np.fromfile(label_file, dtype=np.int32)
    label = label.reshape((-1))  # reshape to vector
    label = label & 0xFFFF       # get lower half for semantics
    if FLAGS.limit is not None:
      label = label[:FLAGS.limit]  # limit to desired length
    label = remap_lut[label]       # remap to xentropy format

    # open prediction
    pred = np.fromfile(pred_file, dtype=np.int32)
    pred = pred.reshape((-1))    # reshape to vector
    pred = pred & 0xFFFF         # get lower half for semantics
    if FLAGS.limit is not None:
      pred = pred[:FLAGS.limit]  # limit to desired length
    pred = remap_lut[pred]       # remap to xentropy format

    # add single scan to evaluation
    evaluator.addBatch(pred, label)

  # when I am done, print the evaluation
  m_accuracy = evaluator.getacc()
  m_jaccard, class_jaccard = evaluator.getIoU()

  print('Validation set:\n'
        'Acc avg {m_accuracy:.3f}\n'
        'IoU avg {m_jaccard:.3f}'.format(m_accuracy=m_accuracy,
                                         m_jaccard=m_jaccard))
  # print also classwise
  for i, jacc in enumerate(class_jaccard):
    if i not in ignore:
      print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
          i=i, class_str=class_strings[class_inv_remap[i]], jacc=jacc))

  # print for spreadsheet
  print("*" * 80)
  print("below can be copied straight for paper table")
  for i, jacc in enumerate(class_jaccard):
    if i not in ignore:
      sys.stdout.write('{jacc:.3f}'.format(jacc=jacc.item()))
      sys.stdout.write(",")
  sys.stdout.write('{jacc:.3f}'.format(jacc=m_jaccard.item()))
  sys.stdout.write(",")
  sys.stdout.write('{acc:.3f}'.format(acc=m_accuracy.item()))
  sys.stdout.write('\n')
  sys.stdout.flush()

  # if codalab is necessary, then do it
  if FLAGS.codalab is not None:
    results = {}
    results["accuracy_mean"] = float(m_accuracy)
    results["iou_mean"] = float(m_jaccard)
    for i, jacc in enumerate(class_jaccard):
      if i not in ignore:
        results["iou_"+class_strings[class_inv_remap[i]]] = float(jacc)
    # save to file
    output_filename = os.path.join(FLAGS.codalab, 'scores.txt')
    with open(output_filename, 'w') as yaml_file:
      yaml.dump(results, yaml_file, default_flow_style=False)
