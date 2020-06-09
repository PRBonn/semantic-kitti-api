#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import os
import yaml
import sys
import numpy as np
import time
import json

from auxiliary.eval_np import PanopticEval

# possible splits
splits = ["train", "valid", "test"]

if __name__ == '__main__':
  parser = argparse.ArgumentParser("./evaluate_panoptic.py")
  parser.add_argument(
      '--dataset',
      '-d',
      type=str,
      required=True,
      help='Dataset dir. No Default',
  )
  parser.add_argument(
      '--predictions',
      '-p',
      type=str,
      required=None,
      help='Prediction dir. Same organization as dataset, but predictions in'
      'each sequences "prediction" directory. No Default. If no option is set'
      ' we look for the labels in the same directory as dataset')
  parser.add_argument(
      '--split',
      '-s',
      type=str,
      required=False,
      choices=["train", "valid", "test"],
      default="valid",
      help='Split to evaluate on. One of ' + str(splits) + '. Defaults to %(default)s',
  )
  parser.add_argument(
      '--data_cfg',
      '-dc',
      type=str,
      required=False,
      default="config/semantic-kitti.yaml",
      help='Dataset config file. Defaults to %(default)s',
  )
  parser.add_argument(
      '--limit',
      '-l',
      type=int,
      required=False,
      default=None,
      help='Limit to the first "--limit" points of each scan. Useful for'
      ' evaluating single scan from aggregated pointcloud.'
      ' Defaults to %(default)s',
  )
  parser.add_argument(
      '--min_inst_points',
      type=int,
      required=False,
      default=50,
      help='Lower bound for the number of points to be considered instance',
  )
  parser.add_argument(
      '--output',
      type=str,
      required=False,
      default=None,
      help='Output directory for scores.txt and detailed_results.html.',
  )

  start_time = time.time()

  FLAGS, unparsed = parser.parse_known_args()

  # fill in real predictions dir
  if FLAGS.predictions is None:
    FLAGS.predictions = FLAGS.dataset

  # print summary of what we will do
  print("*" * 80)
  print("INTERFACE:")
  print("Data: ", FLAGS.dataset)
  print("Predictions: ", FLAGS.predictions)
  print("Split: ", FLAGS.split)
  print("Config: ", FLAGS.data_cfg)
  print("Limit: ", FLAGS.limit)
  print("Min instance points: ", FLAGS.min_inst_points)
  print("Output directory", FLAGS.output)
  print("*" * 80)

  # assert split
  assert (FLAGS.split in splits)

  # open data config file
  DATA = yaml.safe_load(open(FLAGS.data_cfg, 'r'))

  # get number of interest classes, and the label mappings
  # class
  class_remap = DATA["learning_map"]
  class_inv_remap = DATA["learning_map_inv"]
  class_ignore = DATA["learning_ignore"]
  nr_classes = len(class_inv_remap)
  class_strings = DATA["labels"]

  # make lookup table for mapping
  # class
  maxkey = max(class_remap.keys())

  # +100 hack making lut bigger just in case there are unknown labels
  class_lut = np.zeros((maxkey + 100), dtype=np.int32)
  class_lut[list(class_remap.keys())] = list(class_remap.values())

  # class
  ignore_class = [cl for cl, ignored in class_ignore.items() if ignored]

  print("Ignoring classes: ", ignore_class)

  # create evaluator
  class_evaluator = PanopticEval(nr_classes, None, ignore_class, min_points=FLAGS.min_inst_points)

  # get test set
  test_sequences = DATA["split"][FLAGS.split]

  # get label paths
  label_names = []
  for sequence in test_sequences:
    sequence = '{0:02d}'.format(int(sequence))
    label_paths = os.path.join(FLAGS.dataset, "sequences", sequence, "labels")
    # populate the label names
    seq_label_names = sorted([os.path.join(label_paths, fn) for fn in os.listdir(label_paths) if fn.endswith(".label")])
    label_names.extend(seq_label_names)
  # print(label_names)

  # get predictions paths
  pred_names = []
  for sequence in test_sequences:
    sequence = '{0:02d}'.format(int(sequence))
    pred_paths = os.path.join(FLAGS.predictions, "sequences", sequence, "predictions")
    # populate the label names
    seq_pred_names = sorted([os.path.join(pred_paths, fn) for fn in os.listdir(pred_paths) if fn.endswith(".label")])
    pred_names.extend(seq_pred_names)
  # print(pred_names)

  # check that I have the same number of files
  assert (len(label_names) == len(pred_names))

  print("Evaluating sequences: ", end="", flush=True)
  # open each file, get the tensor, and make the iou comparison

  complete = len(label_names)
  count = 0
  percent = 10
  for label_file, pred_file in zip(label_names, pred_names):
    count = count + 1
    if 100 * count / complete > percent:
      print("{}% ".format(percent), end="", flush=True)
      percent = percent + 10
    # print("evaluating label ", label_file, "with", pred_file)
    # open label

    label = np.fromfile(label_file, dtype=np.uint32)

    u_label_sem_class = class_lut[label & 0xFFFF]  # remap to xentropy format
    u_label_inst = label # unique instance ids.
    if FLAGS.limit is not None:
      u_label_sem_class = u_label_sem_class[:FLAGS.limit]
      u_label_sem_cat = u_label_sem_cat[:FLAGS.limit]
      u_label_inst = u_label_inst[:FLAGS.limit]

    label = np.fromfile(pred_file, dtype=np.uint32)

    u_pred_sem_class = class_lut[label & 0xFFFF]  # remap to xentropy format
    u_pred_inst = label # unique instance ids.
    if FLAGS.limit is not None:
      u_pred_sem_class = u_pred_sem_class[:FLAGS.limit]
      u_pred_sem_cat = u_pred_sem_cat[:FLAGS.limit]
      u_pred_inst = u_pred_inst[:FLAGS.limit]

    class_evaluator.addBatch(u_pred_sem_class, u_pred_inst, u_label_sem_class, u_label_inst)

  print("100%")

  complete_time = time.time() - start_time

  # when I am done, print the evaluation
  class_PQ, class_SQ, class_RQ, class_all_PQ, class_all_SQ, class_all_RQ = class_evaluator.getPQ()
  class_IoU, class_all_IoU = class_evaluator.getSemIoU()

  # now make a nice dictionary
  output_dict = {}

  # make python variables
  class_PQ = class_PQ.item()
  class_SQ = class_SQ.item()
  class_RQ = class_RQ.item()
  class_all_PQ = class_all_PQ.flatten().tolist()
  class_all_SQ = class_all_SQ.flatten().tolist()
  class_all_RQ = class_all_RQ.flatten().tolist()
  class_IoU = class_IoU.item()
  class_all_IoU = class_all_IoU.flatten().tolist()

  # fill in with the raw values
  # output_dict["raw"] = {}
  # output_dict["raw"]["class_PQ"] = class_PQ
  # output_dict["raw"]["class_SQ"] = class_SQ
  # output_dict["raw"]["class_RQ"] = class_RQ
  # output_dict["raw"]["class_all_PQ"] = class_all_PQ
  # output_dict["raw"]["class_all_SQ"] = class_all_SQ
  # output_dict["raw"]["class_all_RQ"] = class_all_RQ
  # output_dict["raw"]["class_IoU"] = class_IoU
  # output_dict["raw"]["class_all_IoU"] = class_all_IoU

  things = ['car', 'truck', 'bicycle', 'motorcycle', 'other-vehicle', 'person', 'bicyclist', 'motorcyclist']
  stuff = [
      'road', 'sidewalk', 'parking', 'other-ground', 'building', 'vegetation', 'trunk', 'terrain', 'fence', 'pole',
      'traffic-sign'
  ]
  all_classes = things + stuff

  # class

  output_dict["all"] = {}
  output_dict["all"]["PQ"] = class_PQ
  output_dict["all"]["SQ"] = class_SQ
  output_dict["all"]["RQ"] = class_RQ
  output_dict["all"]["IoU"] = class_IoU

  classwise_tables = {}

  for idx, (pq, rq, sq, iou) in enumerate(zip(class_all_PQ, class_all_RQ, class_all_SQ, class_all_IoU)):
    class_str = class_strings[class_inv_remap[idx]]
    output_dict[class_str] = {}
    output_dict[class_str]["PQ"] = pq
    output_dict[class_str]["SQ"] = sq
    output_dict[class_str]["RQ"] = rq
    output_dict[class_str]["IoU"] = iou

  PQ_all = np.mean([float(output_dict[c]["PQ"]) for c in all_classes])
  PQ_dagger = np.mean([float(output_dict[c]["PQ"]) for c in things] + [float(output_dict[c]["IoU"]) for c in stuff])
  RQ_all = np.mean([float(output_dict[c]["RQ"]) for c in all_classes])
  SQ_all = np.mean([float(output_dict[c]["SQ"]) for c in all_classes])

  PQ_things = np.mean([float(output_dict[c]["PQ"]) for c in things])
  RQ_things = np.mean([float(output_dict[c]["RQ"]) for c in things])
  SQ_things = np.mean([float(output_dict[c]["SQ"]) for c in things])

  PQ_stuff = np.mean([float(output_dict[c]["PQ"]) for c in stuff])
  RQ_stuff = np.mean([float(output_dict[c]["RQ"]) for c in stuff])
  SQ_stuff = np.mean([float(output_dict[c]["SQ"]) for c in stuff])
  mIoU = output_dict["all"]["IoU"]

  codalab_output = {}
  codalab_output["pq_mean"] = float(PQ_all)
  codalab_output["pq_dagger"] = float(PQ_dagger)
  codalab_output["sq_mean"] = float(SQ_all)
  codalab_output["rq_mean"] = float(RQ_all)
  codalab_output["iou_mean"] = float(mIoU)
  codalab_output["pq_stuff"] = float(PQ_stuff)
  codalab_output["rq_stuff"] = float(RQ_stuff)
  codalab_output["sq_stuff"] = float(SQ_stuff)
  codalab_output["pq_things"] = float(PQ_things)
  codalab_output["rq_things"] = float(RQ_things)
  codalab_output["sq_things"] = float(SQ_things)

  print("Completed in {} s".format(complete_time))

  if FLAGS.output is not None:
    table = []
    for cl in all_classes:
      entry = output_dict[cl]
      table.append({
          "class": cl,
          "pq": "{:.3}".format(entry["PQ"]),
          "sq": "{:.3}".format(entry["SQ"]),
          "rq": "{:.3}".format(entry["RQ"]),
          "iou": "{:.3}".format(entry["IoU"])
      })

    print("Generating output files.")
    # save to yaml
    output_filename = os.path.join(FLAGS.output, 'scores.txt')
    with open(output_filename, 'w') as outfile:
      yaml.dump(codalab_output, outfile, default_flow_style=False)

    ## producing a detailed result page.
    output_filename = os.path.join(FLAGS.output, "detailed_results.html")
    with open(output_filename, "w") as html_file:
      html_file.write("""
<!doctype html>
<html lang="en" style="scroll-behavior: smooth;">
<head>
  <script src='https://cdnjs.cloudflare.com/ajax/libs/tabulator/4.4.3/js/tabulator.min.js'></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/tabulator/4.4.3/css/bulma/tabulator_bulma.min.css">
</head>
<body>
  <div id="classwise_results"></div>

<script>
  let table_data = """ + json.dumps(table) + """


  table = new Tabulator("#classwise_results", {
    layout: "fitData",
    data: table_data,
    columns: [{title: "Class", field:"class", width:200}, 
              {title: "PQ", field:"pq", width:100, align: "center"},
              {title: "SQ", field:"sq", width:100, align: "center"},
              {title: "RQ", field:"rq", width:100, align: "center"},
              {title: "IoU", field:"iou", width:100, align: "center"}]
  });
</script>
</body>
</html>""")
