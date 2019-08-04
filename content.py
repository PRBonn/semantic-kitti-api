#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import os
import yaml
import numpy as np
import collections
from auxiliary.laserscan import SemLaserScan


if __name__ == '__main__':
  parser = argparse.ArgumentParser("./content.py")
  parser.add_argument(
      '--dataset', '-d',
      type=str,
      required=True,
      help='Dataset to calculate content. No Default',
  )
  parser.add_argument(
      '--config', '-c',
      type=str,
      required=False,
      default="config/semantic-kitti.yaml",
      help='Dataset config file. Defaults to %(default)s',
  )
  FLAGS, unparsed = parser.parse_known_args()

  # print summary of what we will do
  print("*" * 80)
  print("INTERFACE:")
  print("Dataset", FLAGS.dataset)
  print("Config", FLAGS.config)
  print("*" * 80)

  # open config file
  try:
    print("Opening config file %s" % FLAGS.config)
    CFG = yaml.safe_load(open(FLAGS.config, 'r'))
  except Exception as e:
    print(e)
    print("Error opening yaml file.")
    quit()

  # get training sequences to calculate statistics
  sequences = CFG["split"]["train"]
  print("Analizing sequences", sequences)

  # create content accumulator
  accum = {}
  total = 0.0
  for key, _ in CFG["labels"].items():
    accum[key] = 0

  # itearate over sequences
  for seq in sequences:
    seq_accum = {}
    seq_total = 0.0
    for key, _ in CFG["labels"].items():
      seq_accum[key] = 0

    # make seq string
    print("*" * 80)
    seqstr = '{0:02d}'.format(int(seq))
    print("parsing seq {}".format(seq))

    # does sequence folder exist?
    scan_paths = os.path.join(FLAGS.dataset, "sequences",
                              seqstr, "velodyne")
    if os.path.isdir(scan_paths):
      print("Sequence folder exists!")
    else:
      print("Sequence folder doesn't exist! Exiting...")
      quit()

    # populate the pointclouds
    scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(scan_paths)) for f in fn]
    scan_names.sort()

    # does sequence folder exist?
    label_paths = os.path.join(FLAGS.dataset, "sequences",
                               seqstr, "labels")
    if os.path.isdir(label_paths):
      print("Labels folder exists!")
    else:
      print("Labels folder doesn't exist! Exiting...")
      quit()
    # populate the pointclouds
    label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(label_paths)) for f in fn]
    label_names.sort()

    # check that there are same amount of labels and scans
    print(len(label_names))
    print(len(scan_names))
    assert(len(label_names) == len(scan_names))

    # create a scan
    nclasses = len(CFG["labels"])
    scan = SemLaserScan(nclasses, CFG["color_map"], project=False)

    for idx in range(len(scan_names)):
      # open scan
      print(label_names[idx])
      scan.open_scan(scan_names[idx])
      scan.open_label(label_names[idx])
      # make histogram and accumulate
      count = np.bincount(scan.sem_label)
      seq_total += count.sum()
      for key, data in seq_accum.items():
        if count.size > key:
          seq_accum[key] += count[key]
          # zero the count
          count[key] = 0
      for i, c in enumerate(count):
        if c > 0:
          print("wrong label ", i, ", nr: ", c)

    seq_accum = collections.OrderedDict(
        sorted(seq_accum.items(), key=lambda t: t[0]))

    # print and send to total
    total += seq_total
    print("seq ", seqstr, "total", seq_total)
    for key, data in seq_accum.items():
      accum[key] += data
      print(data)

  # print content to fill yaml file
  print("*" * 80)
  print("Content in training set")
  print(accum)
  accum = collections.OrderedDict(sorted(accum.items(), key=lambda t: t[0]))
  for key, data in accum.items():
    print(" {}: {}".format(key, data / total))
