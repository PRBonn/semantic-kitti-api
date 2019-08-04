#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import os
import yaml
import numpy as np
import collections
from auxiliary.laserscan import SemLaserScan


if __name__ == '__main__':
  parser = argparse.ArgumentParser("./count.py")
  parser.add_argument(
      '--dataset', '-d',
      type=str,
      required=True,
      help='Dataset to calculate scan count. No Default',
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
  sequences.extend(CFG["split"]["valid"])
  sequences.extend(CFG["split"]["test"])
  sequences.sort()
  print("Analizing sequences", sequences, "to count number of scans")

  # iterate over sequences and count scan number
  for seq in sequences:
    seqstr = '{0:02d}'.format(int(seq))
    scan_paths = os.path.join(FLAGS.dataset, "sequences",
                              seqstr, "velodyne")
    if not os.path.isdir(scan_paths):
      print("Sequence", seqstr, "doesn't exist! Exiting...")
      quit()
    scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(scan_paths)) for f in fn]
    print(seqstr, ",", len(scan_names))
