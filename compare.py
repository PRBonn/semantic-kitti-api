#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import os
import yaml
from auxiliary.laserscan import LaserScan, SemLaserScan
from auxiliary.laserscancomp import LaserScanComp

if __name__ == '__main__':
  parser = argparse.ArgumentParser("./compare.py")
  parser.add_argument(
      '--dataset_a',
      type=str,
      required=True,
      help='Dataset A to visualize. No Default',
  )
  parser.add_argument(
      '--dataset_b',
      type=str,
      required=True,
      help='Dataset B to visualize. No Default',
  )
  parser.add_argument(
      '--config', '-c',
      type=str,
      required=False,
      default="config/semantic-kitti.yaml",
      help='Dataset config file. Defaults to %(default)s',
  )
  parser.add_argument(
      '--sequence', '-s',
      type=str,
      default="00",
      required=False,
      help='Sequence to visualize. Defaults to %(default)s',
  )
  parser.add_argument(
      '--ignore_images', '-r',
      dest='ignore_images',
      default=False,
      required=False,
      action='store_true',
      help='Visualize range image projections too. Defaults to %(default)s',
  )
  parser.add_argument(
      '--do_instances', '-d',
      dest='do_instances',
      default=False,
      required=False,
      action='store_true',
      help='Visualize instances too. Defaults to %(default)s',
  )
  parser.add_argument(
      '--link', '-l',
      dest='link',
      default=False,
      required=False,
      action='store_true',
      help='Link viewpoint changes across windows. Defaults to %(default)s',
  )
  parser.add_argument(
      '--offset',
      type=int,
      default=0,
      required=False,
      help='Sequence to start. Defaults to %(default)s',
  )
  parser.add_argument(
      '--ignore_safety',
      dest='ignore_safety',
      default=False,
      required=False,
      action='store_true',
      help='Normally you want the number of labels and ptcls to be the same,'
      ', but if you are not done inferring this is not the case, so this disables'
      ' that safety.'
      'Defaults to %(default)s',
  )
  FLAGS, unparsed = parser.parse_known_args()

  # print summary of what we will do
  print("*" * 80)
  print("INTERFACE:")
  print("Dataset A", FLAGS.dataset_a)
  print("Dataset B", FLAGS.dataset_b)
  print("Config", FLAGS.config)
  print("Sequence", FLAGS.sequence)
  print("ignore_images", FLAGS.ignore_images)
  print("do_instances", FLAGS.do_instances)
  print("link", FLAGS.link)
  print("ignore_safety", FLAGS.ignore_safety)
  print("offset", FLAGS.offset)
  print("*" * 80)

  # open config file
  try:
    print("Opening config file %s" % FLAGS.config)
    CFG = yaml.safe_load(open(FLAGS.config, 'r'))
  except Exception as e:
    print(e)
    print("Error opening yaml file.")
    quit()

  # fix sequence name
  FLAGS.sequence = '{0:02d}'.format(int(FLAGS.sequence))

  # does sequence folder exist?
  scan_a_paths = os.path.join(FLAGS.dataset_a, "sequences", FLAGS.sequence, "velodyne")
  scan_b_paths = os.path.join(FLAGS.dataset_b, "sequences", FLAGS.sequence, "velodyne")

  if os.path.isdir(scan_a_paths):
    print("Sequence folder a exists! Using sequence from %s" % scan_a_paths)
  else:
    print("Sequence folder a doesn't exist! Exiting...")
    quit()

  if os.path.isdir(scan_b_paths):
    print("Sequence folder b exists! Using sequence from %s" % scan_b_paths)
  else:
    print("Sequence folder b doesn't exist! Exiting...")
    quit()

  # populate the pointclouds
  scan_a_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(scan_a_paths)) for f in fn]
  scan_a_names.sort()

  scan_b_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(scan_b_paths)) for f in fn]
  scan_b_names.sort()

  # does sequence folder exist?
  label_a_paths = os.path.join(FLAGS.dataset_a, "sequences", FLAGS.sequence, "labels")
  label_b_paths = os.path.join(FLAGS.dataset_b, "sequences", FLAGS.sequence, "labels")

  if os.path.isdir(label_a_paths):
    print("Labels folder a exists! Using labels from %s" % label_a_paths)
  else:
    print("Labels folder a doesn't exist! Exiting...")
    quit()

  if os.path.isdir(label_b_paths):
    print("Labels folder b exists! Using labels from %s" % label_b_paths)
  else:
    print("Labels folder b doesn't exist! Exiting...")
    quit()

  # populate the pointclouds
  label_a_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(label_a_paths)) for f in fn]
  label_a_names.sort()
  label_b_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(label_b_paths)) for f in fn]
  label_b_names.sort()

  # check that there are same amount of labels and scans
  if not FLAGS.ignore_safety:
    assert len(label_a_names) == len(scan_a_names)
    assert len(label_b_names) == len(scan_b_names)
    assert len(scan_a_names) == len(scan_b_names)

  # create scans
  color_dict = CFG["color_map"]
  nclasses = len(color_dict)
  scan_a = SemLaserScan(nclasses, color_dict, project=True)
  scan_b = SemLaserScan(nclasses, color_dict, project=True)

  # create a visualizer
  images = not FLAGS.ignore_images
  vis = LaserScanComp(scans=(scan_a, scan_b),
                     scan_names=(scan_a_names, scan_b_names),
                     label_names=(label_a_names, label_b_names),
                     offset=FLAGS.offset, images=images, instances=FLAGS.do_instances, link=FLAGS.link)

  # print instructions
  print("To navigate:")
  print("\tb: back (previous scan)")
  print("\tn: next (next scan)")
  print("\tq: quit (exit program)")

  # run the visualizer
  vis.run()
