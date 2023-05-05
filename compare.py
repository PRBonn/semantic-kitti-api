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
      '--dataset', '-d',
      type=str,
      required=True,
      help='Dataset to visualize. No Default',
  )
  parser.add_argument(
      '--labels',
      required=True,
      nargs='+',
      help='Labels A to visualize. No Default',
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
      '--do_instances', '-i',
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
  parser.add_argument(
    '--color_learning_map',
    dest='color_learning_map',
    default=False,
    required=False,
    action='store_true',
    help='Apply learning map to color map: visualize only classes that were trained on',
  )
  FLAGS, unparsed = parser.parse_known_args()

  # print summary of what we will do
  print("*" * 80)
  print("INTERFACE:")
  print("Labels: ", FLAGS.labels)
  print("Config", FLAGS.config)
  print("Sequence", FLAGS.sequence)
  print("ignore_images", FLAGS.ignore_images)
  print("do_instances", FLAGS.do_instances)
  print("link", FLAGS.link)
  print("ignore_safety", FLAGS.ignore_safety)
  print("color_learning_map", FLAGS.color_learning_map)
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
  scan_paths = os.path.join(FLAGS.dataset, "sequences", FLAGS.sequence, "velodyne")

  if os.path.isdir(scan_paths):
    print("Sequence folder a exists! Using sequence from %s" % scan_paths)
  else:
    print(f"Sequence folder {scan_paths} doesn't exist! Exiting...")
    quit()

  # populate the pointclouds
  scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(scan_paths)) for f in fn]
  scan_names.sort()

  print(len(scan_names))

  # does sequence folder exist?
  assert len(FLAGS.labels) == 2
  labels_a, labels_b = FLAGS.labels[0], FLAGS.labels[1]
  label_a_paths = os.path.join(FLAGS.dataset, "sequences", FLAGS.sequence, labels_a)
  label_b_paths = os.path.join(FLAGS.dataset, "sequences", FLAGS.sequence, labels_b)

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
    assert len(label_a_names) == len(scan_names)
    assert len(label_b_names) == len(scan_names)

  # create scans
  color_dict = CFG["color_map"]
  if FLAGS.color_learning_map:
    learning_map_inv = CFG["learning_map_inv"]
    learning_map = CFG["learning_map"]
    color_dict = {key: color_dict[learning_map_inv[learning_map[key]]] for key, value in color_dict.items()}

  scan_b = SemLaserScan(color_dict, project=True)
  scan_a = SemLaserScan(color_dict, project=True)

  # create a visualizer
  images = not FLAGS.ignore_images
  vis = LaserScanComp(scans=(scan_a, scan_b),
                     scan_names=scan_names,
                     label_names=(label_a_names, label_b_names),
                     offset=FLAGS.offset, images=images, instances=FLAGS.do_instances, link=FLAGS.link)

  # print instructions
  print("To navigate:")
  print("\tb: back (previous scan)")
  print("\tn: next (next scan)")
  print("\tq: quit (exit program)")

  # run the visualizer
  vis.run()
