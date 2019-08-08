#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import os
import yaml
import numpy as np
from collections import deque
import shutil
from numpy.linalg import inv
import struct
import time

def parse_calibration(filename):
  """ read calibration file with given filename

      Returns
      -------
      dict
          Calibration matrices as 4x4 numpy arrays.
  """
  calib = {}

  calib_file = open(filename)
  for line in calib_file:
    key, content = line.strip().split(":")
    values = [float(v) for v in content.strip().split()]

    pose = np.zeros((4, 4))
    pose[0, 0:4] = values[0:4]
    pose[1, 0:4] = values[4:8]
    pose[2, 0:4] = values[8:12]
    pose[3, 3] = 1.0

    calib[key] = pose

  calib_file.close()

  return calib


def parse_poses(filename, calibration):
  """ read poses file with per-scan poses from given filename

      Returns
      -------
      list
          list of poses as 4x4 numpy arrays.
  """
  file = open(filename)

  poses = []

  Tr = calibration["Tr"]
  Tr_inv = inv(Tr)

  for line in file:
    values = [float(v) for v in line.strip().split()]

    pose = np.zeros((4, 4))
    pose[0, 0:4] = values[0:4]
    pose[1, 0:4] = values[4:8]
    pose[2, 0:4] = values[8:12]
    pose[3, 3] = 1.0

    poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

  return poses


if __name__ == '__main__':
  start_time = time.time()

  parser = argparse.ArgumentParser("./generate_sequential.py")
  parser.add_argument(
      '--dataset',
      '-d',
      type=str,
      required=True,
      help='dataset folder containing all sequences in a folder called "sequences".',
  )

  parser.add_argument(
      '--output',
      '-o',
      type=str,
      required=True,
      help='output folder for generated sequence scans.',
  )

  parser.add_argument(
      '--sequence_length',
      '-s',
      type=int,
      required=True,
      help='length of sequence, i.e., how many scans are concatenated.',
  )

  FLAGS, unparsed = parser.parse_known_args()

  # print summary of what we will do
  print("*" * 80)
  print(" dataset folder: ", FLAGS.dataset)
  print("  output folder: ", FLAGS.output)
  print("sequence length: ", FLAGS.sequence_length)
  print("*" * 80)

  sequences_dir = os.path.join(FLAGS.dataset, "sequences")
  sequence_folders = [
      f for f in sorted(os.listdir(sequences_dir))
      if os.path.isdir(os.path.join(sequences_dir, f))
  ]

  for folder in sequence_folders:
    input_folder = os.path.join(sequences_dir, folder)
    output_folder = os.path.join(FLAGS.output, "sequences", folder)
    velodyne_folder = os.path.join(output_folder, "velodyne")
    labels_folder = os.path.join(output_folder, "labels")

    if os.path.exists(output_folder) or os.path.exists(
            velodyne_folder) or os.path.exists(labels_folder):
      print("Output folder '{}' already exists!".format(output_folder))
      answer = input("Overwrite? [y/N] ")
      if answer != "y":
        print("Aborted.")
        exit(1)
      if not os.path.exists(velodyne_folder):
        os.makedirs(velodyne_folder)
      if not os.path.exists(labels_folder):
        os.makedirs(labels_folder)
    else:
      os.makedirs(velodyne_folder)
      os.makedirs(labels_folder)

    shutil.copy(os.path.join(input_folder, "poses.txt"), output_folder)
    shutil.copy(os.path.join(input_folder, "calib.txt"), output_folder)

    scan_files = [
        f for f in sorted(os.listdir(os.path.join(input_folder, "velodyne")))
        if f.endswith(".bin")
    ]

    history = deque()

    calibration = parse_calibration(os.path.join(input_folder, "calib.txt"))
    poses = parse_poses(os.path.join(input_folder, "poses.txt"), calibration)

    progress = 10

    print("Processing {} ".format(folder), end="", flush=True)

    for i, f in enumerate(scan_files):
      # read scan and labels, get pose
      scan_filename = os.path.join(input_folder, "velodyne", f)
      scan = np.fromfile(scan_filename, dtype=np.float32)

      scan = scan.reshape((-1, 4))

      label_filename = os.path.join(input_folder, "labels", os.path.splitext(f)[0] + ".label")
      labels = np.fromfile(label_filename, dtype=np.uint32)
      labels = labels.reshape((-1))

      # convert points to homogenous coordinates (x, y, z, 1)
      points = np.ones((scan.shape))
      points[:, 0:3] = scan[:, 0:3]
      remissions = scan[:, 3]

      pose = poses[i]

      # prepare single numpy array for all points that can be written at once.
      num_concat_points = points.shape[0]
      num_concat_points += sum([past["points"].shape[0] for past in history])
      concated_points = np.zeros((num_concat_points * 4), dtype = np.float32)
      concated_labels = np.zeros((num_concat_points), dtype = np.uint32)

      start = 0
      concated_points[4 * start:4 * (start + points.shape[0])] = scan.reshape((-1))
      concated_labels[start:start + points.shape[0]] = labels
      start += points.shape[0]

      for past in history:
        diff = np.matmul(inv(pose), past["pose"])
        tpoints = np.matmul(diff, past["points"].T).T
        tpoints[:, 3] = past["remissions"]
        tpoints = tpoints.reshape((-1))

        concated_points[4 * start:4 * (start + past["points"].shape[0])] = tpoints
        concated_labels[start:start + past["labels"].shape[0]] = past["labels"]
        start += past["points"].shape[0]


      # write scan and labels in one pass.
      concated_points.tofile(os.path.join(velodyne_folder, f))
      concated_labels.tofile(os.path.join(labels_folder, os.path.splitext(f)[0] + ".label")) 

      # append current data to history queue.
      history.appendleft({
          "points": points,
          "labels": labels,
          "remissions": remissions,
          "pose": pose.copy()
      })

      if len(history) >= FLAGS.sequence_length:
        history.pop()

      if 100.0 * i / len(scan_files) >= progress:
        print(".", end="", flush=True)
        progress = progress + 10
    print("finished.")


  print("execution time: {}".format(time.time() - start_time))
