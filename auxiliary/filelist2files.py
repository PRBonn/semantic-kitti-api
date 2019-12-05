#!/usr/bin/python3

import os
import sys
import shutil
import numpy as np
import scipy.io as sio

from tqdm import tqdm

def pack(array):
  """ convert a boolean array into a bitwise array. """
  array = array.reshape((-1))

  #compressing bit flags.
  # yapf: disable
  compressed = array[::8] << 7 | array[1::8] << 6  | array[2::8] << 5 | array[3::8] << 4 | array[4::8] << 3 | array[5::8] << 2 | array[6::8] << 1 | array[7::8]
  # yapf: enable

  return np.array(compressed, dtype=np.uint8)

if __name__ == "__main__":
  """
  Convert a given directory of mat files and the given filelist into a separate directory
  containing the files in the file list.
  """

  if len(sys.argv) < 2:
    print("./filelist2files.py <input-root-directory> <output-root-directory> [<filelist>]")
    exit(1)

  src_dir = sys.argv[1]
  dst_dir = sys.argv[2]

  files = None

  if len(sys.argv) > 3:
    files = [line.strip().split("_") for line in open(sys.argv[3])]
  else:

    seq_dirs = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
    files = []
    for d in seq_dirs:
      files.extend([(d, os.path.splitext(f)[0]) for f in os.listdir(os.path.join(src_dir, d, "input"))])

  print("Processing {} files.".format(len(files)))

  for seq_dir, filename in tqdm(files):

    if os.path.exists(os.path.join(src_dir, seq_dir, "input", filename + ".mat")):
      data = sio.loadmat(os.path.join(src_dir, seq_dir, "input", filename + ".mat"))

      out_dir = os.path.join(dst_dir, seq_dir, "voxels")
      os.makedirs(out_dir, exist_ok=True)

      compressed = pack(data["voxels"])
      compressed.tofile(os.path.join(out_dir, os.path.splitext(filename)[0] + ".bin"))

    if os.path.exists(os.path.join(src_dir, seq_dir, "target_gt", filename + ".mat")):
      data = sio.loadmat(os.path.join(src_dir, seq_dir, "target_gt", filename + ".mat"))

      out_dir = os.path.join(dst_dir, seq_dir, "voxels")
      os.makedirs(out_dir, exist_ok=True)

      labels = data["voxels"].astype(np.uint16)
      labels.tofile(os.path.join(out_dir, os.path.splitext(filename)[0] + ".label"))

      occlusions = pack(data["occluded"])
      occlusions.tofile(os.path.join(out_dir, os.path.splitext(filename)[0] + ".occluded"))

      invalid = pack(data["invalid"])
      invalid.tofile(os.path.join(out_dir, os.path.splitext(filename)[0] + ".invalid"))
