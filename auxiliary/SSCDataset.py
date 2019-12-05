import os
import numpy as np


def unpack(compressed):
  ''' given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
  uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
  uncompressed[::8] = compressed[:] >> 7 & 1
  uncompressed[1::8] = compressed[:] >> 6 & 1
  uncompressed[2::8] = compressed[:] >> 5 & 1
  uncompressed[3::8] = compressed[:] >> 4 & 1
  uncompressed[4::8] = compressed[:] >> 3 & 1
  uncompressed[5::8] = compressed[:] >> 2 & 1
  uncompressed[6::8] = compressed[:] >> 1 & 1
  uncompressed[7::8] = compressed[:] & 1

  return uncompressed


SPLIT_SEQUENCES = {
    "train": ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
    "valid": ["08"],
    "test": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"]
}

SPLIT_FILES = {
    "train": [".bin", ".label", ".invalid", ".occluded"],
    "valid": [".bin", ".label", ".invalid", ".occluded"],
    "test": [".bin"]
}

EXT_TO_NAME = {".bin": "input", ".label": "label", ".invalid": "invalid", ".occluded": "occluded"}

VOXEL_DIMS = (256, 256, 32)


class SSCDataset:
  def __init__(self, directory, split="train"):
    """ Load data from given dataset directory. """

    self.files = {}
    self.filenames = []

    for ext in SPLIT_FILES[split]:
      self.files[EXT_TO_NAME[ext]] = []

    for sequence in SPLIT_SEQUENCES[split]:
      complete_path = os.path.join(directory, "sequences", sequence, "voxels")
      if not os.path.exists(complete_path): raise RuntimeError("Voxel directory missing: " + complete_path)

      files = os.listdir(complete_path)
      for ext in SPLIT_FILES[split]:
        data = sorted([os.path.join(complete_path, f) for f in files if f.endswith(ext)])
        if len(data) == 0: raise RuntimeError("Missing data for " + EXT_TO_NAME[ext])
        self.files[EXT_TO_NAME[ext]].extend(data)

      # this information is handy for saving the data later, since you need to provide sequences/XX/predictions/000000.label:
      self.filenames.extend(
          sorted([(sequence, os.path.splitext(f)[0]) for f in files if f.endswith(SPLIT_FILES[split][0])]))

    self.num_files = len(self.filenames)

    # sanity check:
    for k, v in self.files.items():
      print(k, len(v))
      assert (len(v) == self.num_files)

  def __len__(self):
    return self.num_files

  def __getitem__(self, t):
    """ fill dictionary with available data for given index . """
    collection = {}

    # read raw data and unpack (if necessary)
    for typ in self.files.keys():
      scan_data = None
      if typ == "label":
        scan_data = np.fromfile(self.files[typ][t], dtype=np.uint16)
      else:
        scan_data = unpack(np.fromfile(self.files[typ][t], dtype=np.uint8))

      # turn in actual voxel grid representation.
      collection[typ] = scan_data.reshape(VOXEL_DIMS)

    return self.filenames[t], collection


if __name__ == "__main__":
  # Small example of the usage.
  
  # Replace "/path/to/semantic/kitti/" with actual path to the folder containing the "sequences" folder
  dataset = SSCDataset("/path/to/semantic/kitti/")
  print("# files: {}".format(len(dataset)))

  (seq, filename), data = dataset[100]
  print("Contents of entry 100 (" + seq + ":" + filename + "):")
  for k, v in data.items():
    print("  {:8} : shape = {}, contents = {}".format(k, v.shape, np.unique(v)))
