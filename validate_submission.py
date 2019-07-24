#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import zipfile
import argparse
import os


class ValidationException(Exception):
  pass


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Validate a submission zip file needed to evaluate on CodaLab competitions.\n\nThe verification tool checks:\n  1. correct folder structure,\n  2. existence of label files for each scan,\n  3. count of labels for each scan.\nInvalid labels are ignored by the evaluation script, therefore we don't check\nfor invalid labels.", formatter_class=argparse.RawTextHelpFormatter)

  parser.add_argument(
      "zipfile",
      type=str,
      help='zip file that should be validated.',
  )

  parser.add_argument(
      'dataset',
      type=str,
      help='directory containing the folder "sequences" containing folders "11", ..., "21" with the "velodyne" files.'
  )

  parser.add_argument(
      "--task",
      type=str,
      choices=["segmentation"],
      default="segmentation",
      help='task for which the zip file should be validated.'
  )

  FLAGS, _ = parser.parse_known_args()

  checkmark = "\u2713"

  try:

    print('Validating zip archive "{}".\n'.format(FLAGS.zipfile))

    print("  1. Checking filename.............. ", end="", flush=True)
    if not FLAGS.zipfile.endswith('.zip'):
      raise ValidationException('Competition bundle must end with ".zip"')
    print(checkmark)

    with zipfile.ZipFile(FLAGS.zipfile) as zipfile:
      if FLAGS.task == "segmentation":

        print("  2. Checking directory structure... ", end="", flush=True)

        directories = [folder.filename for folder in zipfile.infolist() if folder.filename.endswith("/")]
        if "sequences/" not in directories:
          raise ValidationException('Directory "sequences" missing inside zip file.')

        for sequence in range(11, 22):
          sequence_directory = "sequences/{}/".format(sequence)
          if sequence_directory not in directories:
            raise ValidationException('Directory "{}" missing inside zip file.'.format(sequence_directory))
          predictions_directory = sequence_directory + "predictions/"
          if predictions_directory not in directories:
            raise ValidationException('Directory "{}" missing inside zip file.'.format(predictions_directory))

        print(checkmark)

        print('  3. Checking file sizes............ ', end='', flush=True)

        prediction_files = {info.filename: info for info in zipfile.infolist() if not info.filename.endswith("/")}

        for sequence in range(11, 22):
          sequence_directory = 'sequences/{}'.format(sequence)
          velodyne_directory = os.path.join(FLAGS.dataset, 'sequences/{}/velodyne/'.format(sequence))

          velodyne_files = sorted([os.path.join(velodyne_directory, file) for file in os.listdir(velodyne_directory)])
          label_files = sorted([os.path.join(sequence_directory, "predictions", os.path.splitext(filename)[0] + ".label")
                                for filename in os.listdir(velodyne_directory)])

          for velodyne_file, label_file in zip(velodyne_files, label_files):
            num_points = os.path.getsize(velodyne_file) / (4 * 4)

            if label_file not in prediction_files:
              raise ValidationException('"' + label_file + '" is missing inside zip.')

            num_labels = prediction_files[label_file].file_size / 4
            if num_labels != num_points:
              raise ValidationException('label file "' + label_file +
                                        "' should have {} labels, but found {} labels!".format(int(num_points), int(num_labels)))

        print(checkmark)
      else:
        # TODO scene completion.
        raise NotImplementedError("Unknown task.")
  except ValidationException as ex:
    print("\n\n  " + "\u001b[1;31m>>> Error: " + str(ex) + "\u001b[0m")
    exit(1)

  print("\n\u001b[1;32mEverything ready for submission!\u001b[0m  \U0001F389")
