# API for SemanticKITTI

This repository contains helper scripts to open, visualize, process, and 
evaluate results for point clouds and labels from the SemanticKITTI dataset.

- Link to original [KITTI Odometry Benchmark](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) Dataset
- Link to [SemanticKITTI dataset](http://semantic-kitti.org/).
- Link to SemanticKITTI benchmark [competition](http://semantic-kitti.org/tasks.html).

---
##### Example of 3D pointcloud from sequence 13:
<img src="https://image.ibb.co/kyhCrV/scan1.png" width="1000">

---

##### Example of 2D spherical projection from sequence 13:
<img src="https://image.ibb.co/hZtVdA/scan2.png" width="1000">

---

##### Example of voxelized point clouds for semantic scene completion:
<img src="https://user-images.githubusercontent.com/11506664/70214770-4d43ff80-173c-11ea-940d-3950d8f24eaf.png" width="1000">

---

## Data organization

The data is organized in the following format:

```
/kitti/dataset/
          └── sequences/
                  ├── 00/
                  │   ├── poses.txt
                  │   ├── image_2/
                  │   ├── image_3/
                  │   ├── labels/
                  │   │     ├ 000000.label
                  │   │     └ 000001.label
                  |   ├── voxels/
                  |   |     ├ 000000.bin
                  |   |     ├ 000000.label
                  |   |     ├ 000000.occluded
                  |   |     ├ 000000.invalid
                  |   |     ├ 000001.bin
                  |   |     ├ 000001.label
                  |   |     ├ 000001.occluded
                  |   |     ├ 000001.invalid
                  │   └── velodyne/
                  │         ├ 000000.bin
                  │         └ 000001.bin
                  ├── 01/
                  ├── 02/
                  .
                  .
                  .
                  └── 21/
```

- From [KITTI Odometry](http://www.cvlibs.net/datasets/kitti/eval_odometry.php): 
  - `image_2` and `image_3` correspond to the rgb images for each sequence.
  - `velodyne` contains the pointclouds for each scan in each sequence. Each 
`.bin` scan is a list of float32 points in [x,y,z,remission] format. See
[laserscan.py](auxiliary/laserscan.py) to see how the points are read.
- From SemanticKITTI:
  - `labels` contains the labels for each scan in each sequence. Each `.label` 
file contains a uint32 label for each point in the corresponding `.bin` scan.
See [laserscan.py](auxiliary/laserscan.py) to see how the labels are read.
  - `poses.txt` contain the manually looped-closed poses for each capture (in
the camera frame) that were used in the annotation tools to aggregate all
the point clouds.
  - `voxels` contains all information needed for the task of semantic scene completion. Each `.bin` file contains for each voxel if that voxel is occupied by laser measurements in a packed binary format. This is the input to the semantic scene completion task and it corresponds to the voxelization of a single LiDAR scan. Each`.label` file contains for each voxel of the completed scene a label in binary format. The label is a 16-bit unsigned integer (aka uint16_t) for each voxel. `.invalid` and `.occluded` contain information about the occlusion of voxel. Invalid voxels are voxels that are occluded from each view position and occluded voxels are occluded in the first view point. See also [SSCDataset.py](auxiliary/SSCDataset.py) for more information on loading the data.

The main configuration file for the data is in `config/semantic-kitti.yaml`. In this file you will find:

- `labels`: dictionary which maps numeric labels in `.label` file to a string class. Example: `10: "car"`
- `color_map`: dictionary which maps numeric labels in `.label` file to a bgr color for visualization. Example `10: [245, 150, 100] # car, blue-ish`
- `content`: dictionary with content of each class in labels, as a ratio to 
the number of total points in the dataset. This can be obtained by running the
[./content.py](./content.py) script, and is used to calculate the weights for the cross
entropy in all baseline methods (in order handle class imbalance).
- `learning_map`: dictionary which maps each class label to its cross entropy
equivalent, for learning. This is done to mask undesired classes, map different
classes together, and because the cross entropy expects a value in
[0, numclasses - 1]. We also provide [./remap_semantic_labels.py](./remap_semantic_labels.py),
a script that uses this dictionary to put the label files in the cross entropy format,
so that you can use the labels directly in your training pipeline.
Examples:
  ```yaml
    0 : 0     # "unlabeled"
    1 : 0     # "outlier" to "unlabeled" -> gets ignored in training, with unlabeled
    10: 1     # "car"
    252: 1    # "moving-car" to "car" -> gets merged with static car class
  ```
- `learning_map_inv`: dictionary with inverse of the previous mapping, allows to
map back the classes only to the interest ones (for saving point cloud predictions
in original label format). We also provide [./remap_semantic_labels.py](./remap_semantic_labels.py),
a script that uses this dictionary to put the label files in the original format,
when instantiated with the `--inverse` flag.
- `learning_ignore`: dictionary that contains for each cross entropy class if it
will be ignored during training and evaluation or not. For example, class `unlabeled` gets
ignored in both training and evaluation.
- `split`: contains 3 lists, with the sequence numbers for training, validation, and evaluation.


## Dependencies for API:

System dependencies

```sh
$ sudo apt install python3-dev python3-pip python3-pyqt5.qtopengl # for visualization
```

Python dependencies

```sh
$ sudo pip3 install -r requirements.txt
```

## Scripts:

**ALL OF THE SCRIPTS CAN BE INVOKED WITH THE --help (-h) FLAG, FOR EXTRA INFORMATION AND OPTIONS.**

#### Visualization 


##### Point Clouds

To visualize the data, use the `visualize.py` script. It will open an interactive
opengl visualization of the pointclouds along with a spherical projection of
each scan into a 64 x 1024 image.

```sh
$ ./visualize.py --sequence 00 --dataset /path/to/kitti/dataset/
```

where:
- `sequence` is the sequence to be accessed.
- `dataset` is the path to the kitti dataset where the `sequences` directory is.

Navigation:
- `n` is next scan,
- `b` is previous scan,
- `esc` or `q` exits.

In order to visualize your predictions instead, the `--predictions` option replaces
visualization of the labels with the visualization of your predictions:

```sh
$ ./visualize.py --sequence 00 --dataset /path/to/kitti/dataset/ --predictions /path/to/your/predictions
```

To directly compare two sets of data, use the `compare.py` script. It will open an interactive
opengl visualization of the pointcloud labels.

```sh
$ ./compare.py --sequence 00 --dataset_a /path/to/dataset_a/ --dataset_b /path/to/kitti/dataset_b/
```

where:
- `sequence` is the sequence to be accessed.
- `dataset_a` is the path to a dataset in KITTI format where the `sequences` directory is.
- `dataset_b` is the path to another dataset in KITTI format where the `sequences` directory is.

Navigation:
- `n` is next scan,
- `b` is previous scan,
- `esc` or `q` exits.

#### Voxel Grids for Semantic Scene Completion

To visualize the data, use the `visualize_voxels.py` script. It will open an interactive
opengl visualization of the voxel grids and options to visualize the provided voxelizations 
of the LiDAR data.

```sh
$ ./visualize_voxels.py --sequence 00 --dataset /path/to/kitti/dataset/
```

where:
- `sequence` is the sequence to be accessed.
- `dataset` is the path to the kitti dataset where the `sequences` directory is.

Navigation:
- `n` is next scan,
- `b` is previous scan,
- `esc` or `q` exits.

Note: Holding the forward/backward buttons triggers the playback mode.


#### LiDAR-based Moving Object Segmentation ([LiDAR-MOS](https://github.com/PRBonn/LiDAR-MOS))

To visualize the data, use the `visualize_mos.py` script. It will open an interactive
opengl visualization of the voxel grids and options to visualize the provided voxelizations 
of the LiDAR data.

```sh
$ ./visualize_mos.py --sequence 00 --dataset /path/to/kitti/dataset/
```

where:
- `sequence` is the sequence to be accessed.
- `dataset` is the path to the kitti dataset where the `sequences` directory is.

Navigation:
- `n` is next scan,
- `b` is previous scan,
- `esc` or `q` exits.

Note: Holding the forward/backward buttons triggers the playback mode.


#### Evaluation

To evaluate the predictions of a method, use the [evaluate_semantics.py](./evaluate_semantics.py) to evaluate 
semantic segmentation, [evaluate_completion.py](./evaluate_completion.py) to evaluate the semantic scene completion and [evaluate_panoptic.py](./evaluate_panoptic.py) to evaluate panoptic segmentation.
**Important:** The labels and the predictions need to be in the original
label format, which means that if a method learns the cross-entropy mapped
classes, they need to be passed through the `learning_map_inv` dictionary
to be sent to the original dataset format. This is to prevent changes in the
dataset interest classes from affecting intermediate outputs of approaches, 
since the original labels will stay the same. 
For semantic segmentation, we provide the `remap_semantic_labels.py` script to make this 
shift before the training, and once again before the evaluation, selecting which are the interest 
classes in the configuration file. 
The data needs to be either:

- In a separate directory with this format:

  ```
  /method_predictions/
            └── sequences
                ├── 00
                │   └── predictions
                │         ├ 000000.label
                │         └ 000001.label
                ├── 01
                ├── 02
                .
                .
                .
                └── 21
  ```

  And run:

  ```sh
  $ ./evaluate_semantics.py --dataset /path/to/kitti/dataset/ --predictions /path/to/method_predictions --split train/valid/test # depending of desired split to evaluate
  ```

  or 

    ```sh
  $ ./evaluate_completion.py --dataset /path/to/kitti/dataset/ --predictions /path/to/method_predictions --split train/valid/test # depending of desired split to evaluate
  ```

  or 

    ```sh
  $ ./evaluate_panoptic.py --dataset /path/to/kitti/dataset/ --predictions /path/to/method_predictions --split train/valid/test # depending of desired split to evaluate
  ```

  or for moving object segmentation

    ```sh
  $ ./evaluate_mos.py --dataset /path/to/kitti/dataset/ --predictions /path/to/method_predictions --split train/valid/test # depending of desired split to evaluate
  ```  

- In the same directory as the dataset

  ```
  /kitti/dataset/
            ├── poses
            └── sequences
                ├── 00
                │   ├── image_2
                │   ├── image_3
                │   ├── labels
                │   │     ├ 000000.label
                │   │     └ 000001.label
                │   ├── predictions
                │   │     ├ 000000.label
                │   │     └ 000001.label
                │   └── velodyne
                │         ├ 000000.bin
                │         └ 000001.bin
                ├── 01
                ├── 02
                .
                .
                .
                └── 21
  ```

  And run (which sets the predictions directory as the same directory as the dataset):

  ```sh
  $ ./evaluate_semantics.py --dataset /path/to/kitti/dataset/ --split train/valid/test # depending of desired split to evaluate
  ```

If instead, the IoU vs distance is wanted, the evaluation is performed in the
same way, but with the [evaluate_semantics_by_distance.py](./evaluate_semantics_by_distance.py) script. This will
analyze the IoU for a set of 5 distance ranges: `{(0m:10m), [10m:20m), [20m:30m), [30m:40m), (40m:50m)}`. 

#### Validation

To ensure that your zip file is valid, we provide a small validation script [validate_submission.py](./validate_submission.py) that checks for the correct folder structure and consistent number of labels for each scan.

The submission folder expects to get an zip file containing the following folder structure (as the separate case above)

  ```
  ├ description.txt (optional)
  sequences
    ├── 11
    │   └── predictions
    │         ├ 000000.label
    │         ├ 000001.label
    │         ├ ...
    ├── 12
    │   └── predictions
    │         ├ 000000.label
    │         ├ 000001.label
    │         ├ ...
    ├── 13
    .
    .
    .
    └── 21
  ```

In summary, you only have to provide the label files containing your predictions for every point of the scan and this is also checked by our validation script.

Run:
  ```sh
  $ ./validate_submission.py --task {segmentation|completion|panoptic} /path/to/submission.zip /path/to/kitti/dataset
  ```
to check your `submission.zip`.

***Note:*** We don't check if the labels are valid, since invalid labels are simply ignored by the evaluation script.

#### (New!) Adding Approach Information

If you want to have more information on the leaderboard in the new updated Codalab competitions under the "Detailed Results", you have to provide an additional `description.txt` file to the submission archive containing information (here just an example):

```
name: Auto-MOS
pdf url: https://arxiv.org/pdf/2201.04501.pdf
code url: https://github.com/PRBonn/auto-mos
```

where `name` corresponds to the name of the method, `pdf url` is a link to the paper pdf url (or empty), and `code url` is a url that directs to the code (or empty). If the information is not available, we will use `Anonymous` for the name, and `n/a` for the urls.



#### Statistics

- [content.py](content.py) allows to evaluate the class content of the training
set, in order to weigh the loss for training, handling imbalanced data.
- [count.py](count.py) returns the scan count for each sequence in the data.

#### Generation

- [generate_sequential.py](generate_sequential.py) generates a sequence of scans using the manually looped closed poses used in our labeling tool, and stores them as individual point clouds. If, for example, we want to generate a dataset containing, for each point cloud, the aggregation of itself with the previous 4 scans, then:

  ```sh
  $ ./generate_sequential.py --dataset /path/to/kitti/dataset/ --sequence_length 5 --output /path/to/put/new/dataset 
  ```
  
- [remap_semantic_labels.py](remap_semantic_labels.py) allows to remap the labels
to and from the cross-entropy format, so that the labels can be used for training,
and the predictions can be used for evaluation. This file uses the `learning_map` and
`learning_map_inv` dictionaries from the config file to map the labels and predictions.

## Docker for API

If not installing the requirements is preferred, then a [docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/) container is
provided to run the scripts.

To build and run the container in an interactive session, which allows to run
X11 apps (and GL), and copies this repo to the working directory, use

```
$ ./docker.sh /path/to/dataset
```

Where `/path/to/dataset` is the location of your semantic kitti dataset, and
will be available inside the image in `~/data` or `/home/developer/data` 
inside the container for further usage with the api. This is done by creating
a shared volume, so it can be any directory containing data that is to be used
by the API scripts.

## Citation:

If you use this dataset and/or this API in your work, please cite its [paper](https://arxiv.org/abs/1904.01416)

```
@inproceedings{behley2019iccv,
    author = {J. Behley and M. Garbade and A. Milioto and J. Quenzel and S. Behnke and C. Stachniss and J. Gall},
     title = {{SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences}},
 booktitle = {Proc. of the IEEE/CVF International Conf.~on Computer Vision (ICCV)},
      year = {2019}
}
```

And the paper for the [original KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php):

```
@inproceedings{geiger2012cvpr,
    author = {A. Geiger and P. Lenz and R. Urtasun},
     title = {{Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite}},
 booktitle = {Proc.~of the IEEE Conf.~on Computer Vision and Pattern Recognition (CVPR)},
     pages = {3354--3361},
      year = {2012}}
```
