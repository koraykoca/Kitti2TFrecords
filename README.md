# Kitti2TFrecords
Multiprocessing script for conversion of KITTI dataset to Tensorflow records.

This script is to extract and encode data related to pointcloud, image, calibration and label information as tfrecords. One tfrecord includes 128 samples and has size of around 320 MB. You can specify the number of samples in one tfrecord and you can choose the number of training samples to be encoded. All configuration parameters of the script:
https://github.com/koraykoca/Kitti2TFrecords/blob/7979b1e1352e780b3aee3ec7d39753046639ea66/kitti2tfrecords.py#L10-L18
The 3D bounding box elements are encoded in this order:
```python
[x, y, z, length, width, height, rotation]
```
The order can be easily modified in the function [get_bbox3d()](https://github.com/koraykoca/Kitti2TFrecords/blob/7979b1e1352e780b3aee3ec7d39753046639ea66/tfrecordsutils.py#L55).

- The converted dataset can be loaded using [TFRecord dataset](https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset).
- An example/test function [convert_from_tfrecords()](https://github.com/koraykoca/Kitti2TFrecords/blob/7979b1e1352e780b3aee3ec7d39753046639ea66/kitti2tfrecords.py#L333) is also available to parse the encoded tfrecord data.  

## Usage
1) Download the dataset from https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d
* Velodyne point clouds (29 GB): Used as input laser information
* Training labels of the object data set (5 MB): Used as input label
* Camera calibration matrices of the object data set (16 MB): For synchronizing images and point clouds, for cropping point clouds, for projection from camera to point cloud coordinate, for visualizing the predictions
* Left color images of the object data set (12 GB): For cropping point clouds, for projection from camera to point cloud coordinate, for visualizing the predictions
 
2) Unzip the files into a folder. Write their directories into the kitti2tfrecord.py and run the script to convert the dataset into TF records!

NOTE: You can crop the point cloud data using [this script](https://github.com/qianguih/voxelnet/blob/master/data/crop.py), because the point clouds are scanned in 360 degrees while the RGB cameras are not (they have a much narrower field of view). In addition, KITTI only provides labels for objects that are within the images. Therefore, we usually need to remove points outside the image coordinates. If you convert the cropped data, then one tfrecord will be around 125 MB. 
