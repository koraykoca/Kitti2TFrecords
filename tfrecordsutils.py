import numpy as np
import tensorflow as tf


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def image_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_png(value).numpy()])
    )


def pc_float_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten()))


def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_feature_list(value):
    """Returns a list of float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_feature_list(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def ry_to_rz(ry):
    """
    param ry (float): yaw angle in cam coordinate system
    return: (flaot): yaw angle in velodyne coordinate system
    """
    angle = -ry - np.pi / 2
    angle = tf.where(tf.greater_equal(angle, np.pi), angle - np.pi, angle)
    angle = tf.where(tf.less(angle, np.pi), 2 * np.pi + angle, angle)
    return angle


def get_bbox3d(obj_xyz_cam, rot_y, dimensions, tr_velo_to_cam, R_cam_to_rect):
    """returns 3D object location center (x, y, z)"""
    length = dimensions[2]
    width = dimensions[1]
    height = dimensions[0]
    rot_z = ry_to_rz(rot_y)

    # projection from camera coordinates to lidar coordinates
    obj_xyz_cam = np.vstack((obj_xyz_cam.reshape(3,1), [1]))
    rot_mat = np.linalg.inv(R_cam_to_rect @ tr_velo_to_cam)
    obj_xyz_lidar = rot_mat @ obj_xyz_cam
    obj_x = obj_xyz_lidar[0][0]
    obj_y = obj_xyz_lidar[1][0]
    obj_z = obj_xyz_lidar[2][0]

    return np.array([obj_x, obj_y, obj_z, length, width, height, rot_z])
