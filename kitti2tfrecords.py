import os
import random
from multiprocessing import Pool

import numpy as np
import tensorflow as tf
import tfrecordsutils as utils


_SOURCE_FOLDER = "/data/datasets/KITTI"
_DESTINATION_FOLDER = "/data/datasets/KITTI_tfrecords-cropped-lw"
_TRAINING_SUBFOLDERS = ["velodyne", "label_2", "image_2", "calib"]
_TESTING_SUBFOLDERS = ["velodyne", "image_2", "calib"]
_FRAMES_PER_TFRECORD = 128
_SAMPLES_FOR_TRAINING = 3713
_CONVERT_TESTING_SAMPLES = True
_OBJ_TYPE_MAP = {"Car": 1, "Pedestrian": 2, "Cyclist": 4}
_VISIBLE_GPUS = [0]


def parse_point_cloud(file_path):
    """(x, y, z, r), where (x, y, z) is the 3D coordinates
    and r is the reflectance value (referred as intensity in Waymo)
    """
    pc_data = np.fromfile(file_path, "<f4")
    pc_data = pc_data.reshape((-1, 4))
    pc_data_xyz = pc_data[:, :3].flatten().tolist()
    pc_data_reflectance = pc_data[:, -1].flatten().tolist()
    pc_data_features = {
        "LiDAR/point_cloud/num_valid_points": utils.int64_feature(pc_data.shape[0]),
        "LiDAR/point_cloud/xyz": utils.float_feature_list(pc_data_xyz),
        "LiDAR/point_cloud/obj_reflectance": utils.float_feature_list(
            pc_data_reflectance
        ),
    }
    return pc_data_features


def parse_labels(file_path, tr_velo_to_cam, matrix_rectification):
    """Extracts relevant information from label file
    0     -> Object type
    1     -> from 0 (non-truncated) to 1 (truncated), where truncated
             refers to the object leaving image boundaries
    2     -> 0 = fully visible, 1 = partly occluded,
             2 = largely occluded, 3 = unknown
    3     -> Observation angle of object [-pi..pi]
    4:7   -> 2D bounding box of object in the image,
             contains left, top, right, bottom pixel coordinates
    8:10  -> 3D object dimensions: height, width, length (in meters)
    11:13 -> The bottom center location x, y, z of the 3D object
             in camera coordinates (in meters)
    14    -> Rotation ry around Y-axis in camera coordinates [-pi..pi]

    Creates 3D bounding box label which contains
    [center (x, y, z), length, width, height, heading]
    """
    with open(file_path) as f:
        lines = f.readlines()

    class_ids_list = []
    truncations_list = []
    occlusions_list = []
    observation_angles_list = []
    bboxs_list = []
    bboxs3D_list = []
    dimensions_list = []
    centers_cam_list = []
    rotations_list = []

    for line in lines:
        obj_label = line.strip().split()
        obj_type = obj_label[0]

        if obj_type in _OBJ_TYPE_MAP:
            class_id = _OBJ_TYPE_MAP[obj_type]
            truncated = float(obj_label[1])
            occluded = int(obj_label[2])
            alpha = float(obj_label[3])
            bbox_coords = np.array(
                [obj_label[4], obj_label[5], obj_label[6], obj_label[7]]
            ).astype(float)
            dimension = np.array([obj_label[8], obj_label[9], obj_label[10]]).astype(
                float
            )
            center_cam = np.array([obj_label[11], obj_label[12], obj_label[13]]).astype(
                float
            )
            rotation = float(obj_label[14])

            bbox_3d_lidar = utils.get_bbox3d(center_cam, rotation, dimension, tr_velo_to_cam, matrix_rectification)

            class_ids_list.append(class_id)
            truncations_list.append(truncated)
            occlusions_list.append(occluded)
            observation_angles_list.append(alpha)
            bboxs_list.append(bbox_coords)
            bboxs3D_list.append(bbox_3d_lidar)
            dimensions_list.append(dimension)
            centers_cam_list.append(center_cam)
            rotations_list.append(rotation)

    num_obj = len(lines)
    num_valid_labels = len(class_ids_list)
    class_ids = np.array(class_ids_list, dtype=np.int64).flatten().tolist()
    truncated = np.array(truncations_list, dtype=np.float32).flatten().tolist()
    occluded = np.array(occlusions_list, dtype=np.int64).flatten().tolist()
    alpha = np.array(observation_angles_list, dtype=np.float32).flatten().tolist()
    bbox = np.array(bboxs_list, dtype=np.float32).flatten().tolist()
    bbox3D = np.array(bboxs3D_list, dtype=np.float32).flatten().tolist()
    dimensions = np.array(dimensions_list, dtype=np.float32).flatten().tolist()
    center_cam = np.array(centers_cam_list, dtype=np.float32).flatten().tolist()
    rotation_y = np.array(rotations_list, dtype=np.float32).flatten().tolist()

    labels_feature_dict = {
        "LiDAR/labels/num_valid_labels": utils.int64_feature(num_valid_labels),
        "LiDAR/labels/num_obj": utils.int64_feature(num_obj),
        "LiDAR/labels/class_ids": utils.int64_feature_list(class_ids),
        "LiDAR/labels/obj_truncated": utils.float_feature_list(truncated),
        "LiDAR/labels/obj_occluded": utils.int64_feature_list(occluded),
        "LiDAR/labels/obj_alpha": utils.float_feature_list(alpha),
        "LiDAR/labels/obj_bbox": utils.float_feature_list(bbox),
        "LiDAR/labels/box_3d": utils.float_feature_list(bbox3D),
        "LiDAR/labels/obj_dimensions": utils.float_feature_list(dimensions),
        "LiDAR/labels/obj_center_cam": utils.float_feature_list(center_cam),
        "LiDAR/labels/obj_rotation_y": utils.float_feature_list(rotation_y),
    }

    return labels_feature_dict


def parse_image(file_path):
    img_data = tf.io.decode_png(
        tf.io.read_file(file_path), channels=3, dtype=tf.dtypes.uint8
    )
    file_idx = int(file_path[-10:-4])
    img_shape = img_data.shape
    img_height = img_data.shape[0]
    img_width = img_data.shape[1]
    img_channels = img_data.shape[2]
    img_feature_dict = {
        "camera/image": utils.image_feature(img_data),
        "camera/image/file_idx": utils.int64_feature(file_idx),
        "camera/image/shape": utils.int64_feature_list(img_shape),
        "camera/image/height": utils.int64_feature(img_height),
        "camera/image/width": utils.int64_feature(img_width),
        "camera/image/channels": utils.int64_feature(img_channels),
    }

    return img_feature_dict


def parse_calib(file_path):
    with open(file_path) as f:
        lines = f.readlines()

    matrix_proj_0 = np.array(lines[0].strip().split(":")[1].split(), dtype=np.float32)
    matrix_proj_1 = np.array(lines[1].strip().split(":")[1].split(), dtype=np.float32)
    matrix_proj_2 = np.array(lines[2].strip().split(":")[1].split(), dtype=np.float32)
    matrix_proj_3 = np.array(lines[3].strip().split(":")[1].split(), dtype=np.float32)
    matrix_rectification = np.array(lines[4].strip().split(":")[1].split(), dtype=np.float32)
    matrix_tr_velo_to_cam = np.array(lines[5].strip().split(":")[1].split(), dtype=np.float32)
    matrix_tr_imu_to_velo = np.array(lines[6].strip().split(":")[1].split(), dtype=np.float32)

    matrix_tr_velo_to_cam = np.vstack((matrix_tr_velo_to_cam.reshape(3,4), [0., 0., 0., 1.]))
    matrix_proj_2 = np.vstack((matrix_proj_2.reshape(3, 4), [0., 0., 0., 0.]))
    R_cam_to_rect = np.eye(4)
    R_cam_to_rect[:3, :3] = matrix_rectification.reshape(3, 3)

    calib_feature_dict = {
        "calib/matrix_proj_0": utils.float_feature_list(matrix_proj_0.flatten().tolist()),
        "calib/matrix_proj_1": utils.float_feature_list(matrix_proj_1.flatten().tolist()),
        "calib/matrix_proj_2": utils.float_feature_list(matrix_proj_2.flatten().tolist()),
        "calib/matrix_proj_3": utils.float_feature_list(matrix_proj_3.flatten().tolist()),
        "calib/matrix_rectification": utils.float_feature_list(R_cam_to_rect.flatten().tolist()),
        "calib/matrix_tr_velo_to_cam": utils.float_feature_list(
            matrix_tr_velo_to_cam.flatten().tolist()
        ),
        "calib/matrix_tr_imu_to_velo": utils.float_feature_list(matrix_tr_imu_to_velo.flatten().tolist()),
    }

    return calib_feature_dict, matrix_tr_velo_to_cam, R_cam_to_rect


def convert_to_tfrecord(data, sample_type, destination_dir):
    if sample_type == "training":
        pc_dir, label_dir, img_dir, calib_dir = data.keys()
        pcs, labels, imgs, calibs = data.values()
        validation_dir = os.path.join(_DESTINATION_FOLDER, "validation")
        os.makedirs(validation_dir, exist_ok=True)
    elif sample_type == "testing":
        pc_dir, img_dir, calib_dir = data.keys()
        pcs, imgs, calibs = data.values()
        labels = None
    else:
        raise ValueError("Only samples in training or testing folder can be converted.")

    tf_index = 0
    example_counter = 0
    example_proto_list = []
    # [None]*len(pcs) is a placeholder for labels when sample_type is "testing"
    # so zip function can work properly for both cases
    for pc, label, img, calib in zip(pcs, labels or [None] * len(pcs), imgs, calibs):

        example_counter += 1
        point_cloud_features = parse_point_cloud(os.path.join(pc_dir, pc))
        calibration_features, matrix_tr_imu_to_velo, R_cam_to_rect = parse_calib(
            os.path.join(calib_dir, calib)
        )
        label_features = {}
        if sample_type == "training":
            label_features = parse_labels(
                os.path.join(label_dir, label), 
                matrix_tr_imu_to_velo,
                R_cam_to_rect
            )
        image_fatures = parse_image(os.path.join(img_dir, img))

        final_features = {
            **point_cloud_features,
            **label_features,
            **image_fatures,
            **calibration_features,
        }

        # Create an example protocol buffer
        example_proto = tf.train.Example(
            features=tf.train.Features(feature=final_features)
        )
        example_proto_list.append(example_proto)
        print(f"Number of encoded {sample_type} samples: {example_counter}", end="\r")

        if sample_type == "training" and example_counter == len(pcs):
            random.shuffle(example_proto_list)
            training_examples = example_proto_list[:_SAMPLES_FOR_TRAINING]
            val_examples = example_proto_list[_SAMPLES_FOR_TRAINING:]
            to_be_written = []
            for idx, train_example in enumerate(training_examples, 1):
                to_be_written.append(train_example)
                if idx % _FRAMES_PER_TFRECORD == 0 or idx == len(training_examples):
                    destination_train = os.path.join(
                        destination_dir, f"{tf_index}.tfrecord"
                    )
                    with tf.io.TFRecordWriter(destination_train) as writer:
                        # Serialize to string and write on the file
                        for example in to_be_written:
                            writer.write(example.SerializeToString())
                        print(
                            f"{len(to_be_written)} training samples encoded"
                            f" and saved to {destination_train}"
                        )
                    to_be_written = []
                    tf_index += 1
            print(
                f"{'-'*100}\n {len(training_examples)} training samples encoded"
                f" and saved to {destination_dir}\n{'-'*100}"
            )
            tf_index = 0
            val_split_file = os.path.join(_DESTINATION_FOLDER, "val_split.txt")
            if os.path.exists(val_split_file):
                os.remove(val_split_file)
            for idx, val_example in enumerate(val_examples, 1):
                to_be_written.append(val_example)
                if idx % _FRAMES_PER_TFRECORD == 0 or idx == len(val_examples):
                    destination_val = os.path.join(
                        validation_dir, f"{tf_index}.tfrecord"
                    )
                    with tf.io.TFRecordWriter(destination_val) as writer:
                        # Serialize to string and write on the file
                        for example in to_be_written:
                            writer.write(example.SerializeToString())
                        print(
                            f"{len(to_be_written)} validation samples encoded"
                            f" and saved to {destination_val}"
                        )
                    to_be_written = []
                    tf_index += 1
            # Create a txt file which includes validation sample numbers
            # It is required for KITTI evaluation
            with open(val_split_file, 'w') as f:
                for example in val_examples:
                    file_idx = example.features.feature["camera/image/file_idx"].int64_list.value[0]
                    f.write(f"{file_idx:06}\n")    
            print(
                f"{'-'*100}\n {len(val_examples)} validation samples encoded"
                f" and saved to {validation_dir}\n{'-'*100}"
            )
        elif sample_type == "testing":
            if example_counter % _FRAMES_PER_TFRECORD == 0 or example_counter == len(
                pcs
            ):
                destination_test = os.path.join(destination_dir, f"{tf_index}.tfrecord")
                with tf.io.TFRecordWriter(destination_test) as writer:
                    # Serialize to string and write on the file
                    for example in example_proto_list:
                        writer.write(example.SerializeToString())
                    print(
                        f"{len(example_proto_list)} {sample_type} samples encoded"
                        f" and saved to {destination_test}"
                    )
                example_proto_list = []
                tf_index += 1
                if example_counter == len(pcs):
                    print(
                        f"{'-'*100}\n {example_counter} {sample_type} samples encoded"
                        f" and saved to {destination_dir}\n{'-'*100}"
                    )


def convert_to_tfrecords(source_folder, destination_folder):
    sample_folders = ["training"]
    if _CONVERT_TESTING_SAMPLES:
        sample_folders.append("testing")

    for folder in sample_folders:
        destination_dir = os.path.join(destination_folder, folder)
        os.makedirs(destination_dir, exist_ok=True)

        if folder == "training":
            subfolders = _TRAINING_SUBFOLDERS
        else:
            subfolders = _TESTING_SUBFOLDERS

        data = dict()
        for subfolder in subfolders:
            source_sub_folder = os.path.join(source_folder, folder, subfolder)

            files = sorted(os.listdir(source_sub_folder))
            data[source_sub_folder] = files

        convert_to_tfrecord(data, folder, destination_dir)


def convert_from_tfrecords(source_dir):
    """This function is to test the conversions from KITTI to tfrecords"""
    import matplotlib.pyplot as plt

    def parse_tfrecord(record):
        feature_description = {
            "LiDAR/point_cloud/num_valid_points": tf.io.FixedLenFeature([1], tf.int64),
            "LiDAR/point_cloud/xyz": tf.io.VarLenFeature(tf.float32),
            "LiDAR/point_cloud/obj_reflectance": tf.io.VarLenFeature(tf.float32),
            "LiDAR/labels/num_valid_labels": tf.io.FixedLenFeature([1], tf.int64),
            "LiDAR/labels/num_obj": tf.io.VarLenFeature(tf.int64),
            "LiDAR/labels/class_ids": tf.io.VarLenFeature(tf.int64),
            "LiDAR/labels/obj_truncated": tf.io.VarLenFeature(tf.float32),
            "LiDAR/labels/obj_occluded": tf.io.VarLenFeature(tf.int64),
            "LiDAR/labels/obj_alpha": tf.io.VarLenFeature(tf.float32),
            "LiDAR/labels/obj_bbox": tf.io.VarLenFeature(tf.float32),
            "LiDAR/labels/box_3d": tf.io.VarLenFeature(tf.float32),
            "LiDAR/labels/obj_dimensions": tf.io.VarLenFeature(tf.float32),
            "LiDAR/labels/obj_center_cam": tf.io.VarLenFeature(tf.float32),
            "LiDAR/labels/obj_rotation_y": tf.io.VarLenFeature(tf.float32),
            # image is a list with ONE element of type "bytes". "bytes" type
            # can itself have variable size (it's a string of bytes, and can
            # have many symbols within it).
            "camera/image": tf.io.FixedLenFeature([], tf.string),
            "camera/image/file_idx": tf.io.FixedLenFeature([1], tf.int64),
            "camera/image/shape": tf.io.FixedLenFeature([3], tf.int64),
            "camera/image/height": tf.io.FixedLenFeature([1], tf.int64),
            "camera/image/width": tf.io.FixedLenFeature([1], tf.int64),
            "camera/image/channels": tf.io.FixedLenFeature([1], tf.int64),
            "calib/matrix_proj_0": tf.io.FixedLenFeature([12], tf.float32),
            "calib/matrix_proj_1": tf.io.FixedLenFeature([12], tf.float32),
            "calib/matrix_proj_2": tf.io.FixedLenFeature([12], tf.float32),
            "calib/matrix_proj_3": tf.io.FixedLenFeature([12], tf.float32),
            "calib/matrix_rectification": tf.io.FixedLenFeature([9], tf.float32),
            "calib/matrix_tr_velo_to_cam": tf.io.FixedLenFeature([12], tf.float32),
            "calib/matrix_tr_imu_to_velo": tf.io.FixedLenFeature([12], tf.float32),
        }
        example = tf.io.parse_single_example(record, feature_description)
        example["camera/image"] = tf.io.decode_png(
            example["camera/image"], channels=3, dtype=tf.dtypes.uint8
        )
        # Convert sparse tensors to dense tensor, because data is actually dense
        example["LiDAR/labels/class_ids"] = tf.sparse.to_dense(
            example["LiDAR/labels/class_ids"]
        )
        example["LiDAR/point_cloud/xyz"] = tf.reshape(tf.sparse.to_dense(
            example["LiDAR/point_cloud/xyz"]), (-1, 3)
        )
        example["LiDAR/point_cloud/obj_reflectance"] = tf.sparse.to_dense(
            example["LiDAR/point_cloud/obj_reflectance"]
        )
        example["LiDAR/labels/obj_dimensions"] = tf.sparse.to_dense(
            example["LiDAR/labels/obj_dimensions"]
        )
        example["LiDAR/labels/obj_bbox"] = tf.sparse.to_dense(
            example["LiDAR/labels/obj_bbox"]
        )
        example["LiDAR/labels/box_3d"] = tf.sparse.to_dense(
            example["LiDAR/labels/box_3d"]
        )
        example["LiDAR/labels/obj_center_cam"] = tf.sparse.to_dense(
            example["LiDAR/labels/obj_center_cam"]
        )
        example["calib/matrix_tr_velo_to_cam"] = example["calib/matrix_tr_velo_to_cam"]
        example["camera/image/file_idx"] = example["camera/image/file_idx"]
        return example

    raw_dataset = tf.data.TFRecordDataset(source_dir)
    parsed_dataset = raw_dataset.map(parse_tfrecord)

    num_samples = parsed_dataset.reduce(0, lambda x, _: x + 1)
    print(f"Number of training samples in the tfrecord: {num_samples.numpy()}")

    for example in parsed_dataset.take(5):
        for key in example.keys():
            if key != "camera/image":
                print(f"{key}: {example[key]}")

    print(f"Image shape: {example['camera/image'].shape}")
    plt.figure(figsize=(7, 7))
    plt.imshow(example["camera/image"].numpy())
    plt.savefig(
        "/data/code/datasets/tfrecord/scripts/KITTI/test_tf_record.png", format="png"
    )


if __name__ == "__main__":
    physical_devices_GPU = tf.config.list_physical_devices("GPU")
    if len(physical_devices_GPU):
        int_list_specify_visible_devices = [
            physical_devices_GPU[eval(str(i))] for i in _VISIBLE_GPUS
        ]
        for device in int_list_specify_visible_devices:
            tf.config.experimental.set_memory_growth(device, enable=True)
        tf.config.set_visible_devices(int_list_specify_visible_devices, "GPU")

    with Pool() as pool:
        convert_to_tfrecords(_SOURCE_FOLDER, _DESTINATION_FOLDER)

    # test a tfrecord
    # convert_from_tfrecords(
    #     "/data/datasets/KITTI_tfrecords-cropped-lw/validation/0.tfrecord"
    # )
