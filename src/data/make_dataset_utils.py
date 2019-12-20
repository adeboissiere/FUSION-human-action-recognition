r"""
This module creates different h5 files that contain the data provided by NTU RGB+D in numpy ready format.

The following functions are provided.

    - *create_h5_2d_ir_skeleton*: Creates h5 with 2D IR skeleton data
    - *create_h5_skeleton_dataset:* Creates h5 with 3D skeleton data
    - *create_h5_ir_dataset*: Creates h5 with raw IR sequences
    - *create_h5_ir_cropped_dataset_from_h5*: Creates h5 containing cropped IR sequences around the subjects with a
      fixed bounding box. Requires *create_h5_ir_dataset* and *create_h5_2d_ir_skeleton* to be run first.
    - *create_h5_ir_cropped_moving_dataset_from_h5*: Creates h5 containing cropped IR sequences around the subjects
      with a moving bounding box. Requires *create_h5_ir_dataset* and *create_h5_2d_ir_skeleton* to be run first.

"""

import cv2
import h5py
import os
import skvideo.io
import skvideo.datasets

from click import progressbar

# Custom modules
from src.data.read_NTU_RGB_D_skeleton import *
from src.utils.joints import *


def create_h5_2d_ir_skeleton(input_path, output_path, compression="", compression_opts=9):
    r"""Creates an h5 dataset of the 2D skeleton projected on the IR frames.
    For each sequence, a new group with the name of the sequence, **SsssCcccPpppRrrrAaaa**, is created.
    In each group, a new dataset is created containing the 2D skeleton data.
    The skeleton data is of shape `(2 {x, y}, max_frame, num_joint, 2 {n_subjects})`

    The h5 may be used as a standalone but is necessary to create the processed IR h5 files (see below).

    The method creates the file "ir_skeleton.h5". **Warning:** The file should not be renamed!


    Inputs:
        - **input_path** (str): Path containing the raw NTU files (default: *./data/raw/.*
          See **Project Organization** in *README.md*)
        - **output_path** (str): Path containing the processed h5 files (default: *./data/processed/.*
          See **Project Organization** in *README.md*)
        - **compression** (str): Compression type for h5. May take values in ["", "lzf", "gzip"]
        - **compression_otps** (int): Compression opts. For "gzip" compression only.
          May take values in the [0; 9] range.

    """

    # Folder containing raw skeleton files (input_path + skeleton_folder)
    skeleton_folder = "nturgb+d_skeletons/"

    # Create a log file to track and debug progress
    open_type = "w"
    file = open(output_path + 'log.txt', 'w')
    file.close()

    # Create h5 file
    with h5py.File(output_path + 'ir_skeleton.h5', open_type) as hdf:
        # Progress bar
        progress_bar = progressbar(iterable=None,
                                   length=len(next(os.walk(input_path + skeleton_folder))[2])
                                   )

        # Loop through skeleton files
        for filename in os.listdir(input_path + skeleton_folder):
            # Sequence name (ie. S001C002P003R004A005)
            sequence_name = os.path.splitext(filename)[0]

            # Retrieve skeleton data of shape (2, max_frame, num_joint, n_subjects)
            skeleton = read_xy_ir(input_path + skeleton_folder + filename)

            # Log current sequence
            f = open(output_path + "log.txt", "a+")
            f.write(sequence_name)
            f.write("\r\n")
            f.close()

            # Create a group for the current sequence
            sample = hdf.create_group(sequence_name)

            # Create a dataset with the skeleton data
            if compression == "":
                sample.create_dataset("ir_skeleton", data=skeleton)
            elif compression == "lzf":
                sample.create_dataset("ir_skeleton", data=skeleton, compression=compression)
            elif compression == "gzip":
                sample.create_dataset("ir_skeleton", data=skeleton,
                                      compression=compression,
                                      compression_opts=compression_opts)
            else:
                print("Compression type not recognized ... Exiting")
                return

            progress_bar.update(1)


def create_h5_skeleton_dataset(input_path, output_path, compression="", compression_opts=9):
    r"""Creates an h5 dataset of the 3D skeleton data.
    For each sequence, a new group with the name of the sequence, **SsssCcccPpppRrrrAaaa**, is created.
    In each group, a new dataset is created containing the 3D skeleton data.
    The skeleton data is of shape `(3 {x, y, z}, max_frame, num_joint, 2 {n_subjects})`

    The method creates the file "skeleton.h5". **Warning:** The file should not be renamed!

    Inputs:
        - **input_path** (str): Path containing the raw NTU files (default: *./data/raw/*.
          See **Project Organization** in *README.md*)
        - **output_path** (str): Path containing the processed h5 files (default: *./data/processed/*.
          See **Project Organization** in *README.md*)
        - **compression** (str): Compression type for h5. May take values in ["", "lzf", "gzip"]
        - **compression_otps** (int): Compression opts. For "gzip" compression only.
          May take values in the [0; 9] range.

    """

    # Folder containing raw skeleton files (input_path + skeleton_folder)
    skeleton_folder = "nturgb+d_skeletons/"

    # Create a log file to track and debug progress
    open_type = "w"
    file = open(output_path + 'log.txt', 'w')
    file.close()

    # Create h5 file
    with h5py.File(output_path + 'skeleton.h5', open_type) as hdf:
        # Progress bar
        progress_bar = progressbar(iterable=None,
                                   length=len(next(os.walk(input_path + skeleton_folder))[2])
                                   )

        # Loop through skeleton files
        for filename in os.listdir(input_path + skeleton_folder):
            # Sequence name (ie. S001C002P003R004A005)
            sequence_name = os.path.splitext(filename)[0]

            # Retrieve skeleton data of shape (2, max_frame, num_joint, n_subjects)
            skeleton = read_xyz(input_path + skeleton_folder + filename)

            # Log current sequence
            f = open(output_path + "log.txt", "a+")
            f.write(sequence_name)
            f.write("\r\n")
            f.close()

            # Create a group for the current sequence
            sample = hdf.create_group(sequence_name)

            # Create a dataset with the skeleton data
            if compression == "":
                sample.create_dataset("skeleton", data=skeleton)
            elif compression == "lzf":
                sample.create_dataset("skeleton", data=skeleton, compression=compression)
            elif compression == "gzip":
                sample.create_dataset("skeleton", data=skeleton,
                                      compression=compression,
                                      compression_opts=compression_opts)
            else:
                print("Compression type not recognized ... Exiting")
                return

            progress_bar.update(1)


def create_h5_ir_dataset(input_path, output_path, compression="", compression_opts=9):
    r"""Creates an h5 dataset of the unprocessed IR sequences.
    For each sequence, a new group with the name of the sequence, **SsssCcccPpppRrrrAaaa**, is created.
    In each group, a new dataset is created containing the unprocessed IR sequence.
    The IR video data is of shape `(n_frames, H, W)`.

    The h5 may be used as a standalone but is necessary to create the processed IR h5 files (see below).

    The method creates the file "ir.h5". **Warning:** The file should not be renamed!

    Inputs:
        - **input_path** (str): Path containing the raw NTU files (default: *./data/raw/*.
          See **Project Organization** in *README.md*)
        - **output_path** (str): Path containing the processed h5 files (default: *./data/processed/*.
          See **Project Organization** in *README.md*)
        - **compression** (str): Compression type for h5. May take values in ["", "lzf", "gzip"]
        - **compression_otps** (int): Compression opts. For "gzip" compression only.
          May take values in the [0; 9] range.

    """

    # Folder containing raw IR files (input_path + ir_folder)
    ir_folder = "nturgb+d_ir/"

    # Create a log file to track and debug progress
    open_type = "w"
    file = open(output_path + 'log.txt', 'w')
    file.close()

    # Create h5 file
    with h5py.File(output_path + 'ir.h5', open_type) as hdf:
        # Progress bar
        progress_bar = progressbar(iterable=None,
                                   length=len(next(os.walk(input_path + ir_folder))[2])
                                   )

        # Loop through skeleton files
        for filename in os.listdir(input_path + ir_folder):
            # Sequence name (ie. S001C002P003R004A005)
            sequence_name = os.path.splitext(filename)[0][:-3]

            # Log current sequence
            f = open(output_path + "log.txt", "a+")
            f.write(sequence_name)
            f.write("\r\n")
            f.close()
            # print(short_filename)

            # Read corresponding video
            video_data = skvideo.io.vread(
                input_path + ir_folder + filename)[:, :, :, 0]  # shape (n_frames, H, W)

            # Get video dimensions
            _, H, W = video_data.shape

            # Create a group for the current sequence
            sample = hdf.create_group(sequence_name)

            # Create a dataset with the skeleton data
            if compression == "":
                sample.create_dataset("ir", data=video_data, chunks=(1, H, W))
            elif compression == "lzf":
                sample.create_dataset("ir", data=video_data, compression=compression, chunks=(1, H, W))
            elif compression == "gzip":
                sample.create_dataset("ir", data=video_data,
                                      compression=compression,
                                      compression_opts=compression_opts,
                                      chunks=(1, H, W))
            else:
                print("Compression type not recognized ... Exiting")
                return

            progress_bar.update(1)


def create_h5_ir_cropped_dataset_from_h5(input_path, output_path, compression="", compression_opts=9):
    r"""Creates an h5 dataset with processed IR sequences.
    The frames are cropped with a bounding box provided by the 2D IR skeleton.
    The bounding box is fixed across all frames.
    For each sequence, a new group with the name of the sequence, **SsssCcccPpppRrrrAaaa**, is created.
    In each group, a new dataset is created containing the unprocessed IR sequence.
    The IR video data is of shape `(n_frames, H, W)`.

    This method depends on the h5 datasets (ir.h5, ir_skeleton.h5) created by the corresponding methods.

    The method creates the file "ir_cropped.h5". **Warning:** The file should not be renamed!

    Inputs:
        - **input_path** (str): Path containing the processed h5 files (default: *./data/processed/*.
          See **Project Organization** in *README.md*)
        - **output_path** (str): Path containing the processed h5 files (default: *./data/processed/*.
          See **Project Organization** in *README.md*)
        - **compression** (str): Compression type for h5. May take values in ["", "lzf", "gzip"]
        - **compression_otps** (int): Compression opts. For "gzip" compression only.
          May take values in the [0; 9] range.

    """

    # Get samples list
    samples_names_list = [line.rstrip('\n') for line in open(input_path + "samples_names.txt")]

    # Existing h5 files
    ir_skeleton_dataset_file_name = "ir_skeleton.h5"
    ir_dataset_file_name = "ir.h5"

    # Offset around bounding box
    offset = 20

    # Create a log file to track and debug progress
    open_type = "w"
    file = open(output_path + 'log.txt', 'w')
    file.close()

    # Open existing h5 files
    ir_skeleton_dataset = h5py.File(input_path + ir_skeleton_dataset_file_name, 'r')
    ir_dataset = h5py.File(input_path + ir_dataset_file_name, 'r')

    # Create h5 file
    with h5py.File(output_path + 'ir_cropped.h5', open_type) as hdf:
        # Progress bar
        progress_bar = progressbar(iterable=None, length=len(samples_names_list))

        # Loop through skeleton files
        for sequence_name in samples_names_list:
            # Log current sequence
            f = open(output_path + "log.txt", "a+")
            f.write(sequence_name)
            f.write("\r\n")
            f.close()

            # Fetch corresponding ir raw sequence
            video_data = ir_dataset[sequence_name]["ir"][:]

            # Pad video to compensate for offset
            cropped_ir_sample = np.pad(video_data, ((0, 0), (offset, offset), (offset, offset)), mode='constant')

            # Get corresponding ir skeleton shape(2 : {y, x}, seq_len, n_joints, n_subjects)
            ir_skeleton = ir_skeleton_dataset[sequence_name]["ir_skeleton"][:].clip(min=0)

            # Check if there is another subject if there exists non zero coordinates for subject 2
            has_2_subjects = np.any(ir_skeleton[:, :, :, 1])

            # Calculate boundaries
            if not has_2_subjects:
                y_min = min(np.uint16(np.amin(ir_skeleton[0, :, :, 0])), video_data.shape[2])
                y_max = min(np.uint16(np.amax(ir_skeleton[0, :, :, 0])), video_data.shape[2])

                x_min = min(np.uint16(np.amin(ir_skeleton[1, :, :, 0])), video_data.shape[1])
                x_max = min(np.uint16(np.amax(ir_skeleton[1, :, :, 0])), video_data.shape[1])

            else:
                y_min = min(np.uint16(np.amin(ir_skeleton[0, :, :, :])), video_data.shape[2])
                y_max = min(np.uint16(np.amax(ir_skeleton[0, :, :, :])), video_data.shape[2])

                x_min = min(np.uint16(np.amin(ir_skeleton[1, :, :, :])), video_data.shape[1])
                x_max = min(np.uint16(np.amax(ir_skeleton[1, :, :, :])), video_data.shape[1])

            # Crop video
            cropped_ir_sample = cropped_ir_sample[:, x_min:x_max + 2 * offset, y_min:y_max + 2 * offset]

            # Get video dimensions
            _, H, W = cropped_ir_sample.shape

            # Create a group for the current sequence
            sample = hdf.create_group(sequence_name)

            # Create a dataset with the skeleton data
            if compression == "":
                sample.create_dataset("ir", data=cropped_ir_sample, chunks=(1, H, W))
            elif compression == "lzf":
                sample.create_dataset("ir", data=cropped_ir_sample, compression=compression, chunks=(1, H, W))
            elif compression == "gzip":
                sample.create_dataset("ir", data=cropped_ir_sample,
                                      compression=compression,
                                      compression_opts=compression_opts,
                                      chunks=(1, H, W))
            else:
                print("Compression type not recognized ... Exiting")
                return

            progress_bar.update(1)

    ir_skeleton_dataset.close()
    ir_dataset.close()


def create_h5_ir_cropped_moving_dataset_from_h5(input_path, output_path, compression="", compression_opts=9):
    r"""Creates an h5 dataset with processed IR sequences.
    The frames are cropped with a bounding box provided by the 2D IR skeleton.
    The bounding box is updated at every frame.
    For each sequence, a new group with the name of the sequence, **SsssCcccPpppRrrrAaaa**, is created.
    In each group, a new dataset is created containing the unprocessed IR sequence.
    The IR video data is of shape `(n_frames, H, W)`.

    This method depends on the h5 datasets (ir.h5, ir_skeleton.h5) created by the corresponding methods.

    The method creates the file "ir_cropped_moving.h5". **Warning:** The file should not be renamed!

    Inputs:
        - **input_path** (str): Path containing the processed h5 files (default: *./data/processed/.*
          See **Project Organization** in *README.md*)
        - **output_path** (str): Path containing the processed h5 files (default: *./data/processed/.*
          See **Project Organization** in *README.md*)
        - **compression** (str): Compression type for h5. May take values in ["", "lzf", "gzip"]
        - **compression_otps** (int): Compression opts. For "gzip" compression only.
          May take values in the [0; 9] range.

    """

    # Get samples list
    samples_names_list = [line.rstrip('\n') for line in open(input_path + "samples_names.txt")]

    # Existing h5 files
    ir_skeleton_dataset_file_name = "ir_skeleton.h5"
    ir_dataset_file_name = "ir.h5"

    # Offset around bounding box
    offset = 20

    # Create a log file to track and debug progress
    open_type = "w"
    file = open(output_path + 'log.txt', 'w')
    file.close()

    # Open existing h5 files
    ir_skeleton_dataset = h5py.File(input_path + ir_skeleton_dataset_file_name, 'r')
    ir_dataset = h5py.File(input_path + ir_dataset_file_name, 'r')

    # Create h5 file
    with h5py.File(output_path + 'ir_cropped_moving.h5', open_type) as hdf:
        # Progress bar
        progress_bar = progressbar(iterable=None, length=len(samples_names_list))

        # Loop through skeleton files
        for sequence_name in samples_names_list:
            # Log current sequence
            f = open(output_path + "log.txt", "a+")
            f.write(sequence_name)
            f.write("\r\n")
            f.close()

            # Fetch corresponding ir raw sequence shape (n_frames, H, W)
            video_data = ir_dataset[sequence_name]["ir"][:]

            # Get corresponding ir skeleton shape(2 : {y, x}, seq_len, n_joints, n_subjects)
            ir_skeleton = ir_skeleton_dataset[sequence_name]["ir_skeleton"][:].clip(min=0)

            # Check if there is another subject if there exists non zero coordinates for subject 2
            has_2_subjects = np.any(ir_skeleton[:, :, :, 1])

            # Calculate boundaries for each frame
            y_min = np.uint16(np.amin(ir_skeleton[0, :, :, 0], axis=1))
            y_max = np.uint16(np.amax(ir_skeleton[0, :, :, 0], axis=1))

            x_min = np.uint16(np.amin(ir_skeleton[1, :, :, 0], axis=1))
            x_max = np.uint16(np.amax(ir_skeleton[1, :, :, 0], axis=1))

            if has_2_subjects:
                y_min = np.minimum(y_min, np.uint16(np.amin(ir_skeleton[0, :, :, 1], axis=1)))
                y_max = np.maximum(y_max, np.uint16(np.amax(ir_skeleton[0, :, :, 1], axis=1)))

                x_min = np.minimum(x_min, np.uint16(np.amin(ir_skeleton[1, :, :, 1], axis=1)))
                x_max = np.maximum(x_max, np.uint16(np.amax(ir_skeleton[1, :, :, 1], axis=1)))

            # Clip to avoid pixel out of frame
            x_min.clip(max=video_data.shape[1])
            x_max.clip(max=video_data.shape[1])
            y_min.clip(max=video_data.shape[2])
            y_max.clip(max=video_data.shape[2])

            # Crop and scale ir video
            new_sequence = []
            for t in range(video_data.shape[0]):
                # Fetch individual frame
                frame = video_data[t]  # shape (H, W)

                # Pad frame with zeros (to compensate for offset)
                frame = np.pad(frame, ((offset, offset), (offset, offset)), mode='constant')

                # Crop frame
                frame = frame[x_min[t]:x_max[t] + 2 * offset,
                        y_min[t]:y_max[t] + 2 * offset]

                # Rescale frame
                ir_frame = cv2.resize(frame, dsize=(112, 112))
                new_sequence.append(ir_frame)

            new_sequence = np.stack(new_sequence, axis=0)  # shape (n_frames, 112, 112)

            # Get video dimensions
            _, H, W = new_sequence.shape

            # Create a group for the current sequence
            sample = hdf.create_group(sequence_name)

            # Create a dataset with the skeleton data
            if compression == "":
                sample.create_dataset("ir", data=new_sequence, chunks=(1, H, W))
            elif compression == "lzf":
                sample.create_dataset("ir", data=new_sequence, compression=compression, chunks=(1, H, W))
            elif compression == "gzip":
                sample.create_dataset("ir", data=new_sequence,
                                      compression=compression,
                                      compression_opts=compression_opts,
                                      chunks=(1, H, W))
            else:
                print("Compression type not recognized ... Exiting")
                return

            progress_bar.update(1)

    ir_skeleton_dataset.close()
    ir_dataset.close()
