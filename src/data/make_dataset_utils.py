import cv2
import h5py
import os
import skvideo.io
import skvideo.datasets

from click import progressbar

# Custom modules
from src.utils.joints import *


def create_2d_ir_skeleton(input_path, output_path, compression="", compression_opts=9):
    """Creates an h5 dataset. Each group corresponds to a clip and contains the numpy array of the skeleton data and the
    numpy array of image crops around the hands

    :param input_path: NTU-RGB-D data path
    :param output_path: location of created h5 dataset
    :param compression: type of compression {"", "lzf", "gzip"}
    :param compression_opts: compression strength {1, .., 9}
    """

    skeleton_folder = "nturgb+d_skeletons/"

    open_type = "w"
    file = open(output_path + 'log.txt', 'w')
    file.close()

    with h5py.File(output_path + 'ir_skeleton.h5', open_type) as hdf:
        # Progress bar
        progress_bar = progressbar(iterable=None,
                                   length=len(next(os.walk(input_path + skeleton_folder))[2])
                                   )

        # Loop through skeleton files
        for filename in os.listdir(input_path + skeleton_folder):
            short_filename = os.path.splitext(filename)[0]

            # Retrieve skeleton data
            skeleton = read_ir_xy(
                input_path + skeleton_folder + filename)  # shape (2, max_frame, num_joint, n_subjects)

            # Sequence code without extension
            f = open(output_path + "log.txt", "a+")
            f.write(short_filename)
            f.write("\r\n")
            f.close()

            sample = hdf.create_group(short_filename)

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
    """Creates an h5 dataset. Each group corresponds to a clip and contains the numpy array of the skeleton data and the
    numpy array of image crops around the hands

    :param input_path: NTU-RGB-D data path
    :param output_path: location of created h5 dataset
    :param compression: type of compression {"", "lzf", "gzip"}
    :param compression_opts: compression strength {1, .., 9}
    """

    skeleton_folder = "nturgb+d_skeletons/"

    open_type = "w"
    file = open(output_path + 'log.txt', 'w')
    file.close()

    with h5py.File(output_path + 'skeleton.h5', open_type) as hdf:
        # Progress bar
        progress_bar = progressbar(iterable=None,
                                   length=len(next(os.walk(input_path + skeleton_folder))[2])
                                   )

        # Loop through skeleton files
        for filename in os.listdir(input_path + skeleton_folder):
            short_filename = os.path.splitext(filename)[0]

            # Retrieve skeleton data
            skeleton = read_xyz(input_path + skeleton_folder + filename)  # shape (2, max_frame, num_joint, n_subjects)
            '''
            skeleton_rgb = read_color_xy(
                input_path + skeleton_folder + filename)  # shape (2, max_frame, num_joint=25, n_subjects)
            '''

            # Assert skeletons are correct
            # assert skeleton.shape[1:] == skeleton_rgb.shape[1:]

            # Sequence code without extension
            f = open(output_path + "log.txt", "a+")
            f.write(short_filename)
            f.write("\r\n")
            f.close()
            # print(short_filename)

            # Read corresponding video
            '''
            videodata = skvideo.io.vread(
                input_path + rgb_folder + short_filename + '_rgb.avi')  # shape (n_frames, 1080, 1920, 3)
            '''

            # Check that video data has same number of frames as skeleton
            # assert skeleton.shape[1] == videodata.shape[0]

            sample = hdf.create_group(short_filename)

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
    """Creates an h5 dataset. Each group corresponds to a clip and contains the numpy array of the skeleton data and the
    numpy array of image crops around the hands

    :param input_path: NTU-RGB-D data path
    :param output_path: location of created h5 dataset
    :param compression: type of compression {"", "lzf", "gzip"}
    :param compression_opts: compression strength {1, .., 9}
    """

    ir_folder = "nturgb+d_ir/"

    open_type = "w"
    file = open(output_path + 'log.txt', 'w')
    file.close()

    with h5py.File(output_path + 'ir.h5', open_type) as hdf:
        # Progress bar
        progress_bar = progressbar(iterable=None,
                                   length=len(next(os.walk(input_path + ir_folder))[2])
                                   )

        # Loop through skeleton files
        for filename in os.listdir(input_path + ir_folder):
            short_filename = os.path.splitext(filename)[0][:-3]

            # Sequence code without extension
            f = open(output_path + "log.txt", "a+")
            f.write(short_filename)
            f.write("\r\n")
            f.close()
            # print(short_filename)

            # Read corresponding video

            videodata = skvideo.io.vread(
                input_path + ir_folder + short_filename + '.avi')[:, :, :, 0]  # shape (n_frames, H, W)

            _, H, W = videodata.shape

            # Check that video data has same number of frames as skeleton
            # assert skeleton.shape[1] == videodata.shape[0]

            sample = hdf.create_group(short_filename)

            if compression == "":
                sample.create_dataset("ir", data=videodata, chunks=(1, H, W))
            elif compression == "lzf":
                sample.create_dataset("ir", data=videodata, compression=compression, chunks=(1, H, W))
            elif compression == "gzip":
                sample.create_dataset("ir", data=videodata,
                                      compression=compression,
                                      compression_opts=compression_opts,
                                      chunks=(1, H, W))
            else:
                print("Compression type not recognized ... Exiting")
                return

            progress_bar.update(1)


def create_h5_ir_cropped_dataset_from_h5(input_path, output_path, compression="", compression_opts=9):
    """Creates an h5 dataset. Each group corresponds to a clip and contains the numpy array of the skeleton data and the
    numpy array of image crops around the hands

    :param input_path: NTU-RGB-D data path
    :param output_path: location of created h5 dataset
    :param compression: type of compression {"", "lzf", "gzip"}
    :param compression_opts: compression strength {1, .., 9}
    """

    # Get samples list
    samples_names_list = [line.rstrip('\n') for line in open(input_path + "samples_names.txt")]

    # Existing h5 files
    ir_skeleton_dataset_file_name = "ir_skeleton.h5"
    ir_dataset_file_name = "ir.h5"

    # Offset around bounding box
    offset = 20

    # Overwrite existing log file and create a new one
    open_type = "w"
    file = open(output_path + 'log.txt', 'w')
    file.close()

    # Open h5 files
    ir_skeleton_dataset = h5py.File(input_path + ir_skeleton_dataset_file_name, 'r')
    ir_dataset = h5py.File(input_path + ir_dataset_file_name, 'r')

    with h5py.File(output_path + 'ir_cropped.h5', open_type) as hdf:
        # Progress bar
        progress_bar = progressbar(iterable=None, length=len(samples_names_list))

        # Loop through skeleton files
        for filename in samples_names_list:

            # Sequence code without extension
            f = open(output_path + "log.txt", "a+")
            f.write(filename)
            f.write("\r\n")
            f.close()

            # Fetch corresponding ir raw sequence
            videodata = ir_dataset[filename]["ir"][:]

            # Pad video
            cropped_ir_sample = np.pad(videodata, ((0, 0), (offset, offset), (offset, offset)), mode='constant')

            # Get corresponding ir skeleton shape(2 : {y, x}, seq_len, n_joints, n_subjects)
            ir_skeleton = ir_skeleton_dataset[filename]["ir_skeleton"][:].clip(min=0)

            # Calculate boundaries
            has_2_subjects = np.any(ir_skeleton[:, :, :, 1])

            if not has_2_subjects:
                y_min = min(np.uint16(np.amin(ir_skeleton[0, :, :, 0])), videodata.shape[2])
                y_max = min(np.uint16(np.amax(ir_skeleton[0, :, :, 0])), videodata.shape[2])

                x_min = min(np.uint16(np.amin(ir_skeleton[1, :, :, 0])), videodata.shape[1])
                x_max = min(np.uint16(np.amax(ir_skeleton[1, :, :, 0])), videodata.shape[1])

            else:
                y_min = min(np.uint16(np.amin(ir_skeleton[0, :, :, :])), videodata.shape[2])
                y_max = min(np.uint16(np.amax(ir_skeleton[0, :, :, :])), videodata.shape[2])

                x_min = min(np.uint16(np.amin(ir_skeleton[1, :, :, :])), videodata.shape[1])
                x_max = min(np.uint16(np.amax(ir_skeleton[1, :, :, :])), videodata.shape[1])

            # Crop video
            cropped_ir_sample = cropped_ir_sample[:, x_min:x_max + 2 * offset, y_min:y_max + 2 * offset]
            _, H, W = cropped_ir_sample.shape

            sample = hdf.create_group(filename)

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
    # Get samples list
    samples_names_list = [line.rstrip('\n') for line in open(input_path + "samples_names.txt")]

    # Existing h5 files
    ir_skeleton_dataset_file_name = "ir_skeleton.h5"
    ir_dataset_file_name = "ir.h5"

    # Offset around bounding box
    offset = 20

    # Overwrite existing log file and create a new one
    open_type = "w"
    file = open(output_path + 'log.txt', 'w')
    file.close()

    # Open h5 files
    ir_skeleton_dataset = h5py.File(input_path + ir_skeleton_dataset_file_name, 'r')
    ir_dataset = h5py.File(input_path + ir_dataset_file_name, 'r')

    with h5py.File(output_path + 'ir_cropped_moving.h5', open_type) as hdf:
        # Progress bar
        progress_bar = progressbar(iterable=None, length=len(samples_names_list))

        # Loop through skeleton files
        for filename in samples_names_list:

            # Sequence code without extension
            f = open(output_path + "log.txt", "a+")
            f.write(filename)
            f.write("\r\n")
            f.close()

            # Fetch corresponding ir raw sequence shape (n_frames, H, W)
            videodata = ir_dataset[filename]["ir"][:]

            # Get corresponding ir skeleton shape(2 : {y, x}, seq_len, n_joints, n_subjects)
            ir_skeleton = ir_skeleton_dataset[filename]["ir_skeleton"][:].clip(min=0)

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

            x_min.clip(max=videodata.shape[1])
            x_max.clip(max=videodata.shape[1])
            y_min.clip(max=videodata.shape[2])
            y_max.clip(max=videodata.shape[2])

            # Crop and scale ir video
            new_sequence = []
            for t in range(videodata.shape[0]):
                # Fetch individual frame
                frame = videodata[t]  # shape (H, W)

                # Pad frame with zeros (to compensate for offset)
                frame = np.pad(frame, ((offset, offset), (offset, offset)), mode='constant')

                # Crop frame
                frame = frame[x_min[t]:x_max[t] + 2 * offset,
                              y_min[t]:y_max[t] + 2 * offset]

                # Rescale frame
                ir_frame = cv2.resize(frame, dsize=(112, 112))
                new_sequence.append(ir_frame)

            new_sequence = np.stack(new_sequence, axis=0)  # shape (n_frames, 112, 112)
            _, H, W = new_sequence.shape

            sample = hdf.create_group(filename)

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


# Code courtesy of yysijie and the awesome paper ST-GCN
# https://github.com/yysijie/st-gcn/
def read_skeleton(file):
    with open(file, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            frame_info['numBody'] = int(f.readline())
            frame_info['bodyInfo'] = []
            for m in range(frame_info['numBody']):
                body_info = {}
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)

    return skeleton_sequence


def read_xyz(file, max_body=2, num_joint=25):
    seq_info = read_skeleton(file)
    data = np.zeros((3, seq_info['numFrame'], num_joint, max_body), dtype=np.float32)
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[:, n, j, m] = [v['x'], v['y'], v['z']]
                else:
                    pass
    data = np.around(data, decimals=3)

    return data


def read_color_xy(file, max_body=2, num_joint=25):
    seq_info = read_skeleton(file)
    data = np.zeros((2, seq_info['numFrame'], num_joint, max_body), dtype=np.float32)

    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[:, n, j, m] = [v['colorX'], v['colorY']]
                else:
                    pass

    return data


def read_ir_xy(file, max_body=2, num_joint=25):
    seq_info = read_skeleton(file)
    data = np.zeros((2, seq_info['numFrame'], num_joint, max_body), dtype=np.float32)

    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[:, n, j, m] = [v['depthX'], v['depthY']]
                else:
                    pass
    return data