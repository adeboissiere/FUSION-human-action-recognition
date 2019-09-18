import os

import h5py

import skvideo.io
import skvideo.datasets

from click import progressbar

# Custom modules
from src.utils.joints import *


def create_2d_ir_skeleton(input_path, output_path, compression ="", compression_opts = 9):
    """Creates an h5 dataset. Each group corresponds to a clip and contains the numpy array of the skeleton data and the
    numpy array of image crops around the hands

    :param input_path: NTU-RGB-D data path
    :param output_path: location of created h5 dataset
    :param compression: type of compression {"", "lzf", "gzip"}
    :param compression_opts: compression strength {1, .., 9}
    """

    skeleton_folder = "nturgb+d_skeletons/"

    open_type = "w"
    file = open(output_path + 'log.txt', 'w+')
    file.close()

    with h5py.File(output_path + 'ir_skeleton.h5', open_type) as hdf:
        # Progress bar
        progress_bar = progressbar(iterable=None,
                                   length = len(next(os.walk(input_path + skeleton_folder))[2])
)

        # Loop through skeleton files
        for filename in os.listdir(input_path + skeleton_folder):
            short_filename = os.path.splitext(filename)[0]

            # Retrieve skeleton data
            skeleton = read_ir_xy(input_path + skeleton_folder + filename)  # shape (2, max_frame, num_joint, n_subjects)

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


def create_h5_skeleton_dataset(input_path, output_path, compression ="", compression_opts = 9):
    """Creates an h5 dataset. Each group corresponds to a clip and contains the numpy array of the skeleton data and the
    numpy array of image crops around the hands

    :param input_path: NTU-RGB-D data path
    :param output_path: location of created h5 dataset
    :param compression: type of compression {"", "lzf", "gzip"}
    :param compression_opts: compression strength {1, .., 9}
    """

    skeleton_folder = "nturgb+d_skeletons/"

    open_type = "w"
    file = open(output_path + 'log.txt', 'w+')
    file.close()

    with h5py.File(output_path + 'skeleton.h5', open_type) as hdf:
        # Progress bar
        progress_bar = progressbar(iterable=None,
                                   length = len(next(os.walk(input_path + skeleton_folder))[2])
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


def create_h5_ir_dataset(input_path, output_path, compression ="", compression_opts = 9):
    """Creates an h5 dataset. Each group corresponds to a clip and contains the numpy array of the skeleton data and the
    numpy array of image crops around the hands

    :param input_path: NTU-RGB-D data path
    :param output_path: location of created h5 dataset
    :param compression: type of compression {"", "lzf", "gzip"}
    :param compression_opts: compression strength {1, .., 9}
    """

    ir_folder = "nturgb+d_ir/"

    open_type = "w"
    file = open(output_path + 'log.txt', 'w+')
    file.close()

    with h5py.File(output_path + 'ir.h5', open_type) as hdf:
        # Progress bar
        progress_bar = progressbar(iterable=None,
                                   length=len(next(os.walk(input_path + ir_folder))[2])
                                   )

        # Loop through skeleton files
        for filename in os.listdir(input_path + ir_folder):
            short_filename = os.path.splitext(filename)[0]

            # Sequence code without extension
            f = open(output_path + "log.txt", "a+")
            f.write(short_filename)
            f.write("\r\n")
            f.close()
            # print(short_filename)

            # Read corresponding video

            videodata = skvideo.io.vread(
                input_path + ir_folder + short_filename + '.avi')  # shape (n_frames, H, W, 3)

            # Check that video data has same number of frames as skeleton
            # assert skeleton.shape[1] == videodata.shape[0]

            sample = hdf.create_group(short_filename)

            if compression == "":
                sample.create_dataset("ir", data=videodata)
            elif compression == "lzf":
                sample.create_dataset("ir", data=videodata, compression=compression)
            elif compression == "gzip":
                sample.create_dataset("ir", data=videodata,
                                      compression=compression,
                                      compression_opts=compression_opts)
            else:
                print("Compression type not recognized ... Exiting")
                return

            progress_bar.update(1)


def create_h5_ir_cropped_dataset(input_path, output_path, compression ="", compression_opts = 9):
    """Creates an h5 dataset. Each group corresponds to a clip and contains the numpy array of the skeleton data and the
    numpy array of image crops around the hands

    :param input_path: NTU-RGB-D data path
    :param output_path: location of created h5 dataset
    :param compression: type of compression {"", "lzf", "gzip"}
    :param compression_opts: compression strength {1, .., 9}
    """

    ir_folder = "nturgb+d_ir/"
    ir_skeleton_dataset_file_name = "ir_skeleton.h5"

    offset = 20

    open_type = "w"
    file = open(output_path + 'log.txt', 'w+')
    file.close()

    # Open ir_skeleton h5 file
    ir_skeleton_dataset = h5py.File(input_path + ir_skeleton_dataset_file_name, 'r')

    with h5py.File(output_path + 'ir_cropped.h5', open_type) as hdf:
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

            # Read corresponding video
            videodata = skvideo.io.vread(
                input_path + ir_folder + short_filename + '_ir.avi')[:, :, :, 0]  # shape (n_frames, H, W)

            # Pad video
            cropped_ir_sample = np.pad(videodata, ((0, 0), (offset, offset), (offset, offset)), mode='constant')

            # Get corresponding ir skeleton shape(2 : {x, y}, seq_len, n_joints)
            ir_skeleton = ir_skeleton_dataset[short_filename]["ir_skeleton"][:].clip(min = 0)

            # Calculate boundaries
            has_2_subjects = np.any(ir_skeleton[:, :, :, 1])
            ir_skeleton = ir_skeleton.clip(min=0)
            if not (has_2_subjects):
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

            sample = hdf.create_group(short_filename)

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

def extract_hands(skeleton_rgb, videodata, crop_size):
    """Extracts the hand crops from a video data in numpy array format.

    Parameters:
    skeleton_rgb -- the 2D projection on RGB image of the skeletons of shape (max_frame, 1080, 1920, 3)
    videodata -- the video in numpy format of shape (max_frame, 1080, 1920, 3)
    crop_size -- the image size of the crop around each hand (crop_size x crop_size x 3)

    Returns:
    np array containing hand crops of shape (n_frames, number of hands (2 subjects), x, y)
    """
    hand_crops = []

    max_x = videodata.shape[1]
    max_y = videodata.shape[2]

    offset = int(crop_size / 2)

    n_frames = skeleton_rgb.shape[1]
    n_subjects = 1

    # Check if some coordinates for skeleton 2 are != 0
    if np.any(skeleton_rgb[:, :, :, 1]):
        n_subjects = 2

    # Get correspoding time coordinates
    for s in range(n_subjects):
        hand_crops_s = np.zeros((n_frames, 2, crop_size, crop_size, 3), dtype=np.uint8)

        for t in range(n_frames):
            # Get right/left hand center coordinates
            left_hand_x = max(min(int(np.nan_to_num(skeleton_rgb[1, t, Joints.HANDLEFT, s])), max_x), 0)
            left_hand_y = max(min(int(np.nan_to_num(skeleton_rgb[0, t, Joints.HANDLEFT, s])), max_y), 0)

            right_hand_x = max(min(int(np.nan_to_num(skeleton_rgb[1, t, Joints.HANDRIGHT, s])), max_x), 0)
            right_hand_y = max(min(int(np.nan_to_num(skeleton_rgb[0, t, Joints.HANDRIGHT, s])), max_y), 0)

            frame = np.pad(videodata[t], ((offset, offset), (offset, offset), (0, 0)),
                           mode='constant')  # shape(1130, 1970, 3)

            hand_crops_s[t, 0] = frame[left_hand_x:left_hand_x + 2 * offset, left_hand_y:left_hand_y + 2 * offset]
            hand_crops_s[t, 1] = frame[right_hand_x:right_hand_x + 2 * offset, right_hand_y:right_hand_y + 2 * offset]

        hand_crops.append(hand_crops_s)

    return np.concatenate(hand_crops, axis=1)


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