r"""
Custom PyTorch dataset that reads from the h5 datasets (see src.data module for more infos).

"""

import h5py
import random
import torch
from torch.utils import data

from src.models.data_augmentation import *


class TorchDataset(torch.utils.data.Dataset):
    r"""This custom PyTorch lazy loads from the h5 datasets. This means that it does not load the entire dataset in
    memory, which would be impossible for the IR sequences. Instead, it opens and reads from the h5 file. This is a bit
    slower, but very memory efficient. Additionally, the lost time is mitigated when using multiple workers for the
    data loaders.

    Attributes:
        - **data_path** (str): Path containing the h5 files (default *./data/processed/*).
        - **model_type** (str): "FUSION" only for now.
        - **use_pose** (bool): Include skeleton data
        - **use_ir** (bool): Include IR data
        - **use_cropped_IR** (bool): Type of IR dataset
        - **sub_sequence_length** (str): Number of frames to subsample from full IR sequences
        - **augment_data** (bool): Choose to augment data by geometric transformation (skeleton data) or horizontal
          flip (IR data)
        - **mirror_skeleton** (bool): Choose to perform mirroring on skeleton data (e.g. left hand becomes right hand)
        - **samples_names** (list): Contains the sequences names of the dataset (ie. train, validation, test)
        - **c_min** (float): Minimum coordinate after camera-subject normalization
        - **c_max** (float): Maximum coordinate after camera-subject normalization

    Methods:
        - *__getitem__(index)*: Returns the processed sequence (skeleton and/or IR) and its label
        - *__len__()*: Returns the number of elements in dataset.

    """
    def __init__(self,
                 model_type,
                 use_pose,
                 use_ir,
                 use_cropped_IR,
                 data_path,
                 sub_sequence_length,
                 augment_data,
                 mirror_skeleton,
                 samples_names,
                 c_min = None,
                 c_max = None):
        super(TorchDataset, self).__init__()

        self.model_type = model_type
        self.use_pose = use_pose
        self.use_ir = use_ir
        self.use_cropped_IR = use_cropped_IR
        self.data_path = data_path
        self.sub_sequence_length = sub_sequence_length
        self.augment_data = augment_data
        self.mirror_skeleton = mirror_skeleton

        self.samples_names = samples_names
        self.c_min = c_min
        self.c_max = c_max

        if self.use_pose and c_max is None and c_min is None:
            print("Computing c_min and c_max. This takes a while ...")

            c_min = []
            c_max = []

            with h5py.File(self.data_path + "skeleton.h5", 'r') as skeleton_dataset:
                for sample_name in self.samples_names:
                    skeleton = skeleton_dataset[sample_name]["skeleton"][:]

                    # Perform normalization step
                    trans_vector = skeleton[:, 0, Joints.SPINEMID, :]  # shape (3, 2)
                    trans_vector[:, 1] = trans_vector[:, 0]
                    skeleton = (skeleton.transpose(1, 2, 0, 3) - trans_vector).transpose(2, 0, 1, 3)

                    # Update c_min and c_max
                    c_min.append(np.amin(skeleton))
                    c_max.append(np.amax(skeleton))

            self.c_min = np.amin(c_min)
            self.c_max = np.amax(c_max)
            print("Done !")

    def __getitem__(self, index):
        r"""Returns a processed sequence and label given an index.

        Inputs:
            - **index** (int): Used as an index for **samples_names** list which will yield a sequence
              name that will be used to address the h5 files.

        Outputs:
            - **skeleton_image** (np array): Skeleton sequence mapped to an image of shape `(3, 224, 224)`.
              Equals -1 if **use_pose** is False.
            - **ir_sequence** (np array): Subsampled IR sequence of shape `(sub_sequence_length, 112, 112)`.
              Equals -1 if **use_ir** is False.
            - **y** (int): Class label of sequence.

        """
        # Get label
        y = int(self.samples_names[index][-3:]) - 1

        # Generate a random value. If <=0.5, the skeleton and IR video will be flipped
        flip_chance = random.random()

        # Open h5 files
        if self.use_pose:
            # retrieve skeleton sequence of shape (3, max_frame, num_joint=25, 2)
            with h5py.File(self.data_path + "skeleton.h5", 'r') as skeleton_dataset:
                skeleton = skeleton_dataset[self.samples_names[index]]["skeleton"][:]

        if self.use_ir:
            if self.use_cropped_IR:
                file_name = "ir_cropped.h5"
            else:
                file_name = "ir.h5"

            # retrieves IR video of shape (n_frames, H, W)
            with h5py.File(self.data_path + file_name, 'r') as ir_dataset:
                ir_video = ir_dataset[self.samples_names[index]]["ir"][:] # shape (n_frames, H, W)

                # 50% chance to flip video
                if self.augment_data and flip_chance <= 0.5:
                    ir_video = np.flip(ir_video, axis=2)

        # Potential outputs
        skeleton_image = -1
        ir_sequence = -1

        # If model requires skeleton data
        if self.use_pose:
            # Normalize skeleton according to S-trans (see View Adaptive Network for details)
            # Subjects 1 and 2 have their own new coordinates system
            trans_vector = skeleton[:, 0, Joints.SPINEMID, :]

            # Subjects 1 and 2 are transposed into the coordinates system of subject 1
            trans_vector[:, 1] = trans_vector[:, 0]

            skeleton = (skeleton.transpose(1, 2, 0, 3) - trans_vector).transpose(2, 0, 1, 3)

            # Data augmentation : rotation around x, y, z axis (see data_augmentation.py for values)
            if self.augment_data:
                if flip_chance <= 0.5 and self.mirror_skeleton:
                    skeleton = get_mirror_skeleton(skeleton)
                skeleton = rotate_skeleton(skeleton)

            # Map to RGB image
            if self.model_type in ['FUSION']:
                # shape (3, 224, 224)
                skeleton_image = np.float32(stretched_image_from_skeleton_sequence(skeleton, self.c_min, self.c_max))

        # If model requires IR data
        if self.use_ir:
            n_frames = ir_video.shape[0]
            n_frames_sub_sequence = n_frames / self.sub_sequence_length  # size of each sub sequence

            ir_sequence = []

            # Create a fixed number number of subwindows (equal to sub_sequence_length) and randomly sample from each
            for sub_sequence in range(self.sub_sequence_length):
                lower_index = int(sub_sequence * n_frames_sub_sequence)
                upper_index = int((sub_sequence + 1) * n_frames_sub_sequence) - 1
                random_index = random.randint(lower_index, upper_index)

                ir_image = cv2.resize(ir_video[random_index], dsize=(112, 112))

                ir_sequence.append(ir_image)

            # Stack the sampled frames to create new sequence
            ir_sequence = np.stack(ir_sequence, axis=0)  # shape (sub_seq_len, 112, 112)
            ir_sequence = np.float32(np.repeat(ir_sequence[:, np.newaxis, :, :], 3, axis=1))

        # Return corresponding data
        if self.model_type in ['FUSION']:
            return [skeleton_image, ir_sequence], y

    def __len__(self):
        r"""Returns number of elements in dataset

        Outputs:
            - **length** (int): Number of elements in dataset.

        """
        return len(self.samples_names)

