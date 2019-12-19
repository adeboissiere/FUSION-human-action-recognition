import h5py
import random
import torch
from torch.utils import data

from src.models.data_loader_utils import *


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self,
                 model_type,
                 use_pose,
                 use_ir,
                 use_cropped_IR,
                 data_path,
                 sub_sequence_length,
                 augment_data,
                 samples_names):
        super(TorchDataset, self).__init__()

        self.model_type = model_type
        self.use_pose = use_pose
        self.use_ir = use_ir
        self.use_cropped_IR = use_cropped_IR
        self.data_path = data_path
        self.sub_sequence_length = sub_sequence_length
        self.augment_data = augment_data

        self.samples_names = samples_names

    def __getitem__(self, index):
        y = int(self.samples_names[index][-3:]) - 1

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
                if self.augment_data and random.random() <= 0.5:
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

            c_min = -4.767
            c_max = 5.188

            skeleton = (skeleton.transpose(1, 2, 0, 3) - trans_vector).transpose(2, 0, 1, 3)

            # Data augmentation : rotation around x, y, z axis (see data_loader_utils.py for values)
            if self.augment_data:
                skeleton = rotate_skeleton(skeleton)

            # Each model has its specific data streams
            if self.model_type in ['FUSION']:
                # shape (3, 224, 224)
                skeleton_image = np.float32(create_stretched_image_from_skeleton_sequence(skeleton, c_min, c_max))

        # If model requires IR data
        if self.use_ir:
            n_frames = ir_video.shape[0]
            n_frames_sub_sequence = n_frames / self.sub_sequence_length  # size of each sub sequence

            ir_sequence = []

            for sub_sequence in range(self.sub_sequence_length):
                lower_index = int(sub_sequence * n_frames_sub_sequence)
                upper_index = int((sub_sequence + 1) * n_frames_sub_sequence) - 1
                random_index = random.randint(lower_index, upper_index)

                ir_image = cv2.resize(ir_video[random_index], dsize=(112, 112))

                ir_sequence.append(ir_image)

            ir_sequence = np.stack(ir_sequence, axis=0)  # shape (sub_seq_len, 112, 112)
            ir_sequence = np.float32(np.repeat(ir_sequence[:, np.newaxis, :, :], 3, axis=1))

        # Return corresponding data
        if self.model_type in ['FUSION']:
            return [skeleton_image, ir_sequence], y

    def __len__(self):
        return len(self.samples_names)

