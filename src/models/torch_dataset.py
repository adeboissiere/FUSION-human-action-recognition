import h5py
import random
import torch
from torch.utils import data

from src.models.data_loader_utils import *

def gen_sets_lists(data_path, evaluation_type):
    # Creates a list of all sample names
    samples_names_list = [line.rstrip('\n') for line in open(data_path + "samples_names.txt")]

    # Create list of samples without skeleton
    missing_skeletons_list = [line.rstrip('\n') for line in open(data_path + "missing_skeleton.txt")]

    # Remove missing skeletons from sample_names_list
    samples_names_list = list(set(samples_names_list) - set(missing_skeletons_list))

    # Contains all training sample names
    training_samples = []

    if evaluation_type == "cross_subject":
        training_subjects = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]

        # Create list of strings in Pxxx format to identify training samples
        training_subjects_pxxx = []
        for s in training_subjects:
            training_subjects_pxxx.append("P{:03d}".format(s))

        training_samples = [s for s in samples_names_list if any(xs in s for xs in training_subjects_pxxx)]

    elif evaluation_type == "cross_view":
        training_cameras = [2, 3]

        # Create list of strings in Cxxx format to identify training samples
        training_cameras_cxxx = []
        for s in training_cameras:
            training_cameras_cxxx.append("C{:03d}".format(s))

        training_samples = [s for s in samples_names_list if any(xs in s for xs in training_cameras_cxxx)]

    # Test set
    testing_samples = list(set(samples_names_list) - set(training_samples))

    training_samples = training_samples.copy()
    testing_samples = testing_samples.copy()

    return training_samples, testing_samples


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self,
                 model_type,
                 batch_size,
                 data_path,
                 evaluation_type,
                 sub_sequence_length,
                 continuous_frames,
                 normalize_skeleton,
                 normalization_type,
                 kinematic_chain_skeleton,
                 augment_data,
                 use_validation,
                 samples_names):
        super(TorchDataset, self).__init__()

        self.model_type = model_type
        self.data_path = data_path
        self.sub_sequence_length = sub_sequence_length
        self.normalize_skeleton = normalize_skeleton
        self.normalization_type = normalization_type
        self.augment_data = augment_data

        self.samples_names = samples_names

    def __getitem__(self, index):
        y = int(self.samples_names[index][-3:]) - 1

        # Open h5 files
        if self.model_type in ['VA-CNN', 'AS-CNN', 'FUSION']:
            # retrieve skeleton sequence of shape (3, max_frame, num_joint=25, 2)
            with h5py.File(self.data_path + "skeleton.h5", 'r') as skeleton_dataset:
                skeleton = skeleton_dataset[self.samples_names[index]]["skeleton"][:]

        if self.model_type in ['CNN3D', 'FUSION']:
            # retrives IR video of shape (n_frames, H, W)
            with h5py.File(self.data_path + "ir_cropped.h5", 'r') as ir_dataset:
                ir_video = ir_dataset[self.samples_names[index]]["ir"][:]

        # If model requires skeleton data
        if self.model_type in ['VA-CNN', 'AS-CNN', 'FUSION']:



            # See jp notebook 4.0 for values
            c_min = 0
            c_max = 0

            if self.normalize_skeleton:
                # Normalize skeleton according to S-trans (see View Adaptive Network for details)
                # Subjects 1 and 2 have their own new coordinates system
                trans_vector = skeleton[:, 0, Joints.SPINEMID, :]

                if self.normalization_type == "2-COORD-SYS":
                    c_min = -4.657
                    c_max = 5.042

                # Subjects 1 and 2 are transposed into the coordinates system of subject 1
                elif self.normalization_type == "1-COORD-SYS":
                    trans_vector[:, 1] = trans_vector[:, 0]

                    c_min = -4.767
                    c_max = 5.188

                skeleton = (skeleton.transpose(1, 2, 0, 3) - trans_vector).transpose(2, 0, 1, 3)

            # Data augmentation : rotation around x, y, z axis (see data_loader_utils.py for values)
            if self.augment_data:
                skeleton = rotate_skeleton(skeleton)

            # Each model has its specific data streams
            if self.model_type in ['VA-CNN', 'FUSION']:
                # shape (3, 224, 224)
                skeleton_image = create_stretched_image_from_skeleton_sequence(skeleton, c_min, c_max)

            if self.model_type in ['AS-CNN']:
                # shape (3, 224, 224)
                skeleton_image = create_padded_image_from_skeleton_sequence(skeleton, c_min, c_max)

                # shape (n_neighbors * n_subjects = 2 * 24, )
                avg_bone_length = compute_avg_bone_length(skeleton)

        # If model requires IR data
        if self.model_type in ['CNN3D', 'FUSION']:
            n_frames = ir_video.shape[0]
            n_frames_sub_sequence = n_frames / self.sub_sequence_length  # size of each sub sequence

            ir_sequence = []

            for sub_sequence in range(self.sub_sequence_length):
                lower_index = int(sub_sequence * n_frames_sub_sequence)
                upper_index = int((sub_sequence + 1) * n_frames_sub_sequence) - 1
                random_index = random.randint(lower_index, upper_index)

                ir_image = cv2.resize(ir_video[random_index], dsize=(112, 112))

                ir_sequence.append(ir_image)

            ir_sequence = np.stack(ir_sequence, axis=0)  # shape (sub_seq_len, 224, 224)

        # Return corresponding data
        if self.model_type in ['VA-CNN']:
            return skeleton_image, y

        elif self.model_type in ['AS-CNN']:
            return skeleton_image, avg_bone_length, y

        elif self.model_type in ['CNN3D']:
            return ir_sequence, y

        elif self.model_type in ['FUSION']:
            return skeleton_image, ir_sequence, y

    def __len__(self):
        return len(self.samples_names)

