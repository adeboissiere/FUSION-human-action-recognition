import numpy as np
import h5py
import random
import cv2

from src.utils.joints import *
from src.models.data_augmentation import *


class DataLoader():
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
                 use_validation):

        self.model_type = model_type
        self.batch_size = batch_size
        self.evaluation_type = evaluation_type
        self.sub_sequence_length = sub_sequence_length
        self.continuous_frames = continuous_frames
        self.normalize_skeleton = normalize_skeleton
        self.normalization_type = normalization_type
        self.kinematic_chain_skeleton = kinematic_chain_skeleton
        self.augment_data = augment_data
        self.use_validation = use_validation

        # Opens h5 file
        self.dataset = h5py.File(data_path + "datasets.h5", 'r')

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

        self.training_samples = training_samples.copy()
        self.training_samples_batch = training_samples.copy()
        self.testing_samples = testing_samples.copy()
        self.testing_samples_batch = testing_samples.copy()

        self.n_batches = int(len(training_samples) / batch_size) + int(len(training_samples) % batch_size != 0)
        self.n_batches_test = int(len(testing_samples) / batch_size) + int(len(testing_samples) % batch_size != 0)

        print("\r\n===== DATA LOADER SUMMARY =====")
        print(str(len(training_samples)) + " training samples")
        print(str(len(testing_samples)) + " testing samples")

        # Validation set (use 5% of training set)
        if self.use_validation:
            validation_samples = [training_samples.pop(random.randrange(len(training_samples))) for _ in range(int(0.05 * len(training_samples)))]

            self.validation_samples = validation_samples.copy()
            self.validation_samples_batch = validation_samples.copy()

            print(str(len(validation_samples)) + " validation samples")
            self.n_batches_val = int(len(validation_samples) / batch_size) + int(len(validation_samples) % batch_size != 0)

        else:
            print("0 validation samples")

    def _skeleton_to_kinematic_chain(self, skeleton):
        # skeleton shape (3, max_frame, num_joint=25, 2)
        max_frame = skeleton.shape[1]

        kinematic_skeleton = np.zeros((3, max_frame, kinematic_chain.shape[0], 2))

        for i in range(kinematic_chain.shape[0]):
            kinematic_skeleton[:, :, i, :] = skeleton[:, :, kinematic_chain[i], :]

        return kinematic_skeleton

    def _gen_non_continuous_sample(self, hand_crops, skeleton, max_frame):
        skeleton_frame = []
        hand_crops_frame = []

        n_frames_sub_sequence = max_frame / self.sub_sequence_length  # size of each sub sequence
        for sub_sequence in range(self.sub_sequence_length):
            lower_index = int(sub_sequence * n_frames_sub_sequence)
            upper_index = int((sub_sequence + 1) * n_frames_sub_sequence) - 1
            random_frame = random.randint(lower_index, upper_index)

            # print(str(random_frame) + " in [" + str(lower_index) + "-" + str(upper_index) + "]")

            skeleton_frame.append(skeleton[:, random_frame, :, :])
            hand_crops_frame.append(hand_crops[random_frame])

        return skeleton_frame, hand_crops_frame

    def _create_arrays_from_batch_samples(self, batch_samples):
        skeletons_list = []
        hand_crops_list = []

        # Access corresponding samples
        for sample_name in batch_samples:
            skeleton = self.dataset[sample_name]["skeleton"][:]  # shape (3, max_frame, num_joint=25, 2)
            hand_crops = self.dataset[sample_name]["rgb"][:]  # shape (max_frame, n_hands = {2, 4}, crop_size, crop_size, 3)

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

            # Data augmentation : rotation around x, y, z axis (see data_augmentation.py for values)
            if self.augment_data:
                skeleton = rotate_skeleton(skeleton)

            # Transform skeleton to place joints adjacently to model spatial dependency
            if self.kinematic_chain_skeleton:
                skeleton = self._skeleton_to_kinematic_chain(skeleton)

            # Pad hand_crops if only one subject found
            if hand_crops.shape[1] == 2:
                pad = np.zeros(hand_crops.shape, dtype=hand_crops.dtype)
                hand_crops = np.concatenate((hand_crops, pad), axis=1)

            max_frame = hand_crops.shape[0]

            if self.model_type in ['GRU', 'VA-LSTM']:
                # Cut sequence into T sub sequences and take a random frame in each (for RNNs)
                if not self.continuous_frames:
                    skeleton_frame, hand_crops_frame = self._gen_non_continuous_sample(hand_crops, skeleton, max_frame)

                    skeletons_list.append(np.stack(skeleton_frame, axis=1))
                    hand_crops_list.append(np.stack(hand_crops_frame, axis=0))

                # Take a random sub sequence
                else:
                    start = random.randint(0, max_frame - self.sub_sequence_length)

                    skeletons_list.append(skeleton[:, start:start + self.sub_sequence_length, :, :])
                    hand_crops_list.append(hand_crops[start:start + self.sub_sequence_length])

            # If hyper parameter sub_sequence_length == 0, then take the entire sequence (for CNNs)
            # The skeleton sequence is then transformed into an image
            elif self.model_type in ['VA-CNN', 'STA-HANDS', 'POSE-RGB']:
                max_frame = skeleton.shape[1]
                n_joints = skeleton.shape[2]

                # Reshape skeleton coordinates into an image
                skeleton_image = np.zeros((3, max_frame, 2 * n_joints))
                skeleton_image[:, :, 0:n_joints] = skeleton[:, :, :, 0]
                skeleton_image[:, :, n_joints:2*n_joints] = skeleton[:, :, :, 1]
                skeleton_image = np.transpose(skeleton_image, (0, 2, 1))

                # Normalize
                skeleton_image = np.floor(255 * (skeleton_image - c_min) / (c_max - c_min)) # shape (3, 2 * n_joints, max_frame)

                # Reshape image for ResNet
                skeleton_image = cv2.resize(skeleton_image.transpose(1, 2, 0), dsize=(224, 224)).transpose(2, 0, 1)
                skeletons_list.append(skeleton_image)

                # Divide hands sequence into T=self.sub_sequence_length even sub sequences, then take one random frame
                # per sub sequence
                _, hand_crops_frame = self._gen_non_continuous_sample(hand_crops, skeleton, max_frame)
                hand_crops_list.append(np.stack(hand_crops_frame, axis=0))

        # X_skeleton shape
        # if sub_sequence_length != 0 : (batch_size, 3, sub_sequence_length, num_joints, 2)
        # if sub_sequence_length == 0 (CNN) : (batch_size, 3, 224, 224)
        X_skeleton = np.stack(skeletons_list)
        X_hands = np.stack(hand_crops_list)  # shape (batch_size, sub_sequence_length, 4, crop_size, crop_size, 3)

        # Extract class vector
        Y = [int(x[-3:]) for x in batch_samples]

        return X_skeleton, X_hands, Y

    def next_batch(self):
        # Take random samples
        # 1. shuffle training_sample_batch
        # 2. Take first n elements
        # 3. Remove first n elements from training_sample_batch
        random.shuffle(self.training_samples_batch)
        n_elements = min(self.batch_size, len(self.training_samples_batch))
        batch_samples = self.training_samples_batch[:n_elements]
        self.training_samples_batch = self.training_samples_batch[n_elements:]

        X_skeleton, X_hands, Y = self._create_arrays_from_batch_samples(batch_samples)

        # Reset batch when epoch complete
        if len(self.training_samples_batch) == 0:
            self.training_samples_batch = self.training_samples.copy()

        return X_skeleton, X_hands, np.asarray(Y) - 1

    def next_batch_validation(self):
        # Takes first elements of testing_samples_batch
        n_elements = min(self.batch_size, len(self.validation_samples_batch))
        batch_samples = self.validation_samples_batch[:n_elements]
        self.validation_samples_batch = self.validation_samples_batch[n_elements:]

        aug_data = self.augment_data
        self.augment_data = False

        X_skeleton, X_hands, Y = self._create_arrays_from_batch_samples(batch_samples)

        self.augment_data = aug_data

        # Reset batch when epoch complete
        if len(self.validation_samples_batch) == 0:
            self.validation_samples_batch = self.validation_samples.copy()

        return X_skeleton, X_hands, np.asarray(Y) - 1

    def next_batch_test(self):
        # Takes first elements of testing_samples_batch
        n_elements = min(self.batch_size, len(self.testing_samples_batch))
        batch_samples = self.testing_samples_batch[:n_elements]
        self.testing_samples_batch = self.testing_samples_batch[n_elements:]

        aug_data = self.augment_data
        self.augment_data = False

        X_skeleton, X_hands, Y = self._create_arrays_from_batch_samples(batch_samples)

        self.augment_data = aug_data

        # Reset batch when epoch complete
        if len(self.testing_samples_batch) == 0:
            self.testing_samples_batch = self.testing_samples.copy()

        return X_skeleton, X_hands, np.asarray(Y) - 1


