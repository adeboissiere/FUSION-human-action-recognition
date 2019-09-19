import numpy as np
import h5py
import random

from src.utils.joints import *
from src.models.data_loader_utils import *


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

        # Opens h5 files
        self.skeleton_dataset = h5py.File(data_path + "skeleton.h5", 'r')
        self.ir_dataset = h5py.File(data_path + "ir_cropped.h5", 'r')

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

    def _create_arrays_from_batch_samples(self, batch_samples):
        skeletons_list = []
        avg_bone_length_list = []
        ir_videos_lists = []

        # Access corresponding samples
        for sample_name in batch_samples:
            skeleton = self.skeleton_dataset[sample_name]["skeleton"][:]  # shape (3, max_frame, num_joint=25, 2)

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
            if self.model_type in ['VA-CNN']:
                # shape (3, 224, 224)
                skeleton_image = create_stretched_image_from_skeleton_sequence(skeleton, c_min, c_max)
                skeletons_list.append(skeleton_image)

            elif self.model_type in ['AS-CNN']:
                # shape (3, 224, 224)
                skeleton_image = create_padded_image_from_skeleton_sequence(skeleton, c_min, c_max)
                skeletons_list.append(skeleton_image)

                # shape (n_neighbors * n_subjects = 2 * 24, )
                avg_bone_length = compute_avg_bone_length(skeleton)
                avg_bone_length_list.append(avg_bone_length)

            elif self.model_type in ['base-IR', 'CNN3D']:
                ir_video = self.ir_dataset[sample_name]["ir"][:] # shape (n_frames, H, W)

                n_frames = ir_video.shape[0]
                n_frames_sub_sequence = n_frames / self.sub_sequence_length  # size of each sub sequence

                ir_sequence = []

                for sub_sequence in range(self.sub_sequence_length):
                    lower_index = int(sub_sequence * n_frames_sub_sequence)
                    upper_index = int((sub_sequence + 1) * n_frames_sub_sequence) - 1
                    random_index = random.randint(lower_index, upper_index)

                    if self.model_type in ['base-IR']:
                        ir_image = cv2.resize(ir_video[random_index], dsize=(224, 224))
                    elif self.model_type in ['CNN3D']:
                        ir_image = cv2.resize(ir_video[random_index], dsize=(112, 112))
                    ir_sequence.append(ir_image)

                ir_sequence = np.stack(ir_sequence, axis=0) # shape (sub_seq_len, 224, 224)
                ir_videos_lists.append(ir_sequence)

        # Extract class vector
        Y = [int(x[-3:]) for x in batch_samples]

        if self.model_type in ['VA-CNN']:
            # X_skeleton shape (batch_size, 3, 224, 224)
            X_skeleton = np.stack(skeletons_list)

            return [X_skeleton], Y

        elif self.model_type in ['AS-CNN']:
            # X_skeleton shape (batch_size, 3, 224, 224)
            X_skeleton = np.stack(skeletons_list)

            # X_bone_length shape (batch_size, n_neighbors * n_subjects = 2 * 24)
            X_bone_length = np.stack(avg_bone_length_list)

            return [X_skeleton, X_bone_length], Y

        elif self.model_type in ['base-IR', 'CNN3D']:
            X_ir = np.repeat(np.stack(ir_videos_lists)[:, :, np.newaxis, :, :], 3, axis=2) #  shape (batch_size, seq_len, 3, H, W)

            return [X_ir], Y

    def next_batch(self):
        # Take random samples
        # 1. shuffle training_sample_batch
        # 2. Take first n elements
        # 3. Remove first n elements from training_sample_batch
        random.shuffle(self.training_samples_batch)
        n_elements = min(self.batch_size, len(self.training_samples_batch))
        batch_samples = self.training_samples_batch[:n_elements]
        self.training_samples_batch = self.training_samples_batch[n_elements:]

        X, Y = self._create_arrays_from_batch_samples(batch_samples)

        # Reset batch when epoch complete
        if len(self.training_samples_batch) == 0:
            self.training_samples_batch = self.training_samples.copy()

        return X, np.asarray(Y) - 1

    def next_batch_validation(self):
        # Takes first elements of testing_samples_batch
        n_elements = min(self.batch_size, len(self.validation_samples_batch))
        batch_samples = self.validation_samples_batch[:n_elements]
        self.validation_samples_batch = self.validation_samples_batch[n_elements:]

        aug_data = self.augment_data
        self.augment_data = False

        X, Y = self._create_arrays_from_batch_samples(batch_samples)

        self.augment_data = aug_data

        # Reset batch when epoch complete
        if len(self.validation_samples_batch) == 0:
            self.validation_samples_batch = self.validation_samples.copy()

        return X, np.asarray(Y) - 1

    def next_batch_test(self):
        # Takes first elements of testing_samples_batch
        n_elements = min(self.batch_size, len(self.testing_samples_batch))
        batch_samples = self.testing_samples_batch[:n_elements]
        self.testing_samples_batch = self.testing_samples_batch[n_elements:]

        aug_data = self.augment_data
        self.augment_data = False

        X, Y = self._create_arrays_from_batch_samples(batch_samples)

        self.augment_data = aug_data

        # Reset batch when epoch complete
        if len(self.testing_samples_batch) == 0:
            self.testing_samples_batch = self.testing_samples.copy()

        return X, np.asarray(Y) - 1


