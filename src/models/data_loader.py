import numpy as np
import h5py
import random


class DataLoader():
    def __init__(self, batch_size, data_path, evaluation_type, sub_sequence_length):
        self.batch_size = batch_size
        self.evaluation_type = evaluation_type
        self.sub_sequence_length = sub_sequence_length

        # Opens h5 file
        self.dataset = h5py.File(data_path + "datasets.h5", 'r')

        # Creates a list of all sample names
        samples_names_list = [line.rstrip('\n') for line in open(data_path + "samples_names.txt")]

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

        testing_samples = set(samples_names_list) - set(training_samples)

        self.training_samples = training_samples.copy()
        self.training_samples_batch = training_samples.copy()
        self.testing_samples = testing_samples.copy()

        self.n_batch = int(len(training_samples) / batch_size) + int(len(training_samples) % batch_size != 0)

    def next_batch(self):
        # Take random samples (1. shuffle training_sample_batch 2. Take first n elements
        # 3. Remove first n elements from training_sample_batch)
        random.shuffle(self.training_samples_batch)
        n_elements = min(self.batch_size, len(self.training_samples_batch))
        batch_samples = self.training_samples_batch[:n_elements]
        self.training_samples_batch = self.training_samples_batch[n_elements:]

        print(len(self.training_samples_batch))

        skeletons_list = []
        hand_crops_list = []
        # Access corresponding samples
        for sample_name in batch_samples:
            skeleton = self.dataset[sample_name]["skeleton"][:] # shape (3, max_frame, num_joint=25, 2)
            hand_crops = self.dataset[sample_name]["rgb"][:] # shape (max_frame, n_hands = {2, 4}, crop_size, crop_size, 3)

            # Pad hand_crops if only one subject found
            if hand_crops.shape[1] == 2:
                pad = np.zeros(hand_crops.shape, dtype=hand_crops.dtype)
                hand_crops = np.concatenate((hand_crops, pad), axis=1)

            # Take random subsequence
            max_frame = hand_crops.shape[0]
            start = random.randint(0, max_frame - self.sub_sequence_length)

            skeletons_list.append(skeleton[:, start:start+self.sub_sequence_length, :, :])
            hand_crops_list.append(hand_crops[start:start+self.sub_sequence_length])

        X_skeleton = np.stack(skeletons_list) # shape (batch_size, 3, sub_sequence_length, num_joints, 2)
        X_hands = np.stack(hand_crops_list) # shape (batch_size, sub_sequence_length, 4, crop_size, crop_size, 3)

        # Extract class vector
        Y = [int(x[-3:]) for x in batch_samples]

        # Reset batch when epoch complete
        if len(self.training_samples_batch) == 0:
            self.training_samples_batch = self.training_samples.copy()

        return X_skeleton, X_hands, Y