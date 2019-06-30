import numpy as np
import h5py
import random


class DataLoader():
    def __init__(self, batch_size, data_path, evaluation_type):
        self.batch_size = batch_size
        self.evaluation_type = evaluation_type

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

    def next_batch(self):
        print("coucou")
