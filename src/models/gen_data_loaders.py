r"""
Used to create the training, validation and test PyTorch data loaders. All data loaders are created from the same
custom PyTorch dataset template (h5_pytorch_dataset.py). A helper function is used to create 3 lists containing the
sequences' names for the 3 sets. These lists are used for the __getitem__ method of the datasets.

The provided functions are as follows:
    - *gen_sets_lists*: Creates lists with the sequences' names of the train-val-test splits
    - *create_data_loaders*: Creates three data loaders corresponding to the train-val-test splits

We use 5% of the training set as our validation set.

**Note** that because we fix the seed, the sets lists are consistent across runs. This is useful when studying the
impact of a given hyperparameter for example.

"""

# Custom imports
from src.models.h5_pytorch_dataset import *


def gen_sets_lists(data_path, evaluation_type):
    r"""Generates 3 lists containing the sequences' names for the train-val-test splits.

    Inputs:
        - **data_path** (str): Path containing the h5 files (default ./data/processed/). This folder should contain
          the *samples_names.txt* file containing all the samples' names.
        - **evaluation_type** (str): Benchmark evaluated. Either "cross_subject" or "cross_view"

    Outputs:
        - **training_samples** (list): All the training sequences' names
        - **validation_samples** (list): All the validation sequences' names
        - **testing samples** (list): All the testing sequences' names

    """
    # Creates a list of all sample names
    samples_names_list = [line.rstrip('\n') for line in open(data_path + "samples_names.txt")]

    # Contains all training sample names
    training_samples = []

    if evaluation_type == "cross_subject":
        # Subjects in cross subject benchmark
        training_subjects = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]

        # Create list of strings in Pxxx format to identify training samples
        training_subjects_pxxx = []
        for s in training_subjects:
            training_subjects_pxxx.append("P{:03d}".format(s))

        # Gather training samples names
        training_samples = [s for s in samples_names_list if any(xs in s for xs in training_subjects_pxxx)]

    elif evaluation_type == "cross_view":
        # Cameras in the cross view benchmark
        training_cameras = [2, 3]

        # Create list of strings in Cxxx format to identify training samples
        training_cameras_cxxx = []
        for s in training_cameras:
            training_cameras_cxxx.append("C{:03d}".format(s))

        # Gather training samples
        training_samples = [s for s in samples_names_list if any(xs in s for xs in training_cameras_cxxx)]

    # Test set
    testing_samples = list(set(samples_names_list) - set(training_samples))

    # Forgot why I had to copy lists ...
    training_samples = training_samples.copy()
    testing_samples = testing_samples.copy()

    # Validation set
    validation_samples = [training_samples.pop(random.randrange(len(training_samples))) for _ in
                          range(int(0.05 * len(training_samples)))]

    return training_samples, validation_samples, testing_samples


def create_data_loaders(data_path,
                        evaluation_type,
                        model_type,
                        use_pose,
                        use_ir,
                        use_cropped_IR,
                        batch_size,
                        sub_sequence_length,
                        augment_data,
                        mirror_skeleton):
    r"""Generates three PyTorch data loaders corresponding to the train-val-test splits.

    Inputs:
        - **data_path** (str): Path containing the h5 files (default ./data/processed/).
        - **evaluation_type** (str): Benchmark evaluated. Either "cross_subject" of "cross_view"
        - **model_type** (str): "FUSION" only for now.
        - **use_pose** (bool): Include skeleton data
        - **use_ir** (bool): Include IR data
        - **use_cropped_IR** (bool): Type of IR dataset
        - **batch_size** (int): Size of batch
        - **sub_sequence_length** (str): Number of frames to subsample from full IR sequences
        - **augment_data** (bool): Choose to augment data by geometric transformation (skeleton data) or horizontal
          flip (IR data)
        - **mirror_skeleton** (bool): Choose to perform mirroring on skeleton data (e.g. left hand becomes right hand)

    Outputs:
        - **training_generator** (PyTorch data loader): Training PyTorch data loader
        - **validation_generator** (PyTorch data loader): Validation PyTorch data loader
        - **testing_generator** (PyTorch data loader): Testing PyTorch data loader
    """

    # Generate the sets samples names lists
    training_samples, validation_samples, testing_samples = gen_sets_lists(data_path, evaluation_type)

    # Create train dataset
    training_set = TorchDataset(model_type,
                                use_pose,
                                use_ir,
                                use_cropped_IR,
                                data_path,
                                sub_sequence_length,
                                augment_data,
                                mirror_skeleton,
                                training_samples)

    # Create train data loader
    training_generator = data.DataLoader(training_set,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         pin_memory=True,
                                         num_workers=8)

    # Create validation dataset
    validation_set = TorchDataset(model_type,
                                  use_pose,
                                  use_ir,
                                  use_cropped_IR,
                                  data_path,
                                  sub_sequence_length,
                                  False,
                                  False,
                                  validation_samples,
                                  training_set.c_min,
                                  training_set.c_max)

    # Create validation data loader
    validation_generator = data.DataLoader(validation_set,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=8)

    # Create test dataset
    testing_set = TorchDataset(model_type,
                               use_pose,
                               use_ir,
                               use_cropped_IR,
                               data_path,
                               sub_sequence_length,
                               False,
                               False,
                               testing_samples,
                               training_set.c_min,
                               training_set.c_max)

    # Create test data loader
    testing_generator = data.DataLoader(testing_set,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        pin_memory=True,
                                        num_workers=8)

    return training_generator, validation_generator, testing_generator
