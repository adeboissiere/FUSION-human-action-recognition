"""
Computes the confusion matrix for a trained model. Takes as input important parameters such as the benchmark studied.
A confusion matrix in .png format is saved in the trained model folder provided.

Plotting the confusion matrix is best called using the provided Makefile provided.

>>> make confusion_matrix \\
    PROCESSED_DATA_PATH=X \\
    MODEL_FOLDER=X \\
    MODEL_FILE=X \\
    EVALUATION_TYPE=X \\
    MODEL_TYPE=X \\
    USE_POSE=X \\
    USE_IR=X \\
    FUSION_SCHEME=X \\
    USE_CROPPED_IR=X \\
    BATCH_SIZE=X \\
    SUB_SEQUENCE_LENGTH=X \\


With the parameters taking from the following values :
    - DATA_PATH:
        Path to h5 files. Default location is *./data/processed/*
    - MODEL_FOLDER:
        Output path to save models and log files. A folder inside that path will be automatically created. Default
        location is *./models/*
    - MODEL_FILE:
        Name of the model.
    - EVALUATION_TYPE:
        [cross_subject | cross_view]
    - MODEL_TYPE:
        [FUSION]
    - USE_POSE:
        [True, False]
    - USE_IR:
        [True, False]
    - FUSION_SCHEME:
        [CONCAT, MAX, SUM, AVG, CONV, MULT]
    - USE_CROPPED_IR:
        [True, False]
    - BATCH_SIZE:
        Whole positive number above 1.
    - SUB_SEQUENCE_LENGTH:
        [1 .. 20]
        Specifies the number of frames to take from a complete IR sequence.

"""
import argparse

from src.models.train_utils import *
from src.models.gen_data_loaders import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot confusion matrix')
    parser.add_argument('--data_path')
    parser.add_argument('--model_folder', default="./models/")
    parser.add_argument('--model_file')
    parser.add_argument('--evaluation_type')
    parser.add_argument('--model_type')
    parser.add_argument('--use_pose', default=False)
    parser.add_argument('--use_ir', default=False)
    parser.add_argument('--fusion_scheme', default="CONCAT")
    parser.add_argument('--use_cropped_IR', default=False)
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--sub_sequence_length', default=20)

    arg = parser.parse_args()

    # Hyper parameters
    data_path = arg.data_path
    model_folder = arg.model_folder
    model_file = arg.model_file
    evaluation_type = arg.evaluation_type
    model_type = arg.model_type
    use_pose = arg.use_pose == "True"
    use_ir = arg.use_ir == "True"
    fusion_scheme = arg.fusion_scheme
    use_cropped_IR = arg.use_cropped_IR == "True"
    batch_size = int(arg.batch_size)
    sub_sequence_length = int(arg.sub_sequence_length)

    if evaluation_type not in ["cross_subject", "cross_view"]:
        print("Error : Evaluation type not recognized")
        print("... Returning")

        exit()

    # Print summary
    print("\r\n\n\n========== PLOT CONFUSION MATRIX ==========")
    print("-> model_folder : " + model_folder)
    print("-> evaluation_type : " + evaluation_type)
    print("-> model_type : " + str(model_type))
    if model_type == "FUSION":
        print("-> use pose : " + str(use_pose))
        print("-> use ir : " + str(use_ir))
        print("-> use cropped ir : " + str(use_cropped_IR))
    print("-> batch size : " + str(batch_size))
    print("-> sub_sequence_length : " + str(sub_sequence_length))
    print()

    # Create data loaders
    _, _, test_generator = create_data_loaders(data_path,
                                               evaluation_type,
                                               model_type,
                                               use_pose,
                                               use_ir,
                                               use_cropped_IR,
                                               batch_size,
                                               sub_sequence_length,
                                               False,
                                               False)

    if model_type == "FUSION":
        model = FUSION(use_pose, use_ir, False, fusion_scheme)
    else:
        print("Model type not recognized. Exiting")
        exit()

    # Load weights from file
    model.load_state_dict(torch.load(model_folder + model_file))
    model.to(device)
    model.eval()

    # Compute confusion matrix
    with torch.no_grad():
        test_accuracy, y_true, y_pred = evaluate_set(model,
                                                     model_type,
                                                     test_generator,
                                                     model_folder,
                                                     "TEST")

        # Plot confusion matrix per epoch
        y_true = np.int32(np.concatenate(y_true))
        y_pred = np.int32(np.concatenate(y_pred))

        # Save predictions to plot confusion matrix and Cohen's Kappa
        pickle_test = open(model_folder + "test_preds" + ".cpkl", 'wb')
        pickle.dump([y_true, y_pred], pickle_test)
        pickle_test.close()

        plot_confusion_matrix(y_true, y_pred, classes, normalize=True, title="Confusion matrix")
        plt.savefig(model_folder + str(model_type) + ".png")

        print("Cohen's kappa : " + str(cohen_kappa_score(y_true, y_pred)))
        print("Accuracy over test set " + str(test_accuracy))
        evaluate_per_action_type(y_true, y_pred)

    # echo -en "\e[?25h"
    print("-> Done !")
