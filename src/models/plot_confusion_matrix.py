import argparse

from src.models.train_utils import *
from src.models.torch_dataset import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot confusion matrix')
    parser.add_argument(
        '--data_path', default="/media/gnocchi/Seagate Backup Plus Drive/NTU-RGB-D/"
    )
    parser.add_argument('--model_folder', default="./models/")
    parser.add_argument('--model_file')
    parser.add_argument('--evaluation_type')
    parser.add_argument('--model_type')
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--sub_sequence_length', default=20)
    parser.add_argument('--normalize_skeleton', default=True)
    parser.add_argument('--normalization_type')
    parser.add_argument('--kinematic_chain_skeleton', default=False)

    arg = parser.parse_args()

    # Hyper parameters
    data_path = arg.data_path
    model_folder = arg.model_folder
    model_file = arg.model_file
    evaluation_type = arg.evaluation_type
    model_type = arg.model_type
    batch_size = int(arg.batch_size)
    sub_sequence_length = int(arg.sub_sequence_length)
    normalize_skeleton = arg.normalize_skeleton == "True"
    normalization_type = arg.normalization_type
    kinematic_chain_skeleton = arg.kinematic_chain_skeleton == "True"

    if evaluation_type not in ["cross_subject", "cross_view"]:
        print("Error : Evaluation type not recognized")
        print("... Returning")

        exit()

    # Print summary
    print("\r\n\n\n========== PLOT CONFUSION MATRIX ==========")
    print("-> model_folder : " + model_folder)
    print("-> evaluation_type : " + evaluation_type)
    print("-> model_type : " + str(model_type))
    print("-> batch size : " + str(batch_size))
    print("-> sub_sequence_length : " + str(sub_sequence_length))
    print("-> normalize_skeleton : " + str(normalize_skeleton))
    print("-> normalization_type : " + str(normalization_type))
    print("-> kinematic chain skeleton : " + str(kinematic_chain_skeleton))
    print()

    # Create data loaders
    _, _, test_generator = create_data_loaders(data_path,
                                               evaluation_type,
                                               model_type,
                                               batch_size,
                                               sub_sequence_length,
                                               normalize_skeleton,
                                               normalization_type,
                                               augment_data=False,
                                               use_validation=False)

    if model_type == "VA-CNN":
        model = VACNN()
    elif model_type == "AS-CNN":
        model = ASCNN()
    elif model_type == "CNN3D":
        model = CNN3D()
    elif model_type == "FUSION":
        model = Fusion()
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

        plot_confusion_matrix(y_true, y_pred, classes, normalize=True,
                              title="Confusion matrix")
        plt.savefig(model_folder + str(model_type) + ".png")

        print("Accuracy over test set " + str(test_accuracy))

    # echo -en "\e[?25h"
    print("-> Done !")
