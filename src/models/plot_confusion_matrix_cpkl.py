"""
Computes the confusion matrix for a trained model. Takes as input important parameters such as the benchmark studied.
A confusion matrix in .png format is saved in the trained model folder provided.

Plotting the confusion matrix is best called using the provided Makefile provided.

>>> make confusion_matrix_cpkl \\
    MODEL_FOLDER=X \\
    CPKL_FILE=X \\


With the parameters taking from the following values :
    - MODEL_FOLDER:
        Output path to save models and log files. A folder inside that path will be automatically created. Default
        location is *./models/*
    - CPKL_FILE:
        Pickle file containing ground truthes and predictions.


"""
import argparse

from src.models.train_utils import *
from sklearn.metrics import cohen_kappa_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot confusion matrix')
    parser.add_argument('--model_folder', default="./models/")
    parser.add_argument('--cpkl_file', default=None)

    arg = parser.parse_args()

    # Hyper parameters
    model_folder = arg.model_folder
    cpkl_file = arg.cpkl_file

    if cpkl_file is None:
        print("Error : Must specify file name")
        print("... Returning")

        exit()

    # Print summary
    print("\r\n\n\n========== PLOT CONFUSION MATRIX ==========")
    print("-> model_folder : " + model_folder)
    print("-> cpkl_file : " + str(cpkl_file))

    pickle_test = open(model_folder + cpkl_file, 'rb')
    data = pickle.load(pickle_test)
    pickle_test.close()
    y_true, y_pred = data

    evaluate_per_action_type(y_true, y_pred)

    plot_confusion_matrix(y_true, y_pred, classes, normalize=True, title="Confusion matrix")
    plt.savefig(model_folder  + "cm.png")

    print("Cohen's kappa : " + str(cohen_kappa_score(y_true, y_pred)))

    # echo -en "\e[?25h"
    print("-> Done !")
