import argparse
from src.models.data_loader import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument(
        '--data_path', default="/media/gnocchi/Seagate Backup Plus Drive/NTU-RGB-D/"
    )
    parser.add_argument(
        '--output_folder', default="./models/"
    )
    parser.add_argument('--evaluation_type')
    parser.add_argument('--model_type')
    parser.add_argument('--optimizer', default="SGD")
    parser.add_argument('--learning_rate', default=1e-9)
    parser.add_argument('--epochs', default=40)
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--sub_sequence_length', default=20)

    arg = parser.parse_args()

    # Hyper parameters
    data_path = arg.data_path
    output_folder = arg.output_folder
    evaluation_type = arg.evaluation_type
    model_type = arg.model_type
    optimizer = arg.optimizer
    learning_rate = float(arg.learning_rate)
    epochs = int(arg.epochs)
    batch_size = int(arg.batch_size)
    sub_sequence_length = int(arg.sub_sequence_length)

    if evaluation_type not in ["cross_subject", "cross_view"]:
        print("Error : Evaluation type not recognized")
        print("... Returning")

        exit()

    # Print summary
    print("========== TRAIN MODEL ==========")
    print("-> h5 dataset folder path : " + data_path)
    print("-> output folder : " + output_folder)
    print("-> evaluation type : " + evaluation_type)
    print("-> optimizer : " + optimizer)
    print("-> learning rate : " + str(learning_rate))
    print("-> max epochs : " + str(epochs))
    print("-> batch size : " + str(batch_size))
    print("-> sub_sequence_length : " + str(sub_sequence_length))

    # Create dataloader
    data_loader = DataLoader(batch_size, data_path, evaluation_type)
    data_loader.next_batch()
    data_loader.next_batch()
    print("-> Done !")
