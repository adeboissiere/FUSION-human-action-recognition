import argparse
import datetime
import os

from src.models.data_loader import *
from src.models.train_utils import *

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
    parser.add_argument('--optimizer', default="ADAM")
    parser.add_argument('--learning_rate', default=1e-4)
    parser.add_argument('--epochs', default=40)
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--sub_sequence_length', default=20)
    parser.add_argument('--include_pose', default=True)
    parser.add_argument('--include_rgb', default=True)
    parser.add_argument('--continuous_frames', default=True)

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
    include_pose = arg.include_pose == "True"
    include_rgb = arg.include_rgb == "True"
    continuous_frames = arg.continuous_frames == "True"

    if evaluation_type not in ["cross_subject", "cross_view"]:
        print("Error : Evaluation type not recognized")
        print("... Returning")

        exit()

    # Print summary
    print("\r\n\n\n========== TRAIN MODEL ==========")
    print("-> h5 dataset folder path : " + data_path)
    print("-> output_folder : " + output_folder)
    print("-> evaluation_type : " + evaluation_type)
    print("-> model_type : " + str(model_type))
    print("-> optimizer : " + optimizer)
    print("-> learning rate : " + str(learning_rate))
    print("-> max epochs : " + str(epochs))
    print("-> batch size : " + str(batch_size))
    print("-> sub_sequence_length : " + str(sub_sequence_length))
    print("-> include_pose : " + str(include_pose))
    print("-> include_rgb : " + str(include_rgb))
    print("-> continuous_frames : " + str(continuous_frames))

    # Create data loader
    data_loader = DataLoader(batch_size, data_path, evaluation_type, sub_sequence_length, continuous_frames)
    X_skeleton, X_hands, Y = data_loader.next_batch()

    if model_type == "GRU":
        model = FskDeepGRU().to(device)
    elif model_type == "STA-HANDS":
        model = STAHandsCNN(60, include_pose, include_rgb).to(device)
    else:
        print("Model type not recognized. Exiting")
        exit()

    # Create folder for output files
    now = datetime.datetime.now()

    output_folder += str(model_type) + '_' + str(now.year) + '_' + str(now.month) + '_' + str(now.day) + \
                    '_' + str(now.hour) + 'h' + str(now.minute) + '_' + evaluation_type + '_'+ str(optimizer) + \
                    '_lr=' + str(learning_rate) + '_epochs=' + str(epochs) + '_batch=' + str(batch_size) +'_seq_len=' +\
                     str(sub_sequence_length) + '_cont_frames=' + str(continuous_frames) + '/'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    train_model(model, data_loader, optimizer, learning_rate, epochs, output_folder)

    print("-> Done !")
