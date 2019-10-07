import argparse
import datetime
import os

from src.models.train_utils import *

from src.models.torch_dataset import *

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
    parser.add_argument('--use_pose', default=False)
    parser.add_argument('--use_ir', default=False)
    parser.add_argument('--pretrained', default=False)
    parser.add_argument('--use_cropped_IR', default=False)
    parser.add_argument('--optimizer', default="ADAM")
    parser.add_argument('--learning_rate', default=1e-4)
    parser.add_argument('--weight_decay', default=0)
    parser.add_argument('--gradient_threshold', default=0)
    parser.add_argument('--epochs', default=40)
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--accumulation_steps', default=1)
    parser.add_argument('--sub_sequence_length', default=20)
    parser.add_argument('--normalize_skeleton', default=True)
    parser.add_argument('--normalization_type')
    parser.add_argument('--kinematic_chain_skeleton', default=False)
    parser.add_argument('--augment_data', default=True)
    parser.add_argument('--use_validation', default=True)
    parser.add_argument('--evaluate_test', default=True)

    arg = parser.parse_args()

    # Hyper parameters
    data_path = arg.data_path
    output_folder = arg.output_folder
    evaluation_type = arg.evaluation_type
    model_type = arg.model_type
    use_pose = arg.use_pose == "True"
    use_ir = arg.use_ir == "True"
    pretrained = arg.pretrained == "True"
    use_cropped_IR = arg.use_cropped_IR == "True"
    optimizer = arg.optimizer
    learning_rate = float(arg.learning_rate)
    weight_decay = float(arg.weight_decay)
    gradient_threshold = float(arg.gradient_threshold)
    epochs = int(arg.epochs)
    batch_size = int(arg.batch_size)
    accumulation_steps = int(arg.accumulation_steps)
    sub_sequence_length = int(arg.sub_sequence_length)
    normalize_skeleton = arg.normalize_skeleton == "True"
    normalization_type = arg.normalization_type
    kinematic_chain_skeleton = arg.kinematic_chain_skeleton == "True"
    augment_data = arg.augment_data == "True"
    use_validation = arg.use_validation == "True"
    evaluate_test = arg.evaluate_test == "True"

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
    if model_type == "FUSION":
        print("-> use pose : " + str(use_pose))
        print("-> use ir : " + str(use_ir))
        print("-> pretrained : " + str(pretrained))
        print("-> use cropped ir : " + str(use_cropped_IR))
    print("-> optimizer : " + optimizer)
    print("-> learning rate : " + str(learning_rate))
    print("-> weight decay : " + str(weight_decay))
    print("-> gradient threshold : " + str(gradient_threshold))
    print("-> max epochs : " + str(epochs))
    print("-> batch size : " + str(batch_size))
    print("-> accumulation steps : " + str(accumulation_steps))
    print("-> sub_sequence_length : " + str(sub_sequence_length))
    print("-> normalize_skeleton : " + str(normalize_skeleton))
    print("-> normalization_type : " + str(normalization_type))
    print("-> kinematic chain skeleton : " + str(kinematic_chain_skeleton))
    print("-> augment_data : " + str(augment_data))
    print("-> use_validation : " + str(use_validation))
    print("-> evaluate_test : " + str(evaluate_test))
    print()

    # Keep different trainings consistent
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # Create data loaders
    train_generator, validation_generator, test_generator = create_data_loaders(data_path,
                                                                                evaluation_type,
                                                                                model_type,
                                                                                use_pose,
                                                                                use_ir,
                                                                                use_cropped_IR,
                                                                                batch_size,
                                                                                sub_sequence_length,
                                                                                normalize_skeleton,
                                                                                normalization_type,
                                                                                augment_data,
                                                                                use_validation)

    if model_type == "VA-CNN":
        model = VACNN()
    elif model_type == "AS-CNN":
        model = ASCNN()
    elif model_type == "CNN3D":
        model = CNN3D()
    elif model_type == "FUSION":
        model = FUSION(use_pose, use_ir, pretrained)
    else:
        print("Model type not recognized. Exiting")
        exit()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    '''
    with torch.no_grad():
        X, Y = data_loader.next_batch()
        X = prime_X_fusion(X)
        model(X)
    exit()
    '''

    # Create folder for output files
    now = datetime.datetime.now()

    output_folder += str(model_type) + '_' + str(now.year) + '_' + str(now.month) + '_' + str(now.day) + \
                    '_' + str(now.hour) + 'h' + str(now.minute)

    if model_type == "FUSION":
        output_folder += "_pose=" + str(use_pose) + \
                         "_ir=" + str(use_ir) + \
                         "_pretrained=" + str(pretrained) + \
                         "_cropped_IR=" + str(use_cropped_IR)

    output_folder += '_' + evaluation_type + '_' + str(optimizer) + \
                     '_lr=' + str(learning_rate) + \
                     '_wd=' + str(weight_decay) + \
                     '_gt=' + str(gradient_threshold) + \
                     '_epochs=' + str(epochs) + \
                     '_batch=' + str(batch_size) + \
                     '_steps=' + str(accumulation_steps) + \
                     '_seq_len=' +\
                     str(sub_sequence_length) +  '/'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    train_model_new(model,
                    model_type,
                    optimizer,
                    learning_rate,
                    weight_decay,
                    gradient_threshold,
                    epochs,
                    accumulation_steps,
                    evaluate_test,
                    output_folder,
                    train_generator,
                    test_generator,
                    validation_generator)

    # echo -en "\e[?25h"
    print("-> Done !")
