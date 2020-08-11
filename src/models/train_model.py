"""
The main file for the *src.models* module. Takes as input the different hyperparameters and starts training the model.
The model is saved after every epoch. A `batch_log.txt` keeps the record of the accuracy and loss of each batch.
A `log.txt` keeps a record of the accuracy of the train-val-test sets after each epoch.

**Note** that although we compute the test set at each epoch, we take the final decision based on the validation set
only.

Training is best called using the provided Makefile provided.

>>> make train \\
    PROCESSED_DATA_PATH=X \\
    MODEL_FOLDER=X \\
    EVALUATION_TYPE=X \\
    MODEL_TYPE=X \\
    USE_POSE=X \\
    USE_IR=X \\
    PRETRAINED=X \\
    USE_CROPPED_IR=X \\
    LEARNING_RATE=X \\
    WEIGHT_DECAY=X \\
    GRADIENT_THRESHOLD=X \\
    EPOCHS=X \\
    BATCH_SIZE=X \\
    ACCUMULATION_STEPS=X \\
    SUB_SEQUENCE_LENGTH=X \\
    AUGMENT_DATA=X \\
    MIRROR_SKELETON=X \\
    EVALUATE_TEST=X \\
    SEED=X


With the parameters taking from the following values :
    - PROCESSED_DATA_PATH:
        Path to h5 files. Default location is *./data/processed/*
    - MODEL_FOLDER:
        Output path to save models and log files. A folder inside that path will be automatically created. Default
        location is *./models/*
    - EVALUATION_TYPE:
        [cross_subject | cross_view]
    - MODEL_TYPE:
        [FUSION]
    - USE_POSE:
        [True, False]
    - USE_IR:
        [True, False]
    - PRETRAINED:
        [True, False]
    - USE_CROPPED_IR:
        [True, False]
    - LEARNING_RATE:
        Real positive number.
    - WEIGHT_DECAY:
        Real positive number. If 0, then no weight decay is applied.
    - EPOCHS:
        Whole positive number above 1.
    - BATCH_SIZE:
        Whole positive number above 1.
    - GRADIENT_THRESHOLD:
        Real positive number. If 0, then no threshold is applied
    - ACCUMULATION_STEPS:
        Accumulate gradient across batches. This is a trick to virtually train larger batches on modest architectures.
    - SUB_SEQUENCE_LENGTH:
        [1 .. 20]
        Specifies the number of frames to take from a complete IR sequence.
    - AUGMENT_DATA
        [True, False]
    - MIRROR_SKELETON
        [True, False]
    - EVALUATE_TEST
        [True, False]
    - SEED
        Positive whole number. Used to make training replicable.

"""

import argparse
import datetime
import os

from src.models.train_utils import *

from src.models.gen_data_loaders import *

if __name__ == '__main__':
    # Parser to gather hyperparameters
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--data_path')
    parser.add_argument('--output_folder', default="./models/")
    parser.add_argument('--evaluation_type')
    parser.add_argument('--model_type')
    parser.add_argument('--use_pose', default=False)
    parser.add_argument('--use_ir', default=False)
    parser.add_argument('--pretrained', default=False)
    parser.add_argument('--use_cropped_IR', default=False)
    parser.add_argument('--fusion_scheme', default="CONCAT")
    parser.add_argument('--optimizer', default="ADAM")
    parser.add_argument('--learning_rate', default=1e-4)
    parser.add_argument('--weight_decay', default=0)
    parser.add_argument('--gradient_threshold', default=0)
    parser.add_argument('--epochs', default=40)
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--accumulation_steps', default=1)
    parser.add_argument('--sub_sequence_length', default=20)
    parser.add_argument('--augment_data', default=True)
    parser.add_argument('--mirror_skeleton', default=False)
    parser.add_argument('--evaluate_test', default=True)
    parser.add_argument('--seed', default=0)
    arg = parser.parse_args()

    # Extract hyperparameters
    data_path = arg.data_path
    output_folder = arg.output_folder
    evaluation_type = arg.evaluation_type
    model_type = arg.model_type
    use_pose = arg.use_pose == "True"
    use_ir = arg.use_ir == "True"
    pretrained = arg.pretrained == "True"
    use_cropped_IR = arg.use_cropped_IR == "True"
    fusion_scheme = arg.fusion_scheme
    optimizer = arg.optimizer
    learning_rate = float(arg.learning_rate)
    weight_decay = float(arg.weight_decay)
    gradient_threshold = float(arg.gradient_threshold)
    epochs = int(arg.epochs)
    batch_size = int(arg.batch_size)
    accumulation_steps = int(arg.accumulation_steps)
    sub_sequence_length = int(arg.sub_sequence_length)
    augment_data = arg.augment_data == "True"
    mirror_skeleton = arg.mirror_skeleton == "True"
    evaluate_test = arg.evaluate_test == "True"
    seed = int(arg.seed)

    # Check evaluation type is a known benchmark
    if evaluation_type not in ["cross_subject", "cross_view"]:
        print("Error : Evaluation type not recognized")
        print("... Returning")

        exit()

    # Print summary
    print("\r\n\n\n========== SUMMARY ==========")
    print("-> h5 dataset folder path : " + data_path)
    print("-> output_folder : " + output_folder)
    print("-> evaluation_type : " + evaluation_type)
    print("-> model_type : " + str(model_type))
    if model_type == "FUSION":
        print("-> use pose : " + str(use_pose))
        print("-> use ir : " + str(use_ir))
        print("-> pretrained : " + str(pretrained))
        print("-> use cropped ir : " + str(use_cropped_IR))
        print("-> fusion scheme : " + str(fusion_scheme))
    print("-> optimizer : " + optimizer)
    print("-> learning rate : " + str(learning_rate))
    print("-> weight decay : " + str(weight_decay))
    print("-> gradient threshold : " + str(gradient_threshold))
    print("-> max epochs : " + str(epochs))
    print("-> batch size : " + str(batch_size))
    print("-> accumulation steps : " + str(accumulation_steps))
    print("-> sub_sequence_length : " + str(sub_sequence_length))
    print("-> augment_data : " + str(augment_data))
    print("-> mirror skeleton : " + str(mirror_skeleton))
    print("-> evaluate_test : " + str(evaluate_test))
    print("-> seed : " + str(seed))
    print()

    # Keep different trainings consistent
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Create data loaders
    train_generator, validation_generator, test_generator = create_data_loaders(data_path,
                                                                                evaluation_type,
                                                                                model_type,
                                                                                use_pose,
                                                                                use_ir,
                                                                                use_cropped_IR,
                                                                                batch_size,
                                                                                sub_sequence_length,
                                                                                augment_data,
                                                                                mirror_skeleton)

    # Create model
    if model_type == "FUSION":
        model = FUSION(use_pose, use_ir, pretrained, fusion_scheme)
    else:
        print("Model type not recognized. Exiting ...")
        exit()

    # If multiple GPUs are available (not guaranteed to work)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    # Push model to first GPU if available
    model.to(device)

    '''
    with torch.no_grad():
        X, Y = data_loader.next_batch()
        X = prime_X_fusion(X)
        model(X)
    exit()
    '''

    # Create folder name for output files
    now = datetime.datetime.now()

    if model_type == "FUSION":
        output_folder += "pretrained=" + str(pretrained) + \
                         "_cropped_IR=" + str(use_cropped_IR) + \
                         "_fusion_scheme=" + str(fusion_scheme)

    output_folder += '_' + evaluation_type + \
                     '_seq_len=' + str(sub_sequence_length) + \
                     '_aug=' + str(augment_data) + \
                     '_mirror_s=' + str(mirror_skeleton) + \
                     '_seed=' + str(seed) + \
                      '/'

    # Create folder if does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Start training
    train_model(model,
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
