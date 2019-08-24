import argparse

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from src.models.data_loader import *
from src.models.train_utils import *


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots(figsize=(27, 27))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


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
    parser.add_argument('--normalize_skeleton', default=True)
    parser.add_argument('--normalization_type')
    parser.add_argument('--augment_data', default=True)
    parser.add_argument('--use_validation', default=True)
    parser.add_argument('--evaluate_test', default=True)

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
    normalize_skeleton = arg.normalize_skeleton == "True"
    normalization_type = arg.normalization_type
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
    print("-> optimizer : " + optimizer)
    print("-> learning rate : " + str(learning_rate))
    print("-> max epochs : " + str(epochs))
    print("-> batch size : " + str(batch_size))
    print("-> sub_sequence_length : " + str(sub_sequence_length))
    print("-> include_pose : " + str(include_pose))
    print("-> include_rgb : " + str(include_rgb))
    print("-> continuous_frames : " + str(continuous_frames))
    print("-> normalize_skeleton : " + str(normalize_skeleton))
    print("-> normalization_type : " + str(normalization_type))
    print("-> augment_data : " + str(augment_data))
    print("-> use_validation : " + str(use_validation))
    print("-> evaluate_test : " + str(evaluate_test))

    # Create data loader
    data_loader = DataLoader(model_type,
                             batch_size,
                             data_path,
                             evaluation_type,
                             sub_sequence_length,
                             continuous_frames,
                             normalize_skeleton,
                             normalization_type,
                             augment_data,
                             use_validation)

    if model_type == "GRU":
        model = FskDeepGRU().to(device)
    elif model_type == "STA-HANDS":
        model = STAHandsCNN(sub_sequence_length, include_pose, include_rgb).to(device)
    elif model_type == "VA-LSTM":
        model = VALSTM(sub_sequence_length).to(device)
    elif model_type == "VA-CNN":
        model = VACNN().to(device)
    else:
        print("Model type not recognized. Exiting")
        exit()

    model.load_state_dict(torch.load("./models/VA_CNN.pt"))
    model.eval()

    classes = ['drink water', 'eat meal/snack',
               'brushing teeth', 'brushing hair',
               'drop', 'pickup',
               'throw', 'sitting down',
               'standing up', 'clapping',
               'reading', 'writing',
               'tear up paper', 'wear jacket',
               'take off jacket', 'wear a shoe',
               'take off a shoe', "wear on glasses",
               'take off glasses', 'put on a hat/cap',
               'take off a hat/cap', 'cheer up',
               'hand waving', 'kicking something',
               'reach into pocket', 'hopping (one foot jumping)',
               'jump up', 'make a phone call/answer phone',
               'playing with phone/tablet', 'typing on keyboard',
               'pointing to something with finger', 'taking a selfie',
               'check time (watch)', 'rub hands together',
               'nod head/bow', 'shake head',
               'wipe face', 'salute',
               'put the palms together', 'cross hands in front',
               'sneeze/couggh', 'staggering',
               'falling', 'touch head',
               'touch chest', 'touch back',
               'touch neck', 'nausea or vomiting',
               'use a fan', 'punching/slapping other person',
               'kicking other person', 'pushing other person',
               'pat on back of other person', 'point finger at other person',
               'hugging other person', 'giving something to other person',
               'touch other person pocket', 'handshaking',
               'walk toward other', 'walk apart from each other']

    y_true, y_pred = confusion_test_set(model, data_loader)

    # y_true = [np.linspace(0, 60, num=60, endpoint=True)]
    # y_pred = [np.zeros(30, ), np.ones(30,)]

    y_true = np.int32(np.concatenate(y_true))
    y_pred = np.int32(np.concatenate(y_pred))

    print(y_true.shape)
    print(y_true)

    plot_confusion_matrix(y_true, y_pred, classes, normalize=True, title="Confusion matrix VA-CNN")

    plt.savefig('test.png')
    # plt.show()
