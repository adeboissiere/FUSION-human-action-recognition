from click import progressbar
import time

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from src.models.VA_CNN import *
from src.models.AS_CNN import *
from src.models.torchvision_models import *
from src.models.cnn3D import *
from src.models.pose_ir_fusion import *

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
        # print("Normalized confusion matrix")
    else:
        None
        # print('Confusion matrix, without normalization')

    # print(cm)

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


def prime_X_cnn3d(X):
    X = torch.from_numpy(np.float32(X[0] / 255))

    # Normalize X
    normalize_values = torch.tensor([[0.43216, 0.394666, 0.37645],
                                     [0.22803, 0.22145, 0.216989]])  # [[mean], [std]]
    X = ((X.permute(0, 1, 3, 4, 2) - normalize_values[0]) / normalize_values[1]).permute(0, 1, 4, 2, 3)

    return X.permute(0, 2, 1, 3, 4).to(device)

def prime_X_fusion(X):
    X_skeleton = torch.from_numpy(np.float32(X[0])) / 255
    X_ir = torch.from_numpy(np.float32(X[1])) / 255

    # Normalize X_skeleton
    normalize_values = torch.tensor([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])  # [[mean], [std]]
    X_skeleton = ((X_skeleton.permute(0, 2, 3, 1) - normalize_values[0]) / normalize_values[1]).permute(0, 3, 1, 2)

    # Normalize X
    normalize_values = torch.tensor([[0.43216, 0.394666, 0.37645],
                                     [0.22803, 0.22145, 0.216989]])  # [[mean], [std]]
    X_ir = ((X_ir.permute(0, 1, 3, 4, 2) - normalize_values[0]) / normalize_values[1]).permute(0, 1, 4, 2, 3)

    return [X_skeleton.to(device), X_ir.permute(0, 2, 1, 3, 4).to(device)]


def calculate_accuracy(Y_hat, Y):
    _, Y_hat = Y_hat.max(1)
    trues = (Y_hat == Y.long()) * 1
    trues = trues.cpu().numpy()

    accuracy = np.mean(trues)

    return accuracy, Y_hat.cpu().numpy(), trues


def evaluate_validation_set(model, model_type, data_loader, output_folder):
    model.eval()
    average_accuracy = 0

    for batch_idx in range(data_loader.n_batches_val):
        X, Y = data_loader.next_batch_validation()
        Y = torch.from_numpy(Y).to(device)

        batch_size = X[0].shape[0]

        if model_type in ['CNN3D']:
            X = prime_X_cnn3d(X)

        elif model_type in ['FUSION']:
            X = prime_X_fusion(X)

        out = model(X)

        accuracy, _, _ = calculate_accuracy(out, Y)
        average_accuracy += accuracy * batch_size

        batch_log = open(output_folder + "batch_log.txt", "a+")
        batch_log.write("[VAL - " + str(batch_idx) + "/" + str(data_loader.n_batches_val) +
                        "] Accuracy : " + str(accuracy))
        batch_log.write("\r\n")
        batch_log.close()

    return average_accuracy / len(data_loader.validation_samples)


def evaluate_test_set(model, model_type, data_loader, output_folder):
    model.eval()
    average_accuracy = 0

    y_true = []
    y_pred = []

    for batch_idx in range(data_loader.n_batches_test):
        X, Y = data_loader.next_batch_test()
        Y = torch.from_numpy(Y).to(device)
        
        batch_size = X[0].shape[0]

        if model_type in ['CNN3D']:
            X = prime_X_cnn3d(X)

        elif model_type in ['FUSION']:
            X = prime_X_fusion(X)

        out = model(X)

        accuracy, Y_hat, Y = calculate_accuracy(out, Y)

        # For confusion matrix
        y_true.append(Y)
        y_pred.append(Y_hat)

        average_accuracy += accuracy * batch_size

        batch_log = open(output_folder + "batch_log.txt", "a+")
        batch_log.write("[TEST - " + str(batch_idx) + "/" + str(data_loader.n_batches_test) +
                        "] Accuracy : " + str(accuracy))
        batch_log.write("\r\n")
        batch_log.close()

    return average_accuracy / len(data_loader.testing_samples), y_true, y_pred


def train_model(model,
                model_type,
                data_loader,
                optimizer,
                learning_rate,
                weight_decay,
                gradient_threshold,
                epochs,
                evaluate_test,
                output_folder):

    # Lists for plotting
    time_batch = []
    time_epoch = [0]
    loss_batch = []
    loss_epoch = []

    train_errors = []

    if optimizer == "ADAM":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        print("Optimizer not recognized ... exit()")
        exit()

    # Progress bar
    progress_bar = progressbar(iterable=None, length=epochs * data_loader.n_batches)

    for e in range(epochs):
        start = time.time()

        model.train()
        errors_temp = []

        for batch_idx in range(data_loader.n_batches):
            X, Y = data_loader.next_batch()
            Y = torch.from_numpy(Y).to(device)

            if model_type in ['CNN3D']:
                X = prime_X_cnn3d(X)

            elif model_type in ['FUSION']:
                X = prime_X_fusion(X)

            out = model(X)

            loss = F.cross_entropy(out, Y.long())
            loss.backward()

            # Gradient clipping
            if gradient_threshold > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_threshold)

            optimizer.step()

            # Save loss per batch
            time_batch.append(e + batch_idx / data_loader.n_batches)
            loss_batch.append(loss.item())

            # Accuracy over batch
            accuracy, _, _ = calculate_accuracy(out, Y)
            batch_log = open(output_folder + "batch_log.txt", "a+")
            batch_log.write("[" + str(e) + " - " + str(batch_idx) + "/" + str(data_loader.n_batches) +
                            "] Accuracy : " + str(accuracy) + ", loss : " + str(loss.item()))
            batch_log.write("\r\n")
            batch_log.close()
            errors_temp.append(1 - accuracy)

            # Training mode
            progress_bar.update(1)

        if data_loader.use_validation:
            with torch.no_grad():
                validation_accuracy = evaluate_validation_set(model, model_type, data_loader, output_folder)

        if evaluate_test:
            with torch.no_grad():
                test_accuracy, y_true, y_pred = evaluate_test_set(model, model_type, data_loader, output_folder)

                # Plot confusion matrix per epoch
                y_true = np.int32(np.concatenate(y_true))
                y_pred = np.int32(np.concatenate(y_pred))

                plot_confusion_matrix(y_true, y_pred, classes, normalize=True, title="Confusion matrix " + str(CNN3D))
                plt.savefig(output_folder + str(model_type) + str(e) + ".png")

        # Save loss per epoch
        time_epoch.append(e + 1)
        loss_epoch.append(
            sum(loss_batch[e * data_loader.n_batches: (e + 1) * data_loader.n_batches]) / data_loader.n_batches)

        # Average accuracy over epoch
        train_errors.append(np.mean(errors_temp))

        # Write log data
        # Log file (open and close after each epoch so we can read realtime
        end = time.time()
        log = open(output_folder + "log.txt", "a+")
        log.write("Epoch : " + str(e) + ", err train : " + str(np.mean(errors_temp)))
        if data_loader.use_validation:
            log.write(", val accuracy : " + str(validation_accuracy))
        if evaluate_test:
            log.write(", test accuracy : " + str(test_accuracy) + " ")
        log.write("in : " + str(end - start) + " seconds")
        log.write("\r\n")
        log.close()

        # Save model
        torch.save(model.state_dict(), str(output_folder) + "model" + str(e) + ".pt")

    return model
