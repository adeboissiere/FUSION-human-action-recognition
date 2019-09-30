from click import progressbar
import time

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from src.models.VA_CNN import *
from src.models.AS_CNN import *
from src.models.torchvision_models import *
from src.models.cnn3D import *
from src.models.pose_ir_fusion import *


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


def calculate_accuracy(Y_hat, Y):
    _, Y_hat = Y_hat.max(1)
    trues = (Y_hat == Y.long()) * 1
    trues = trues.cpu().numpy()

    accuracy = np.mean(trues)

    return accuracy, Y_hat.cpu().numpy(), trues


def evaluate_set(model, model_type, data_loader, output_folder, set_name):
    model.eval()

    average_accuracy = 0

    y_true = []
    y_pred = []

    for batch_idx, batch in enumerate(data_loader):
        print(str(batch_idx) + " / " + str(len(data_loader)))
        X = batch[0]
        Y = batch[1].to(device)

        batch_size = Y.shape[0]

        if model_type == "CNN3D":
            X = prime_X_cnn3d(X).to(device)

        elif model_type == "FUSION":
            X = prime_X_fusion(X)

        out = model(X)

        accuracy, Y_hat, Y = calculate_accuracy(out, Y)
        average_accuracy += accuracy * batch_size

        y_true.append(Y)
        y_pred.append(Y_hat)

        batch_log = open(output_folder + "batch_log.txt", "a+")
        batch_log.write("[" + str(set_name) + " - " + str(batch_idx) + "/" + str(len(data_loader)) +
                        "] Accuracy : " + str(accuracy))
        batch_log.write("\r\n")
        batch_log.close()

    return average_accuracy / len(data_loader), y_true, y_pred


def train_model_new(model,
                    model_type,
                    optimizer,
                    learning_rate,
                    weight_decay,
                    gradient_threshold,
                    epochs,
                    evaluate_test,
                    output_folder,
                    train_generator,
                    test_generator,
                    validation_generator = None):

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

    for e in range(epochs):
        model.train()
        errors_temp = []

        start = time.time()

        start_batch = time.time()
        for batch_idx, batch in enumerate(train_generator):
            # BATCH TRAINING
            print(str(e) + " - " + str(batch_idx) + "/" + str(len(train_generator)))
            X = batch[0]
            Y = batch[1].to(device)

            if model_type == "CNN3D":
                X = prime_X_cnn3d(X)

            elif model_type == "FUSION":
                X = prime_X_fusion(X)

            out = model(X)

            loss = F.cross_entropy(out, Y.long())
            loss.backward()

            # Gradient clipping
            if gradient_threshold > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_threshold)

            optimizer.step()

            # Save loss per batch
            time_batch.append(e + batch_idx / len(train_generator))
            loss_batch.append(loss.item())

            # Accuracy over batch
            accuracy, _, _ = calculate_accuracy(out, Y)
            batch_log = open(output_folder + "batch_log.txt", "a+")
            batch_log.write("[" + str(e) + " - " + str(batch_idx) + "/" + str(len(train_generator)) +
                            "] Accuracy : " + str(accuracy) + ", loss : " + str(loss.item()))
            batch_log.write("\r\n")
            batch_log.close()
            errors_temp.append(1 - accuracy)

            print("Batch took : " + str(time.time() - start_batch) + "s")
            start_batch = time.time()

        # VALIDATION STEP
        if validation_generator is not None:
            with torch.no_grad():
                validation_accuracy, _, _ = evaluate_set(model,
                                                         model_type,
                                                         validation_generator,
                                                         output_folder,
                                                         "VAL")

        # TEST STEP
        if evaluate_test:
            with torch.no_grad():
                test_accuracy, y_true, y_pred = evaluate_set(model,
                                                             model_type,
                                                             test_generator,
                                                             output_folder,
                                                             "TEST")

                # Plot confusion matrix per epoch
                y_true = np.int32(np.concatenate(y_true))
                y_pred = np.int32(np.concatenate(y_pred))

                plot_confusion_matrix(y_true, y_pred, classes, normalize=True,
                                      title="Confusion matrix")
                plt.savefig(output_folder + str(model_type) + str(e) + ".png")

        # Save loss per epoch
        time_epoch.append(e + 1)
        loss_epoch.append(
            sum(loss_batch[e * len(train_generator): (e + 1) * len(train_generator)]) / len(train_generator))

        # Average accuracy over epoch
        train_errors.append(np.mean(errors_temp))

        # Write log data
        # Log file (open and close after each epoch so we can read realtime
        end = time.time()
        log = open(output_folder + "log.txt", "a+")
        log.write("Epoch : " + str(e) + ", err train : " + str(np.mean(errors_temp)))
        if validation_generator is not None:
            log.write(", val accuracy : " + str(validation_accuracy))
        if evaluate_test:
            log.write(", test accuracy : " + str(test_accuracy) + " ")
        log.write("in : " + str(end - start) + " seconds")
        log.write("\r\n")
        log.close()

        # Save model
        torch.save(model.state_dict(), str(output_folder) + "model" + str(e) + ".pt")

    return model
