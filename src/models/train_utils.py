from click import progressbar
import time

from src.models.VA_CNN import *
from src.models.AS_CNN import *


def calculate_accuracy(Y_hat, Y):
    _, Y_hat = Y_hat.max(1)
    trues = (Y_hat == Y.long()) * 1
    trues = trues.cpu().numpy()

    accuracy = np.mean(trues)

    return accuracy


def evaluate_validation_set(model, data_loader, output_folder):
    model.eval()
    average_accuracy = 0

    for batch_idx in range(data_loader.n_batches_val):
        X, Y = data_loader.next_batch_validation()
        Y = torch.from_numpy(Y).to(device)

        out = model(X)

        accuracy = calculate_accuracy(out, Y)
        average_accuracy += accuracy * X[0].shape[0]

        batch_log = open(output_folder + "batch_log.txt", "a+")
        batch_log.write("[VAL - " + str(batch_idx) + "/" + str(data_loader.n_batches_val) +
                        "] Accuracy : " + str(accuracy))
        batch_log.write("\r\n")
        batch_log.close()

    return average_accuracy / len(data_loader.validation_samples)


def evaluate_test_set(model, data_loader, output_folder):
    model.eval()
    average_accuracy = 0

    for batch_idx in range(data_loader.n_batches_test):
        X, Y = data_loader.next_batch_test()
        Y = torch.from_numpy(Y).to(device)

        out = model(X)

        accuracy = calculate_accuracy(out, Y)
        average_accuracy += accuracy * X[0].shape[0]

        batch_log = open(output_folder + "batch_log.txt", "a+")
        batch_log.write("[TEST - " + str(batch_idx) + "/" + str(data_loader.n_batches_test) +
                        "] Accuracy : " + str(accuracy))
        batch_log.write("\r\n")
        batch_log.close()

    return average_accuracy / len(data_loader.testing_samples)


def confusion_test_set(model, data_loader):
    model.eval()
    y_true = []
    y_pred = []

    for batch_idx in range(data_loader.n_batches_test):
        print(str(batch_idx) + " / " + str(data_loader.n_batches_test))
        X, Y = data_loader.next_batch_test()
        Y = torch.from_numpy(Y).to(device)

        Y_hat = model(X)
        _, Y_hat = Y_hat.max(1)

        # Appends np arrays of shape (batch_size, )
        y_true.append(Y.cpu().numpy())
        y_pred.append(Y_hat.cpu().numpy())

        print(Y.cpu().numpy())
        print(Y_hat.cpu().numpy())

    return y_true, y_pred


def train_model(model,
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
            accuracy = calculate_accuracy(out, Y)
            batch_log = open(output_folder + "batch_log.txt", "a+")
            batch_log.write("[" + str(e) + " - " + str(batch_idx) + "/" + str(data_loader.n_batches) +
                            "] Accuracy : " + str(accuracy) + ", loss : " + str(loss.item()))
            batch_log.write("\r\n")
            batch_log.close()
            errors_temp.append(1 - accuracy)

            # Training mode
            progress_bar.update(1)

        if data_loader.use_validation:
            validation_accuracy = evaluate_validation_set(model, data_loader, output_folder)

        if evaluate_test:
            test_accuracy = evaluate_test_set(model, data_loader, output_folder)

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
