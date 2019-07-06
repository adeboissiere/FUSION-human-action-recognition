from click import progressbar
import time
import pickle

from src.models.models import *


def evaluate_accuracy_set(model, samples_list, batch_size, h5_dataset):
    batch_samples_list = [samples_list[x:x+batch_size] for x in range(0, len(samples_list), batch_size)]

    for batch_samples in batch_samples_list:
        for sample_name in batch_samples:
            skeleton = h5_dataset[sample_name]["skeleton"][:]  # shape (3, max_frame, num_joint=25, 2)
            hand_crops = h5_dataset[sample_name]["rgb"][:]  # shape (max_frame, n_hands = {2, 4}, crop_size, crop_size, 3)

            # Pad hand_crops if only one subject found
            if hand_crops.shape[1] == 2:
                pad = np.zeros(hand_crops.shape, dtype=hand_crops.dtype)
                hand_crops = np.concatenate((hand_crops, pad), axis=1)


def calculate_accuracy(model, Y_hat, Y):
    _, Y_hat = Y_hat.max(1)
    trues = (Y_hat == Y.long()) * 1
    trues = trues.cpu().numpy()

    accuracy = np.mean(trues)

    return accuracy


def train_model(model, data_loader, optimizer, learning_rate, epochs, output_folder):
    # Lists for plotting
    time_batch = []
    time_epoch = [0]
    loss_batch = []
    loss_epoch = []

    train_errors = []

    if optimizer == "ADAM":
        optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    elif optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr = learning_rate, weight_decay=0)
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
            X_skeleton, X_hands, Y = data_loader.next_batch()
            Y = torch.from_numpy(Y).to(device)

            out = model(X_skeleton, X_hands)

            loss = F.cross_entropy(out, Y.long())
            loss.backward()

            optimizer.step()

            # Save loss per batch
            time_batch.append(e + batch_idx / data_loader.n_batches)
            loss_batch.append(loss.item())

            # Accuracy over batch
            accuracy = calculate_accuracy(model, out, Y)
            batch_log = open(output_folder + "batch_log.txt", "a+")
            batch_log.write("[" + str(batch_idx) + "/" + str(data_loader.n_batches) + "] Accuracy : " + str(accuracy) + ", loss : " + str(loss.item()))
            batch_log.write("\r\n")
            batch_log.close()
            errors_temp.append(1 - accuracy)

            # Training mode
            progress_bar.update(1)

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
        log.write("In : " + str(end - start) + " seconds")
        log.write("\r\n")
        log.close()

        # Save model
        torch.save(model.state_dict(), str(output_folder) + "model" + str(e) + ".pt")

    return model