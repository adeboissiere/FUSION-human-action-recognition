import torch
from torch import nn, optim
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F

from PIL import Image

from src.utils.joints import *
from src.models.device import *


class FskCNN(nn.Module):
    def __init__(self):
        super(FskCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 1, kernel_size = (9, 3), padding = (4, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = (9, 3), padding = (4, 1)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 1024, kernel_size = (5, 82)),
            nn.ReLU()
        )

    def forward(self, X):
        """ Forward propagation of f_sk module

        :param X: shape (batch_size, 3, seq_len, 330)
        :return: 1x1x1024 feature map
        """
        out = self.layer1(X)
        out = self.layer2(out)
        out = self.layer3(out)

        return out  # shape (batch_size, 1024, 1, 1)


class FskDeepGRU(nn.Module):
    def __init__(self):
        super(FskDeepGRU, self).__init__()
        self.gru = nn.GRU(input_size = 25 * 3 * 2,
                          hidden_size = 150,
                          num_layers = 3,
                          batch_first = True,
                          dropout = 0.5,
                          bidirectional = False)

        self.fc = nn.Linear(150, 60)

    def forward(self, X_skeleton, X_hands):
        X_skeleton = np.float32(X_skeleton.transpose(0, 2, 3, 1, 4))  # shape (batch_size, seq_len, n_joints, 3, 2)

        batch_size = X_skeleton.shape[0]
        seq_len = X_skeleton.shape[1]

        X_skeleton = X_skeleton.reshape(batch_size, seq_len, 25 * 3 * 2)  # shape (batch_size, seq_len, 25 * 3 * 2)
        X_skeleton = torch.from_numpy(X_skeleton).to(device)

        out, _ = self.gru(X_skeleton)  # shape (batch_size, seq_len, 150)

        predictions = []
        for t in range(seq_len):
            pred_t = self.fc(out[:, t, :])  # shape (batch_size, 60)
            pred_t = F.log_softmax(pred_t, dim = 1)  # shape (batch_size, 60)
            predictions.append(pred_t)

        predictions = torch.stack(predictions, dim = 1)  # shape (batch_size, seq_len, 60)
        out = torch.mean(predictions, dim = 1)  # shape (batch_size, 60)

        return out


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class Fg(nn.Module):
    def __init__(self, model_name):
        """ F_g module (see paper). Only AlexNet and inception_v3 supported for now.

        :param model_name: type of trained model to extract feature of hands
        - AlexNet : 61 100 840 parameters
        - ResNet18 : 11 689 512 parameters
        - VGG11 : 132 863 336 parameters
        - SqueezeNet1_0 : 1 248 424 parameters
        - SqueezeNet1_1 : 1 235 496 parameters

        """
        super(Fg, self).__init__()
        self.trained_cnn = None
        self.model_name = model_name

        input_size = 224
        output_size = 2048
        feature_extract = True

        if model_name == "alexnet":
            self.trained_cnn = models.alexnet(pretrained=True)
            set_parameter_requires_grad(self.trained_cnn, feature_extract)
            num_ftrs = self.trained_cnn.classifier[6].in_features
            self.trained_cnn.classifier[6] = nn.Linear(num_ftrs, output_size)

        elif model_name == "inception":
            self.trained_cnn = models.inception_v3(pretrained=True)
            set_parameter_requires_grad(self.trained_cnn, feature_extract)
            # Handle the auxilary net
            num_ftrs = self.trained_cnn.AuxLogits.fc.in_features
            self.trained_cnn.AuxLogits.fc = nn.Linear(num_ftrs, output_size)
            # Handle the primary net
            num_ftrs = self.trained_cnn.fc.in_features
            self.trained_cnn.fc = nn.Linear(num_ftrs, output_size)
            input_size = 299

        else:
            print("Invalid model name for f_g module, exiting...")
            exit()

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            normalize
        ])

    def forward(self, X_hands):
        """ Forward propagation of f_g module (inception_v3).
        It takes about 4.5s to apply transformation to all images for batch_size = 32, seq_len = 20

        :param X_hands: shape (batch_size, seq_len, 4, crop_size, crop_size, 3)
        :return:
        """
        batch_size = X_hands.shape[0]
        seq_len = X_hands.shape[1]

        v_tji = [] # See paper for notation
        for hand in range(4):
            v_tj = []
            for t in range(seq_len):
                batch = []

                for b in range(batch_size):
                    # Transform to PIL format
                    hand_crop = Image.fromarray(X_hands[b, t, hand])

                    # Apply transformation
                    hand_crop = self.transform(hand_crop)
                    batch.append(hand_crop)

                # Create batch for given hand and fixed t
                X = torch.stack(batch).to(device)  # shape (batch_size, 3, H, W)

                # Compute feature vector
                out = self.trained_cnn(X)  # shape (batch_size, output_size)

                # Append to list
                if self.model_name == "alexnet":
                    v_tj.append(out)
                elif self.model_name == "inception":
                    v_tj.append(out[0])

            # Stack list to new dimension
            X = torch.stack(v_tj)  # shape (seq_len, batch_size, output_size)
            v_tji.append(X)

        X = torch.stack(v_tji)  # shape (4, seq_len, batch_size, output_size)

        return X.permute(2, 1, 3, 0)


class Fp(nn.Module):
    def __init__(self):
        super(Fp, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2048, 256),
            nn.Sigmoid(),
            nn.Linear(256, 4),
            nn.Sigmoid()
        )

    def forward(self, X):
        return self.mlp(X)


class Fu(nn.Module):
    def __init__(self):
        super(Fu, self).__init__()
        self.linear_ReLU = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU()
        )

    def forward(self, X):
        return self.linear_ReLU(X)


class FpPrime(nn.Module):
    def __init__(self):
        super(FpPrime, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(80 + 1024, 512),
            nn.Sigmoid(),
            nn.Linear(512, 20),
            nn.Sigmoid()
        )

    def forward(self, X):
        return self.mlp(X)


class STAHandsCNN(nn.Module):
    def __init__(self, n_classes, include_pose, include_rgb):
        super(STAHandsCNN, self).__init__()
        self.n_classes = n_classes
        self.include_pose = include_pose
        self.include_rgb = include_rgb

        # Pose network
        self.fsk = FskCNN()

        # Glimpse sensor
        self.fg = Fg("inception")

        # Spatial attention network
        self.fp = Fp()

        # LSTM cell
        self.lstm = nn.LSTMCell(input_size = 2048, hidden_size = 1024)

        # Feature extractor
        self.fu = Fu()

        # Temporal attention network
        self.fp_prime = FpPrime()

        # Output & prediction
        self.fy_pose = nn.Linear(1024, 60)
        self.fy_rgb = nn.Linear(1024, 60)
        self.fy_fuse = nn.Linear(2048, 60)

    def forward(self, X_skeleton, X_hands):
        """ Forward propagation of STAHandsCNN

        :param X_skeleton: shape (batch_size, 3, sub_sequence_length, num_joints, 2)
        :param X_hands: shape (batch_size, sub_sequence_length, 4, crop_size, crop_size, 3)
        :return:
        """
        batch_size = X_skeleton.shape[0]
        seq_len = X_skeleton.shape[2]

        s = torch.zeros((batch_size, 1024, 1, 1)).to(device)
        u_tilde = None

        if self.include_pose:
            # ===== Convolutional pose features (f_sk) =====
            # 1. Transform X_skeleton into a X_{t, j, k} matrix where t is time, j is joint (x, y, z) and k e {1, 2, 3}
            # for motion, velocity and acceleration. The 2 skeletons are stacked on on the j dimension

            X_skeleton = X_skeleton.transpose(0, 2, 3, 1, 4) # shape (batch_size, seq_len, n_joints, 3, 2)
            X = np.zeros((batch_size, seq_len, 2 * kinematic_chain.shape[0] * 3, 3))

            # Populate X
            i = 0
            for joint in kinematic_chain:
                # Subject 1
                X[:, :, i*3:i*3+3, 0] = X_skeleton[:, :, joint, :, 0]

                # Subject 2
                X[:, :, 3*len(kinematic_chain)+i*3:3*len(kinematic_chain)+i*3+3, 0] = X_skeleton[:, :, joint, :, 1]
                i += 1

            # Calculate velocities and and accelerations
            dt = 1 / 30
            X[:, 1:, :, 1] = (X[:, :-1, :, 0] - X[:, 1:, :, 0]) / dt # Velocity
            X[:, 1:, :, 2] = (X[:, :-1, :, 0] - X[:, 1:, :, 0]) / (dt ** 2)  # Acceleration

            X = torch.from_numpy(np.float32(X.transpose(0, 3, 1, 2))) # shape (batch_size, 3, seq_len, 330)

            s = self.fsk(X.to(device))  # shape (batch_size, 1024, 1, 1)

        if self.include_rgb:
            # ===== Spatial attention =====
            # 1. Glimpse representation (f_g)
            # /!\ Computes for all timesteps at once
            v_tji = self.fg(X_hands) # shape (batch_size, seq_len, output_size = 2048, 4)

            h_t = torch.zeros(batch_size, 1024).to(device)
            c_t= torch.zeros(batch_size, 1024).to(device)

            U = []
            P = []

            for t in range(seq_len):
                # Concatenate # shape (batch_size, 1024, 1, 1) and h_t
                fp_input = torch.cat([s[:, :, 0, 0], h_t], dim = 1)  # shape (batch_size, 2048)

                # Compute p_t vector (see paper)
                p_t = self.fp(fp_input) # shape (batch_size, 4)
                P.append(p_t)

                v_tilde_t = (v_tji[:, t, :, :].permute(1, 0, 2) * p_t).permute(1, 0, 2)  # shape (batch_size, 2048, 4)
                v_tilde_t = torch.sum(v_tilde_t, dim = 2)  # shape (batch_size, 2048)

                h_t, c_t = self.lstm(v_tilde_t, (h_t, c_t))

                u_t = self.fu(h_t) # shape (batch_size, 1024)
                U.append(u_t)

            P = torch.cat(P, dim = 1)  # shape (batch_size, seq_len * 4)

            fp_prime_input = torch.cat([P, s[:, :, 0, 0]], dim = 1)  # shape (batch_size,seq_len * 4 + 1024)

            p_prime = self.fp_prime(fp_prime_input)  # shape (batch_size, seq_len)

            # Compute u_tilde
            U = torch.stack(U, dim = 2)  # shape (batch_size, 1024, seq_len)

            u_tilde = (U.permute(1, 0, 2) * p_prime).permute(1, 0, 2)  # shape (batch_size, 1024, seq_len) -> need to double check
            u_tilde = torch.sum(u_tilde, dim = 2) # shape (batch_size, 1024

        # Prediction
        if self.include_pose and not self.include_rgb:
            out = self.fy_pose(s[:, :, 0, 0])
            out = F.log_softmax(out, dim=1)  # shape (batch_size, n_classes)

            return out

        elif self.include_pose and self.include_rgb:
            out = self.fy_fuse(torch.cat([u_tilde, s[:, :, 0, 0]], dim = 1))  # shape (batch_size, n_classes)
            out = F.log_softmax(out, dim=1)

            return out

        elif self.include_rgb and not self.include_pose:
            out = self.fy_rgb(u_tilde)
            out = F.log_softmax(out, dim=1)

            return out









