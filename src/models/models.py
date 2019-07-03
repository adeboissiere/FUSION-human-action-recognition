import torch
from torch import nn, optim
import torchvision.transforms as transforms
import torchvision.models as models

from PIL import Image

import time

from src.utils.joints import *

use_cuda = False
use_cuda = torch.cuda.is_available()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Fsk(nn.Module):
    def __init__(self):
        super(Fsk, self).__init__()
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
        print(out.shape)

        return out


class Fg(nn.Module):
    def __init__(self):
        super(Fg, self).__init__()
        self.inception_v3 = models.inception_v3(pretrained = True)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize
        ])

    def forward(self, X_hands):
        """ Forward propagation of f_g module (inception_v3).
        It takes about 4.5s to apply transformation to all images for batch_size = 32, seq_len = 20

        :param X_hands: shape (batch_size, seq_len, 4, crop_size, crop_size, 3)
        :return:
        """
        # Apply transform to batch
        '''
        image_test = X_hands[0, 0, 0]
        print(image_test[:, :, 0])
        image_test = Image.fromarray(image_test)
        image_test = self.transform(image_test)

        print(image_test.shape)
        print(image_test[0])
        '''
        batch_size = X_hands.shape[0]
        seq_len = X_hands.shape[1]

        print("start")
        start = time.time()
        for b in range(batch_size):
            for t in range(seq_len):
                for hand in range(4):
                    # Transform to PIL format
                    hand_crop = Image.fromarray(X_hands[b, t, hand])

                    # Apply transformation
                    hand_crop = self.transform(hand_crop)

        print("Converting images took : " + str(time.time() - start) + " seconds")


class STAHandsCNN(nn.Module):
    def __init__(self, n_classes):
        super(STAHandsCNN, self).__init__()
        self.n_classes = n_classes
        self.fsk = Fsk()

        self.fg = Fg()



    def forward(self, X_skeleton, X_hands):
        """ Forward propagation of STAHandsCNN

        :param X_skeleton: shape (batch_size, 3, sub_sequence_length, num_joints, 2)
        :param X_hands: shape (batch_size, sub_sequence_length, 4, crop_size, crop_size, 3)
        :return:
        """
        batch_size = X_skeleton.shape[0]
        seq_len = X_skeleton.shape[2]

        # ===== Convolutional pose features (f_sk) =====
        # 1. Transform X_skeleton into a X_{t, j, k} matrix where t is time, j is joint (x, y, z) and k e {1, 2, 3}
        # for motion, velocity and acceleration. The 2 skeletons are stacked on on the j dimension
        '''
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

        out = self.fsk(X.to(device))
        '''

        # ===== Spatial attention =====
        # 1. Glimpse representation (f_g)
        self.fg(X_hands)