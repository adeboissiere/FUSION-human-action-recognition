import torch
from torch import nn
import torchvision.models as models
import torch.nn.functional as F

from src.models.device import *
from src.models.AS_CNN_utils import *

import numpy as np


class ASModule(nn.Module):
    def __init__(self):
        super(ASModule, self).__init__()

        self.as_mlp = nn.Sequential(
            nn.Linear(48, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.as_subnetwork_cnn = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(56)
        )

        self.as_subnetwork_linear = nn.Linear(256, 48)
        torch.nn.init.zeros_(self.as_subnetwork_linear.weight)
        torch.nn.init.zeros_(self.as_subnetwork_linear.bias)

    def forward(self, X_skeleton, X_bone_length):
        out_mlp = self.as_mlp(X_bone_length) # shape (batch_size, 128)
        out_cnn = self.as_subnetwork_cnn(X_skeleton)[:, :, 0, 0] # shape (batch_size, 128)

        out_concat = torch.cat([out_mlp, out_cnn], dim=1) # shape (batch_size, 256)

        out = self.as_subnetwork_linear(out_concat) # shape (batch_size, n_neighbors * n_subjects = 2 * 24)

        return 1. + out


class ASCNN(nn.Module):
    def __init__(self):
        super(ASCNN, self).__init__()

        # Subnetwork
        self.as_subnetwork = ASModule()
        # When feature_extracting = True, model is frozen (bypass)
        set_parameter_requires_grad(self.as_subnetwork, feature_extracting=False)

        # Pretrained model
        self.trained_cnn = models.resnet50(pretrained=True)

        # When feature_extracting = False, sets model to finetuning. Else to feature extraction
        set_parameter_requires_grad(self.trained_cnn, feature_extracting=False)

        # Reshapes output
        self.trained_cnn.fc = nn.Linear(2048, 60)

    def forward(self, X):
        """

        :param X: [X_skeleton, X_bone_length]
        X_skeleton shape shape (batch_size, 3, 224, 224)
        X_bone shape (batch_size, n_neighbors * n_subjects = 2 * 24)

        :return:
        """

        X_skeleton = torch.from_numpy(np.float32(X[0])).to(device)
        X_bone_length = torch.from_numpy(np.float32(X[1])).to(device)

        # Parameters
        bach_size = X_skeleton.shape[0]

        # Compute rescale parameters shape (batch_size, n_neighbors * n_subjects = 2 * 24)
        scale_factors = self.as_subnetwork(X_skeleton, X_bone_length)

        subject_0 = X_skeleton[:, :, 0:25, :]
        subject_1 = X_skeleton[:, :, 25:50, :]

        scale_factors_subject_0 = scale_factors[:, :24]
        scale_factors_subject_1 = scale_factors[:, 24:]

        # shape (batch_size, 3, n_joints = 25, seq_len = 224)
        scaled_subject_0 = rescale_skeleton(subject_0.clone(), scale_factors_subject_0)
        scaled_subject_1 = rescale_skeleton(subject_1.clone(), scale_factors_subject_1)

        # Prime image for classification network
        normalize_values = torch.tensor([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]).to(device)  # [[mean], [std]]
        X_prime = (torch.ones([bach_size, 224, 224, 3]).to(device) * normalize_values[0]).permute(0, 3, 1, 2)

        X_prime[:, :, 0:25, :] = scaled_subject_0 / 255
        X_prime[:, :, 25:50, :] = scaled_subject_1 / 255

        X_prime = ((X_prime.permute(0, 2, 3, 1) - normalize_values[0]) / normalize_values[1]).permute(0, 3, 1, 2)

        out = self.trained_cnn(X_prime)  # shape (batch_size, 60)
        out = F.log_softmax(out, dim=1)

        return out
