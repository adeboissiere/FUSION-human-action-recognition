import torch
from torch import nn
import torchvision.models as models
import torch.nn.functional as F

from src.models.device import *

import numpy as np


class VASubCNN(nn.Module):
    def __init__(self):
        super(VASubCNN, self).__init__()

        self.va_subnetwork_cnn = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(56)
        )

        self.va_subnetwork_linear = nn.Linear(128, 6)
        torch.nn.init.zeros_(self.va_subnetwork_linear.weight)
        torch.nn.init.zeros_(self.va_subnetwork_linear.bias)

    def forward(self, X_skeleton):
        out = self.va_subnetwork_cnn(X_skeleton) # shape (batch_size, 128, 1, 1)
        out = self.va_subnetwork_linear(out[:, :, 0, 0]) # shape (batch_size, 6)

        return out


class VACNN(nn.Module):
    def __init__(self):
        super(VACNN, self).__init__()

        # Subnetwork
        self.va_subnetwork = VASubCNN()

        # Pretrained model
        self.trained_cnn = models.resnet50(pretrained=True)

        # When feature_extracting = False, sets model to finetuning. Else to feature extraction
        set_parameter_requires_grad(self.trained_cnn, feature_extracting=False)

        # Reshapes output
        self.trained_cnn.fc = nn.Linear(2048, 60)

    def forward(self, X):
        X = X[0]
        c_min = -4.767
        c_max = 5.188

        # X_skeleton shape (batch_size, 3, 224, 224)
        batch_size, _, H, W = X.shape
        X = torch.from_numpy(np.float32(X)).to(device)

        # Computes transformation parameters
        transform_params = self.va_subnetwork(X / 255) # shape (batch_size, 6)

        alpha = transform_params[:, 0]
        beta = transform_params[:, 1]
        gamma = transform_params[:, 2]

        cos_alpha = torch.cos(alpha)
        sin_alpha = torch.sin(alpha)
        cos_beta = torch.cos(beta)
        sin_beta = torch.sin(beta)
        cos_gamma = torch.cos(gamma)
        sin_gamma = torch.sin(gamma)

        rot_x = torch.zeros(batch_size, 3, 3).to(device)
        rot_y = torch.zeros(batch_size, 3, 3).to(device)
        rot_z = torch.zeros(batch_size, 3, 3).to(device)
        rot_x[:, 0, 0] = rot_y[:, 1, 1] = rot_z[:, 2, 2] = 1

        # rot_x
        rot_x[:, 1, 1] = rot_x[:, 2, 2] = cos_alpha
        rot_x[:, 1, 2] = - sin_alpha
        rot_x[:, 2, 1] = sin_alpha

        # rot_y
        rot_y[:, 0, 0] = rot_y[:, 2, 2] = cos_beta
        rot_y[:, 2, 0] = -sin_beta
        rot_y[:, 0, 2] = sin_beta

        # rot_z
        rot_z[:, 0, 0] = rot_z[:, 1, 1] = cos_gamma
        rot_z[:, 0, 1] = -sin_gamma
        rot_z[:, 1, 0] = sin_gamma

        # Rotation matrix
        R = torch.bmm(torch.bmm(rot_x, rot_y), rot_z)

        # Translation vector
        dx = transform_params[:, 3]
        dy = transform_params[:, 4]
        dz = transform_params[:, 5]
        d = torch.stack([dx, dy, dz], dim=1) # shape (batch_size, 3)

        # See equation (12) of
        # View_Adaptive_Neural_Networks_for_High_Performance_Skeleton_based_Human_Action_Recognition
        # The equation has the form X' = A + B where B is a constant given R and d (sequence wise)

        # Computation of B
        B = c_min - d # shape (batch_size, 3)
        B = torch.matmul(R, B[:, :, None])
        B = 255 * (B - c_min) / (c_max - c_min)

        # Computation of A
        A = torch.matmul(R, X.reshape(batch_size, 3, H * W)) # shape (batch_size, 3, H*W)

        # Computation of X'
        X_prime = (A + B).reshape(batch_size, 3, H, W) # shape (batch_size, 3, H, W)

        # Normalize inputs for pretrained CNN (see PyTorch doc for details)
        X_prime /= 255

        normalize_values = torch.tensor([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]).to(device) # [[mean], [std]]
        X_prime = ((X_prime.permute(0, 2, 3, 1) - normalize_values[0]) / normalize_values[1]).permute(0, 3, 1, 2)

        out = self.trained_cnn(X_prime) # shape (batch_size, 60)
        out = F.log_softmax(out, dim=1)

        '''
        res50_conv = nn.Sequential(*list(self.trained_cnn.children())[:-1])
        print(res50_conv)
        test = res50_conv(X)
        print(test.shape)
        '''

        return out



