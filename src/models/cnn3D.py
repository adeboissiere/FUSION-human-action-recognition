from torch import nn
from src.models.torchvision_models import *
import torch.nn.functional as F

from src.models.device import *
from src.models.AS_CNN_utils import *

import numpy as np


class CNN3D(nn.Module):
    def __init__(self):
        super(CNN3D, self).__init__()

        # Pretrained model
        self.trained_cnn3D = r2plus1d_18(pretrained=True, progress=True)
        self.trained_cnn3D.fc = nn.Linear(self.trained_cnn3D.fc.in_features, 60)

    def forward(self, X):
        """

        :param X: list containing a single tensor of sampled ir videos of shape (batch_size, seq_len, 3, 224, 224)

        :return:
        """

        X = torch.from_numpy(np.float32(X[0] / 255)).to(device)
        batch_size, seq_len, _, _, _ = X.shape

        # Normalize X
        normalize_values = torch.tensor([[0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989]]).to(device)  # [[mean], [std]]
        X = ((X.permute(0, 1, 3, 4, 2) - normalize_values[0]) / normalize_values[1]).permute(0, 1, 4, 2, 3)

        # Features
        out = self.trained_cnn3D(X.permute(0, 2, 1, 3, 4)) # shape (batch_size, 60)
        out = F.log_softmax(out, dim=1)

        return out
