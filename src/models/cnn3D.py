from torch import nn
from src.models.torchvision_models import *
import torch.nn.functional as F


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

        # Features
        out = self.trained_cnn3D(X) # shape (batch_size, 60)
        out = F.log_softmax(out, dim=1)

        return out
