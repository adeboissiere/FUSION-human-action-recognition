import torch
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
        None