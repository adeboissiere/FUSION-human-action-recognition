import torch
from torch import nn
import torchvision.models as models
import torch.nn.functional as F

from src.models.device import *

import numpy as np


class ASCNN(nn.Module):
    def __init__(self):
        super(ASCNN, self).__init__()

        # Pretrained model
        self.trained_cnn = models.resnet50(pretrained=True)

