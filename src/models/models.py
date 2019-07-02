import torch
from torch import nn, optim
import torch.nn.functional as F
import math

use_cuda = False
use_cuda = torch.cuda.is_available()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class STAHandsCNN(nn.Module):
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def forward(self):
        None