import torch
from torch import nn
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import numpy as np


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class VACNN(nn.Module):
    def __init__(self):
        super(VACNN, self).__init__()

        # Pretrained model
        self.trained_cnn = models.resnet18(pretrained=True)

        # When feature_extracting = False, sets model to finetuning. Else to feature extraction
        set_parameter_requires_grad(self.trained_cnn, feature_extracting=False)

        # Reshapes output
        self.trained_cnn.fc = nn.Linear(512, 60)

        input_size = 224

        # Pretrained models expect normalized inputs
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            normalize
        ])

    def forward(self, X):
        None

