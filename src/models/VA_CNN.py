import torch
from torch import nn
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F

from src.models.device import *

import numpy as np
from PIL import Image


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

    def forward(self, X_skeleton, X_hands):
        # X_skeleton shape (batch_size, 3, 224, 224)
        batch_size = X_skeleton.shape[0]

        batch = []

        for b in range(batch_size):
            # Transform to PIL format
            skeleton_image_PIL = Image.fromarray(X_skeleton[b].transpose(1, 2, 0).astype(np.uint8))

            # Apply transformation
            skeleton_image = self.transform(skeleton_image_PIL)
            batch.append(skeleton_image)

        X = torch.stack(batch).to(device)  # shape (batch_size, 3, H, W)

        out = self.trained_cnn(X) # shape (batch_size, 60)
        out = F.log_softmax(out, dim=1)

        '''
        res50_conv = nn.Sequential(*list(self.trained_cnn.children())[:-1])
        print(res50_conv)
        test = res50_conv(X)
        print(test.shape)
        '''

        return out



