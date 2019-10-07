import torch
from torch import nn
from src.models.torchvision_models import *
import torchvision.models as models
import torch.nn.functional as F

from src.models.cnn3D import *


class FUSION(nn.Module):
    def __init__(self, use_pose, use_ir, pretrained):
        super(FUSION, self).__init__()

        # Parts of the network to activate
        self.use_pose = use_pose
        self.use_ir = use_ir

        # Pretrained pose network
        if use_pose:
            self.pose_net = nn.Sequential(*list(models.resnet18(pretrained=pretrained).children())[:-1])

        # Pretrained IR network
        if use_ir:
            self.ir_net = nn.Sequential(*list(r2plus1d_18(pretrained=pretrained).children())[:-1])

        # Classification MLP
        self.class_mlp = nn.Sequential(
            nn.BatchNorm1d((int(use_pose) + int(use_ir)) * 512),
            nn.Linear((int(use_pose) + int(use_ir)) * 512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 60)
        )

    def forward(self, X):
        X_skeleton = X[0] # shape (batch_size, 3, 224, 224) or None
        X_ir = X[1] # shape(batch_size, 3, seq_len, 112, 112) or None

        # Compute each data stream
        if self.use_pose:
            out_pose = self.pose_net(X_skeleton)[:, :, 0, 0] # shape (batch_size, 512)
        if self.use_ir:
            out_ir = self.ir_net(X_ir)[:, :, 0, 0, 0] # shape (batch_size, 512)

        # Create feature vector
        if self.use_pose and not self.use_ir:
            features = out_pose
        elif not self.use_pose and self.use_ir:
            features = out_ir
        elif self.use_pose and self.use_ir:
            features = torch.cat([out_pose, out_ir], dim=1)

        pred = self.class_mlp(features) # shape (batch_size, 60)
        pred = F.softmax(pred, dim=1)

        return torch.log(pred + 1e-12)


def prime_X_fusion(X, use_pose, use_ir):
    if use_pose:
        X_skeleton = X[0] / 255.0 # shape (batch_size, 3, 224, 224)

        # Normalize X_skeleton
        normalize_values = torch.tensor([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])  # [[mean], [std]]
        X_skeleton = ((X_skeleton.permute(0, 2, 3, 1) - normalize_values[0]) / normalize_values[1]).permute(0, 3, 1, 2)

        if not use_ir:
            return [X_skeleton.to(device), None]

    if use_ir:
        X_ir = X[1] / 255.0 # shape (batch_size, seq_len, 3, 113, 112)

        # Normalize X
        normalize_values = torch.tensor([[0.43216, 0.394666, 0.37645],
                                         [0.22803, 0.22145, 0.216989]])  # [[mean], [std]]
        X_ir = ((X_ir.permute(0, 1, 3, 4, 2) - normalize_values[0]) / normalize_values[1]).permute(0, 1, 4, 2, 3)

        if not use_pose:
            return [None, X_ir.permute(0, 2, 1, 3, 4).to(device)]

    return [X_skeleton.to(device), X_ir.permute(0, 2, 1, 3, 4).to(device)]
