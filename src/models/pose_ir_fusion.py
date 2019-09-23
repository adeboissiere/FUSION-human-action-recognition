import torch
from torch import nn
from src.models.torchvision_models import *
import torchvision.models as models
import torch.nn.functional as F

from src.models.cnn3D import *

class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()

        # Pretrained pose network
        self.pose_net = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-1])
        self.pose_fc = nn.Linear(512, 60)

        # Pretrained IR network
        self.ir_net = nn.Sequential(*list(r2plus1d_18(pretrained=True, progress=True).children())[:-1])
        self.ir_fc = nn.Linear(512, 60)

        # Auto-learn weights
        self.avg_net = nn.Sequential(
            nn.Linear(2 * 512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, X):
        X_skeleton = X[0] # shape (batch_size, 3, 224, 224)
        X_ir = X[1] # shape(batch_size, 3, seq_len, 112, 112)

        out_pose = self.pose_net(X_skeleton)[:, :, 0, 0] # shape (batch_size, 512)
        out_ir = self.ir_net(X_ir)[:, :, 0, 0, 0] # shape (batch_size, 512)

        weighted_prediction = self.avg_net(torch.cat([out_pose, out_ir], dim=1)) # shape (batch_size, 2)

        pred_pose = F.softmax(self.pose_fc(out_pose), dim=1) # shape (batch_size, 60)
        pred_ir = F.softmax(self.ir_fc(out_ir), dim=1) # shape (batch_size, 60)

        pred = (weighted_prediction[:, 0] * pred_pose.transpose(1, 0)
                + weighted_prediction[:, 1] * pred_ir.transpose(1, 0)).transpose(1, 0)

        return torch.log(pred)

