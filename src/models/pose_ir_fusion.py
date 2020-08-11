r"""
Contains a PyTorch model fusing IR and pose data for improved classification. Also contains a helper function which
normalizes pose and IR tensors.

"""
import torch
from torch import nn
from src.models.torchvision_models import *
import torchvision.models as models
import torch.nn.functional as F

# Custom imports
from src.models.utils import *


class FUSION(nn.Module):
    r"""This model is built on three submodules. The first is called a "pose module", which takes a skeleton sequence
    mapped to an image an outputs a 512-long feature vector. The second one is an "IR module", which takes an IR sequence
    and outputs a 512-long feature vector. The third one is a "classification module", which combines the 2 feature
    vectors (concatenation) and predicts a class via an MLP. This model can achieve over 90% accuracy on both
    benchmarks of the NTU RGB+D (60) dataset.

    Attributes:
        - **use_pose** (bool): Include skeleton data
        - **use_ir** (bool): Include IR data
        - **pose_net** (PyTorch model): Pretrained ResNet-18. Only exists if **use_pose** is True.
        - **ir_net** (PyTorch model): Pretrained R(2+1)D-18. Only exists if **use_ir** is True.
        - **class_mlp** (PyTorch model): Classification MLP. Input size is adjusted depending on the modules used.
          Input size is 512 if only one module is used, 1024 for two modules.

    Methods:
        *forward(X)*: Forward step. X contains pose/IR data

    """
    def __init__(self, use_pose, use_ir, pretrained, fusion_scheme):
        super(FUSION, self).__init__()

        # Parts of the network to activate
        self.use_pose = use_pose
        self.use_ir = use_ir
        self.fusion_scheme = fusion_scheme

        # Pretrained pose network
        if use_pose:
            self.pose_net = nn.Sequential(*list(models.resnet18(pretrained=pretrained).children())[:-1])

        # Pretrained IR network
        if use_ir:
            self.ir_net = nn.Sequential(*list(r2plus1d_18(pretrained=pretrained).children())[:-1])

        # Compute number of classification MLP input features
        mlp_input_features = 512
        if use_ir and use_pose:
            if self.fusion_scheme == "CONCAT":
                mlp_input_features = 2 * 512
            elif self.fusion_scheme == "SUM":
                mlp_input_features = 512
            elif self.fusion_scheme == "MULT":
                mlp_input_features = 512
            elif self.fusion_scheme == "AVG":
                mlp_input_features = 512
            elif self.fusion_scheme == "MAX":
                mlp_input_features = 512
            elif self.fusion_scheme == "CONV":
                mlp_input_features = 512
                self.conv_features = nn.Conv2d(1, 1, kernel_size=(1, 2))		
            else:
                print("Fusion scheme not recognized. Exiting ...")
                exit()

        # Classification MLP
        self.class_mlp = nn.Sequential(
            nn.BatchNorm1d(mlp_input_features),
            nn.Linear(mlp_input_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 60)
        )

    def forward(self, X):
        r"""Forward step of the FUSION model. Input X contains a list of 2 tensors containing pose and IR data. The
        input is already normalized as specified in the PyTorch pretrained vision models documentation, using the
        *prime_X_fusion* function. Each tensor is then passed to its corresponding module. The 2 feature vectors are
        concatenated, then fed to the classification module (MLP) which then outputs a prediction.

        Inputs:
            **X** (list of PyTorch tensors): Contains the following tensors:
                - **X_skeleton** (PyTorch tensor): pose images of shape `(batch_size, 3, 224, 224)` if **use_pose** is
                  True. Else, tensor = None.
                - **X_ir** (PyTorch tensor): IR sequences of shape `(batch_size, 3, seq_len, 112, 112)` if **use_ir** is
                  True. Else, tensor = None

        Outputs:
            **pred** (PyTorch tensor): Contains the the log-Softmax normalized predictions of shape
            `(batch_size, n_classes=60)`

        """
        X_skeleton = X[0]  # shape (batch_size, 3, 224, 224) or None
        X_ir = X[1]  # shape(batch_size, 3, seq_len, 112, 112) or None

        # Compute each data stream
        if self.use_pose:
            out_pose = self.pose_net(X_skeleton)[:, :, 0, 0]  # shape (batch_size, 512)
        if self.use_ir:
            out_ir = self.ir_net(X_ir)[:, :, 0, 0, 0]  # shape (batch_size, 512)

        # Create feature vector
        if self.use_pose and not self.use_ir:
            features = out_pose
        elif not self.use_pose and self.use_ir:
            features = out_ir
        elif self.use_pose and self.use_ir:
            if self.fusion_scheme == "CONCAT":
                features = torch.cat([out_pose, out_ir], dim=1)
            elif self.fusion_scheme == "SUM":
                features = out_pose + out_ir
            elif self.fusion_scheme == "MULT":
                features = out_pose * out_ir
            elif self.fusion_scheme == "AVG":
                features = out_pose + out_ir / 2
            elif self.fusion_scheme == "MAX":
                features = torch.max(out_pose, out_ir)
            elif self.fusion_scheme == "CONV":
                features = torch.stack([out_pose, out_ir], dim=2).unsqueeze(1)
                features = self.conv_features(features).squeeze(1).squeeze(2)

        pred = self.class_mlp(features)  # shape (batch_size, 60)
        pred = F.softmax(pred, dim=1)

        return torch.log(pred + 1e-12)


def prime_X_fusion(X, use_pose, use_ir):
    r"""Normalizes X (list of tensors) as defined in the pretrained Torchvision models documentation. **Note** that
    **X_ir** is reshaped in this function.

    Inputs:
        - **X** (list of PyTorch tensors): Contains the following tensors:
            - **X_skeleton** (PyTorch tensor): pose images of shape `(batch_size, 3, 224, 224)` if **use_pose** is
              True. Else, tensor = -1.
            - **X_ir** (PyTorch tensor): IR sequences of shape `(batch_size, seq_len, 3, 112, 112)` if **use_ir** is
              True. Else, tensor = -1
        - **use_pose** (bool): Include skeleton data
        - **use_ir** (bool): Include IR data

    Outputs:
        **X** (list of PyTorch tensors): Contains the following tensors:
            - **X_skeleton** (PyTorch tensor): pose images of shape `(batch_size, 3, 224, 224)` if **use_pose** is
              True. Else, tensor = None.
            - **X_ir** (PyTorch tensor): IR sequences of shape `(batch_size, 3, seq_len, 112, 112)` if **use_ir** is
              True. Else, tensor = None
    """
    if use_pose:
        X_skeleton = X[0] / 255.0  # shape (batch_size, 3, 224, 224)

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