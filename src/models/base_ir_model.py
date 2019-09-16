import torch
from torch import nn
import torchvision.models as models
import torch.nn.functional as F

from src.models.device import *
from src.models.AS_CNN_utils import *

import numpy as np


class BaseIRCNN(nn.Module):
    def __init__(self):
        super(BaseIRCNN, self).__init__()

        # Pretrained model
        self.trained_cnn = models.resnet50(pretrained=True)
        self.trained_cnn = nn.Sequential(*list(self.trained_cnn.children()))[:-1]

        # When feature_extracting = False, sets model to finetuning. Else to feature extraction
        set_parameter_requires_grad(self.trained_cnn, feature_extracting=False)

        self.lstm = nn.LSTM(input_size=2048,
                            hidden_size=2048,
                            num_layers=2,
                            batch_first=True,  # input shape must be (batch, seq, feature)
                            dropout=0.5,
                            bidirectional=False)

        self.fc = nn.Linear(2048, 60)

    def forward(self, X):
        """

        :param X: list containing a single tensor of sampled ir videos of shape (batch_size, seq_len, 3, 224, 224)

        :return:
        """
        X = torch.from_numpy(np.float32(X[0] / 255)).to(device)
        batch_size, seq_len, _, _, _ = X.shape

        # Normalize X
        normalize_values = torch.tensor([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]).to(device)  # [[mean], [std]]
        X = ((X.permute(0, 1, 3, 4, 2) - normalize_values[0]) / normalize_values[1]).permute(0, 1, 4, 2, 3)

        rnn_inputs = []

        # Compute rnn inputs
        for t in range(seq_len):
            out = self.trained_cnn(X[:, t])[:, :, 0, 0] # shape (batch_size, n_features)
            rnn_inputs.append(out)

        rnn_inputs = torch.stack(rnn_inputs, dim=1) # shape (batch_size, seq_len, n_features)
        rnn_outputs, _ = self.lstm(rnn_inputs) # shape (batch_size, seq_len, n_features)

        predictions = self.fc(rnn_outputs) # shape (batch_size, seq_len, 60)
        predictions = F.softmax(predictions, dim=2) # shape (batch_size, seq_len, 60)
        prediction = torch.mean(predictions, dim=1) # shape (batch_size, 60)

        return prediction




