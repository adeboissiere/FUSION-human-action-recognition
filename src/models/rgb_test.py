import torch
from torch import nn, optim
import torch.nn.functional as F

import numpy as np

from src.models.device import *

class RGB_CNN(nn.Module):
    def __init__(self):
        super(RGB_CNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1), # shape (batch_size, 64, 50, 50)
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=1, padding=1), # shape (batch_size, 128, 50, 50)
            nn.ReLU(),
            nn.MaxPool2d(2, padding=1),  # shape (batch_size, 128, 26, 26)
            nn.Conv2d(128, 256, 3, stride=1, padding=1), # shape (batch_size, 256, 26, 26)
            nn.ReLU(),
            nn.MaxPool2d(2), # shape (batch_size, 256, 13, 13),
            nn.Conv2d(256, 512, 3, stride=1, padding=1), # shape (batch_size, 128, 13, 13),
            nn.ReLU(),
            nn.MaxPool2d(13)
        )

    def forward(self, X_skeleton, X_hands):
        # X_hands: shape (batch_size, seq_len, 4, crop_size = 25, crop_size = 25, 3)

        features = []
        for hand in range(4):
            time_seq = []
            for t in range(X_hands.shape[1]):
                hand_tensor = torch.from_numpy(np.float32(X_hands[:, :, hand][:, t].transpose(0, 3, 1, 2))).to(device)

                out = self.cnn(hand_tensor) # shape (batch_size, 512, 1, 1)

                time_seq.append(out[:, :, 0, 0])

            time_seq = torch.stack(time_seq)

            features.append(time_seq)

        features = torch.stack(features) # shape batch_size (4, seq_len, batch_size, 512)

        return features.permute(2, 0, 1, 3) # shape batch_size (batch_size, 4, seq_len, n_features)


class RGB_Classifier(nn.Module):
    def __init__(self):
        super(RGB_Classifier, self).__init__()

        self.rgb_cnn = RGB_CNN()
        self.rnn = nn.LSTM(input_size=512,
                           hidden_size=1024,
                           num_layers=2,
                           batch_first=True,
                           dropout=0.3)

        self.fc = nn.Linear(1024, 60)

    def forward(self, X_skeleton, X_hands):
        features = self.rgb_cnn(X_skeleton, X_hands) # shape batch_size (batch_size, 4, seq_len, n_features)

        features = features.mean(dim = 1) # shape (batch_size, seq_len, n_features)

        output, _ = self.rnn(features) # shape (batch_size, seq_len, n_features)

        out_fc = []
        for t in range(output.shape[1]):
            out = self.fc(output[:, t, :])
            out_fc.append(out)

        out_fc = torch.stack(out_fc) # shape (batch_size, seq_len, n_classes)
        out_fc = out_fc.mean(dim = 1)

        out = F.log_softmax(out, dim=1)

        return out






