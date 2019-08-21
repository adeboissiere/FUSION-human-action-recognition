import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

from src.models.device import *


class VALSTM(nn.Module):
    def __init__(self, seq_length, n_layers = 3):
        super(VALSTM, self).__init__()
        dropout = 0.5
        n_features = 2 * 3 * 25

        self.hidden_size = 100
        self.n_layers = n_layers

        self.lstm_rotation = nn.LSTMCell(input_size=n_features, hidden_size = self.hidden_size)

        self.lstm_translation = nn.LSTMCell(input_size=n_features, hidden_size = self.hidden_size)

        self.lstm_layers = nn.LSTM(input_size=n_features,
                                   hidden_size=self.hidden_size,
                                   num_layers=n_layers,
                                   dropout=dropout)

        self.fc_rotation = nn.Linear(self.hidden_size, 3)
        self.fc_translation = nn.Linear(self.hidden_size, 3)
        self.fc = nn.Linear(self.hidden_size * seq_length, 60)

    def forward(self, X_skeleton, X_hands):
        # X_skeleton shape (batch_size, 3, sub_sequence_length, num_joints, 2)
        batch_size = X_skeleton.shape[0]
        n_features = 2 * 3 * 25
        seq_len = X_skeleton.shape[2]
        n_joints = X_skeleton.shape[3]

        X_skeleton = X_skeleton.transpose(0, 2, 3, 4, 1)  # shape (batch_size, seq_len, n_joints, 2, 3)
        X_skeleton = X_skeleton.reshape(batch_size, seq_len, n_joints * 2, 3)  # shape (batch_size, seq_len, 25 * 3 * 2)

        X_3D = X_skeleton.reshape(batch_size, seq_len, int(n_features / 3), 3)  # shape (batch, seq_length, n_joints, 3)
        X_skeleton = X_skeleton.reshape(batch_size, seq_len, 2 * 3 * n_joints)

        h_t_rotation = torch.zeros(batch_size, self.hidden_size).to(device)
        c_t_rotation = torch.zeros(batch_size, self.hidden_size).to(device)

        h_t_translation = torch.zeros(batch_size, self.hidden_size).to(device)
        c_t_translation = torch.zeros(batch_size, self.hidden_size).to(device)

        X_trans_rot = torch.zeros(batch_size, seq_len, int(n_features / 3), 3).to(device)

        X_skeleton = torch.from_numpy(np.float32(X_skeleton)).to(device)
        X_3D = torch.from_numpy(np.float32(X_3D)).to(device)

        # Compute translation/rotation parameters of subnetwork
        for t in range(seq_len):
            h_t_rotation, c_t_rotation = self.lstm_rotation(X_skeleton[:, t, :], (h_t_rotation, c_t_rotation))
            h_t_translation, c_t_translation = self.lstm_translation(X_skeleton[:, t, :], (h_t_translation, c_t_translation))

            # Compute translation/rotation parameters
            translation_params = self.fc_translation(h_t_translation)  # shape(batch, 3)
            rotation_params = self.fc_rotation(h_t_rotation)  # shape (batch, 3)

            sin_rot_x = torch.sin(rotation_params[:, 0])
            cos_rot_x = torch.cos(rotation_params[:, 0])
            sin_rot_y = torch.sin(rotation_params[:, 1])
            cos_rot_y = torch.cos(rotation_params[:, 1])
            sin_rot_z = torch.sin(rotation_params[:, 2])
            cos_rot_z = torch.cos(rotation_params[:, 2])

            # Create rotation matrices
            rot_x = rot_y = rot_z = torch.zeros(batch_size, 3, 3).to(device)
            rot_x[:, 0, 0] = rot_y[:, 1, 1] = rot_z[:, 2, 2] = 1

            # rot_x
            rot_x[:, 1, 1] = rot_x[:, 2, 2] = cos_rot_x
            rot_x[:, 1, 2] = - sin_rot_x
            rot_x[:, 2, 1] = sin_rot_x

            # rot_y
            rot_y[:, 0, 0] = rot_y[:, 2, 2] = cos_rot_y
            rot_y[:, 2, 0] = -sin_rot_y
            rot_y[:, 0, 2] = sin_rot_y

            # rot_z
            rot_z[:, 0, 0] = rot_z[:, 1, 1] = cos_rot_z
            rot_z[:, 0, 1] = -sin_rot_z
            rot_z[:, 1, 0] = sin_rot_z

            # Rotation matrix
            rot = torch.bmm(torch.bmm(rot_x, rot_y), rot_z)

            # Apply transformation
            X_trans_rot[:, t, :, :] = X_3D[:, t, :, :] - translation_params.unsqueeze(1).repeat(1, int(n_features / 3),
                                                                                                1)
            X_trans_rot[:, t, :, :] = torch.bmm(rot, X_trans_rot[:, t, :, :].clone().permute(0, 2, 1)).permute(0, 2, 1)

        # Reshape for 2nd subnetwork
        X_trans_rot = X_trans_rot.reshape(batch_size, seq_len, n_features)

        out, _ = self.lstm_layers(X_trans_rot, None)  # out shape (batch_size, seq_length, hidden_size)
        out = out.view(X_skeleton.shape[0], -1)  # out shape (batch_size, seq_length * hidden_size)
        out = self.fc(out)  # out shape (batch_size, n_classes)
        out = F.log_softmax(out, dim=1)

        return out