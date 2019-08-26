import torch
from torch import nn

from src.models.VA_CNN import *


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class TrainedPoseNetwork(nn.Module):
    def __init__(self):
        super(TrainedPoseNetwork, self).__init__()

        self.pose_network = VACNN()
        # self.pose_network.load_state_dict(torch.load("./models/VA_CNN.pt"))
        self.pose_network = nn.Sequential(*list(self.pose_network.trained_cnn.children()))[:-1]

        # When feature_extracting = False, sets model to finetuning. Else to feature extraction
        set_parameter_requires_grad(self.pose_network, feature_extracting=False)

        input_size = 224

        # Pretrained models expects normalized inputs
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            normalize
        ])

    def forward(self, X_skeleton):
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

        pose_features = self.pose_network(X)  # shape (batch_size, n_features, 1, 1)

        return pose_features[:, :, 0, 0] # shape (batch_size, n_features)


class TrainedRGBNetwork(nn.Module):
    def __init__(self):
        super(TrainedRGBNetwork, self).__init__()

        # RGB network - set to feature extract (no backprop)
        self.rgb_network = models.resnet18(pretrained=True)
        self.rgb_network = nn.Sequential(*list(self.rgb_network.children()))[:-1]
        set_parameter_requires_grad(self.rgb_network, feature_extracting=False)

        input_size = 224

        # Pretrained models expects normalized inputs
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            normalize
        ])

    def forward(self, X_hands):
        # X_hands shape (batch_size, seq_len, 4, crop_size, crop_size, 3)
        batch_size = X_hands.shape[0]
        seq_len = X_hands.shape[1]
        n_hands = X_hands.shape[2]

        v_tji = [] # See paper for notation
        for hand in range(n_hands):
            v_tj = []
            for t in range(seq_len):
                batch = []

                for b in range(batch_size):
                    # Transform to PIL format
                    hand_crop = Image.fromarray(X_hands[b, t, hand])

                    # Apply transformation
                    hand_crop = self.transform(hand_crop)
                    batch.append(hand_crop)

                # Create batch for given hand and fixed t
                X = torch.stack(batch).to(device)  # shape (batch_size, 3, H, W)

                # Compute feature vector
                out = self.rgb_network(X)[:, :, 0, 0]  # shape (batch_size, output_size)

                # Append to list:
                v_tj.append(out)

            # Stack list to new dimension
            X = torch.stack(v_tj)  # shape (seq_len, batch_size, output_size)
            v_tji.append(X)

        X = torch.stack(v_tji)  # shape (4, seq_len, batch_size, output_size)

        return X.permute(2, 1, 3, 0) # shape (batch_size, seq_len, output_size, 4)


class TemporalAttention(nn.Module):
    def __init__(self, seq_len):
        super(TemporalAttention, self).__init__()

        self.temp_model = nn.Sequential(
                                nn.Linear(512 + seq_len*512, 1024),
                                nn.ReLU(),
                                nn.Linear(1024, 512),
                                nn.ReLU(),
                                nn.Linear(512, seq_len),
                                nn.Sigmoid()
                            )

    def forward(self, pose_features, post_lstm_fc_output):
        # pose_features shape (batch_size, n_features)
        # post_lstm_fc_output shape (batch_size, seq_len, n_features)
        batch_size = pose_features.shape[0]
        seq_len = post_lstm_fc_output.shape[1]

        pose_features_n_features = pose_features.shape[1]
        post_lstm_fc_output_n_features = post_lstm_fc_output.shape[2]

        X = torch.cat(
            [pose_features,
             post_lstm_fc_output.reshape(batch_size, seq_len * post_lstm_fc_output_n_features)],
            dim=1)

        out = self.temp_model(X) # shape (batch_size, seq_len)

        return out


class PoseRGB(nn.Module):
    def __init__(self, seq_len, include_pose):
        super(PoseRGB, self).__init__()
        self.include_pose = include_pose

        self.trained_pose_network = TrainedPoseNetwork()
        self.trained_rgb_network = TrainedRGBNetwork()

        rgb_cnn_output_n_features = 512
        self.fc_rgb_cnn = nn.Linear(4 * rgb_cnn_output_n_features, 1024)

        self.lstm = nn.LSTM(input_size= int((512 * 4) / 2),
                            hidden_size=1024,
                            num_layers=2,
                            batch_first=True, # input shape must be (batch, seq, feature)
                            dropout=0.5,
                            bidirectional=False)

        self.post_lstm_fc = nn.Linear(1024, 512)

        self.temp_attention = TemporalAttention(seq_len)

        self.classification = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 60)
        )

    def forward(self, X_skeleton, X_hands):
        # X_skeleton shape (batch_size, 3, 224, 224)
        # X_hands shape (batch_size, seq_len, n_hands, crop_size, crop_size, 3)
        batch_size = X_hands.shape[0]
        seq_len = X_hands.shape[1]
        n_hands = X_hands.shape[2]

        if self.include_pose == True:
            pose_features = self.trained_pose_network(X_skeleton) # shape (batch_size, n_features)
        else:
            pose_features = torch.zeros((batch_size, 512)).to(device)
        rgb_features = self.trained_rgb_network(X_hands) # shape (batch_size, seq_len, n_features, n_hands)

        # Concat rgb_features shape (batch_size, seq_len, n_features * n_hands)
        rgb_features = rgb_features.reshape(batch_size, seq_len, rgb_features.shape[2] * n_hands)
        rgb_features = self.fc_rgb_cnn(rgb_features) # shape (batch_size, seq_len, n_features)
        rgb_features = F.relu(rgb_features)

        lstm_output, _ = self.lstm(rgb_features) # shape (batch_size, seq_len, n_features)
        lstm_output = F.relu(lstm_output)

        post_lstm_fc_output = []
        for t in range(seq_len):
            post_lstm_fc_output.append(self.post_lstm_fc(lstm_output[:, t, :]))

        post_lstm_fc_output = torch.stack(post_lstm_fc_output, dim = 1) # shape (batch_size, seq_len, n_features)

        temp_attention_weights = self.temp_attention(pose_features, post_lstm_fc_output) # shape (batch_size, seq_len)

        # shape (batch_size, seq_len, n_features)
        temporal_rgb_features = (post_lstm_fc_output.permute(2, 0, 1) * temp_attention_weights).permute(1, 2, 0)

        temporal_rgb_features = torch.sum(temporal_rgb_features, dim=1) # shape (batch_size, n_features)

        # Classify
        out = self.classification(torch.cat([pose_features, temporal_rgb_features], dim=1)) # shape (batch_size, 60)
        out = F.log_softmax(out, dim=1)

        return out
