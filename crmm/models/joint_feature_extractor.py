import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attn_weights = F.softmax(q @ k.transpose(-2, -1) / math.sqrt(k.size(-1)), dim=-1)
        output = attn_weights @ v
        return output, attn_weights

class JointFeatureExtractor(nn.Module):
    def __init__(self, modality_feat_dims, hidden_dims, dropout, modality_fusion_method='conv'):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.fcs = nn.ModuleDict()
        self.ws = nn.ParameterDict()  # weights for modality features; i.e. weighted fusion
        for modality, feat_dim in modality_feat_dims.items():
            fc = []
            input_dim = feat_dim
            for i, hidden_dim in enumerate(hidden_dims):
                fc.append(nn.Linear(input_dim, hidden_dim))
                fc.append(nn.ReLU(inplace=True))
                fc.append(nn.Dropout(p=dropout))
                input_dim = hidden_dim
            self.fcs[modality] = nn.Sequential(*fc)
            self.ws[modality] = nn.Parameter(torch.ones(1))

        self.modality_fusion_method = modality_fusion_method
        self.num_fuse_modalities = len(modality_feat_dims)
        if self.modality_fusion_method == 'conv':
            self.conv1d = nn.Conv1d(in_channels=self.num_fuse_modalities, out_channels=1, kernel_size=3, padding=1)
            self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        elif self.modality_fusion_method == 'self_attention':
            self.attention_dim = 64  # Or other value
            self.self_attention = SelfAttention(self.hidden_dims[-1], self.attention_dim)
        self.flatten = nn.Flatten()

    def forward(self, x):
        if len(x) != self.num_fuse_modalities:
            # if input miss a modality, create data with 0s !
            missing_modality = [m for m in self.ws if m not in x][0]
            x[missing_modality] = torch.zeros_like(list(x.values())[0])
        if self.modality_fusion_method == 'conv':
            x_aligned = []  # aligned features to hidden_dim
            for modality, x in x.items():
                # Apply linear transformation
                x = self.fcs[modality](x)
                # Apply sigmoid to weights
                weights = torch.sigmoid(self.ws[modality])
                x = x * weights  # weighted
                # x = x * self.ws[modality]  # weighted
                x_aligned.append(x.unsqueeze(1))  # Add channel dimension (B, C, L) -> (B, 1, L)

            # Stack modality features along the channel dimension
            x_stacked = torch.cat(x_aligned, dim=1)  # (B, num_modalities, L)

            # Apply convolution for feature fusion
            x_res = x_stacked
            x_fused = self.conv1d(x_stacked)

            # Add residual connection
            # x_fused = x_fused + x_res[:, :1, :]  # Make sure dimensions match before adding
            # x_fused = x_fused + sum([x_res[:, i, :] for i in range(1, self.num_modalities)]).unsqueeze(1)
            x_fused = x_fused + torch.mean(x_res, dim=1, keepdim=True)
            # Apply Max Pooling
            x_fused = self.pool(x_fused)

            # Apply ReLU activation
            x_fused = F.relu(x_fused)
            x_fused = x_fused.squeeze(1)  # Remove channel dimension (B, 1, L) -> (B, L)
            output = self.flatten(x_fused)
        elif self.modality_fusion_method == 'concat':
            x_concat = []
            for modality, x in x.items():
                # Apply linear transformation
                x = self.fcs[modality](x)
                # Apply sigmoid to weights
                weights = torch.sigmoid(self.ws[modality])
                x = x * weights
                x_concat.append(x)
            output = torch.cat(x_concat, dim=1)
        elif self.modality_fusion_method == 'self_attention':
            x_att = []
            for modality, x in x.items():
                # Apply linear transformation
                x = self.fcs[modality](x)
                # Apply sigmoid to weights
                weights = torch.sigmoid(self.ws[modality])
                x = x * weights
                x_att.append(x)

            x_att, _ = self.self_attention(torch.stack(x_att, dim=0))
            output = x_att.sum(dim=0)
        else:
            raise ValueError('Unknown modality fusion method: {}'.format(self.modality_fusion_method))
        return output

    def get_output_dim(self):
        if self.modality_fusion_method == 'concat':
            return self.hidden_dims[-1] * self.num_fuse_modalities
        elif self.modality_fusion_method == 'conv':
            return 1 * self.hidden_dims[-1] // 2
        elif self.modality_fusion_method == 'self_attention':
            return self.hidden_dims[-1]


if __name__ == '__main__':
    model = JointFeatureExtractor(modality_feat_dims={'audio': 128, 'text': 768}, hidden_dim=128)
    print(model)
