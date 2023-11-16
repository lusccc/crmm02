import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu2(out)
        out = self.dropout2(out)
        return out


class NumFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super(NumFeatureExtractor, self).__init__()
        self.hidden_dims = hidden_dims
        self.input_layer = nn.Linear(input_dim, self.hidden_dims[0])
        self.residual_blocks = nn.ModuleList()
        for i in range(len(self.hidden_dims) - 1):
            self.residual_blocks.append(ResidualBlock(self.hidden_dims[i], self.hidden_dims[i + 1], dropout))

    def forward(self, x):
        out = self.input_layer(x)
        out = F.relu(out)
        for block in self.residual_blocks:
            out = block(out)
        return out

    def get_output_dim(self):
        return self.hidden_dims[-1]
