import torch
import torch.nn as nn
import torch.nn.functional as F


# class SimpleCatFeatureExtractor(nn.Module):
#
#     def __init__(self, num_embeddings, embedding_dims, hidden_dim, output_dim, dropout_prob):
#         super(SimpleCatFeatureExtractor, self).__init__()
#         self.embedding_modules = nn.ModuleList()
#         self.num_embeddings = num_embeddings
#         self.embedding_dims = embedding_dims
#
#         for n_emb, emb_dim in zip(self.num_embeddings, self.embedding_dims):
#             # note is n_emb + 1, because the nan values are 0
#             embedding_module = nn.Embedding(n_emb + 1, emb_dim)
#             self.embedding_modules.append(embedding_module)
#
#         self.equal_emb_dim = all(element == embedding_dims[0] for element in embedding_dims)
#         n_cat_feat = len(self.num_embeddings)
#         input_dim = n_cat_feat if self.equal_emb_dim else sum(embedding_dims)
#
#         # if input_dim=32, equal emb_dim.
#         # 32 * 4 => 32 * 16
#         self.conv1 = nn.Conv1d(in_channels=input_dim,
#                                out_channels=16,
#                                kernel_size=3,
#                                stride=1,
#                                #  padding=(1*(32-1) - 32 + 3) / 2 = 1
#                                padding=1)
#         # 32 * 16 => 16 * 16
#         self.pool1 = nn.MaxPool1d(kernel_size=2,
#                                   stride=2,
#                                   # padding=(2*(16-1) - 32 + 2) / 2 = 0
#                                   padding=0)
#         self.fc1 = nn.Linear(16 * 16, hidden_dim)
#         self.dropout = nn.Dropout(p=dropout_prob)
#         self.fc2 = nn.Linear(hidden_dim, output_dim)
#
#     def forward(self, x):
#         x_embed_list = []
#         for i in range(x.shape[1]):
#             x_i = x[:, i].unsqueeze(1)
#             x_embed = self.embedding_modules[i](x_i)
#             x_embed = x_embed.squeeze(1)
#             x_embed_list.append(x_embed)
#         """
#         an idea here: the embedding dim make to same for each cat feature,
#         and we then vertical stack and use conv1d !
#         """
#         if self.equal_emb_dim:
#             embed = torch.stack(x_embed_list, dim=1)  # batch_size * n_cat_feat * emb_dim
#         else:
#             embed = torch.cat(x_embed_list, dim=1)  # batch_size * 1 * sum(embedding_dims)
#             embed = embed.unsqueeze(1)
#         out = self.conv1(embed)
#         out = F.relu(out)
#         out = self.pool1(out)
#         out = out.view(out.size(0), -1)
#         out = self.fc1(out)
#         out = F.relu(out)
#         out = self.dropout(out)
#         out = self.fc2(out)
#         return out


class ConvResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout=0.5):
        super(ConvResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout)

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        out = self.bn2(self.conv2(out))
        out = self.dropout2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CatFeatureExtractor(nn.Module):
    def __init__(self, num_embeddings, embedding_dims, hidden_dim, dropout):
        super(CatFeatureExtractor, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dims = embedding_dims
        self.hidden_dim = hidden_dim

        self.embedding_layers = nn.ModuleList()
        self.fcs = nn.ModuleList()
        for n_emb, emb_dim in zip(self.num_embeddings, self.embedding_dims):
            # note is n_emb + 1, because the nan values are 0
            self.embedding_layers.append(nn.Embedding(n_emb + 1, emb_dim))
            self.fcs.append(nn.Linear(emb_dim, hidden_dim))

        self.res_block1 = ConvResidualBlock(len(self.num_embeddings), 32, dropout=dropout)
        # self.res_block2 = ConvResidualBlock(32, 64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x_embed_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i].unsqueeze(1)
            x_embed = self.embedding_layers[i](x_i)
            x_embed = self.fcs[i](x_embed)
            x_embed = x_embed.squeeze(1)
            x_embed_list.append(x_embed)

        embed = torch.stack(x_embed_list, dim=1)

        out = self.res_block1(embed)
        # out = self.res_block2(out)
        out = self.pool1(out)
        out = out.view(out.size(0), -1)
        return out

    def get_output_dim(self):
        return 32 * self.hidden_dim // 2
