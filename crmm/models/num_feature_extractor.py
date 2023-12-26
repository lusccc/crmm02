import torch
import torch.nn as nn

from crmm.models.num_cat_language_model import NumCatLanguageModel


# @@@ try: initial idea
# class ResidualBlock(nn.Module):
#     def __init__(self, input_dim, hidden_dim, dropout):
#         super(ResidualBlock, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.bn1 = nn.BatchNorm1d(hidden_dim)
#         self.relu1 = nn.ReLU()
#         self.dropout1 = nn.Dropout(p=dropout)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.bn2 = nn.BatchNorm1d(hidden_dim)
#         self.tanh = nn.Tanh()
#         self.dropout2 = nn.Dropout(p=dropout)
#
#     def forward(self, x):
#         residual = x
#         out = self.fc1(x)
#         out = self.bn1(out)
#         out = self.relu1(out)
#         out = self.dropout1(out)
#         out = self.fc2(out)
#         out = self.bn2(out)
#         out += residual
#         out = self.tanh(out)
#         out = self.dropout2(out)
#         return out
#
#
# class NumFeatureExtractor(nn.Module):
#     def __init__(self, input_dim, hidden_dims, dropout):
#         super(NumFeatureExtractor, self).__init__()
#         self.hidden_dims = hidden_dims
#         self.input_layer = nn.Linear(input_dim, self.hidden_dims[0])
#         self.residual_blocks = nn.ModuleList()
#         for i in range(len(self.hidden_dims) - 1):
#             self.residual_blocks.append(ResidualBlock(self.hidden_dims[i], self.hidden_dims[i + 1], dropout))
#
#     def forward(self, x):
#         out = self.input_layer(x)
#         out = F.relu(out)
#         for block in self.residual_blocks:
#             out = block(out)
#         return out
#
#     def get_output_dim(self):
#         return self.hidden_dims[-1]


# @@@ try 2: section 3.1 in paper: ClipCap: CLIP Prefix for Image Captioning
# class LightMapping(nn.Module):
#
#     def forward(self, x):
#         return self.model(x)
#
#     def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
#         super(LightMapping, self).__init__()
#         layers = []
#         for i in range(len(sizes) - 1):
#             layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
#             if i < len(sizes) - 2:
#                 layers.append(act())
#         self.model = nn.Sequential(*layers)


# @@@@ try 3, only use LN as the Tab-Transformer in paper
# `TabTransformer: Tabular Data Modeling Using Contextual Embeddings`
# but BN, light mapping are also added to test
# class NumFeatureExtractor(nn.Module):
#     def __init__(self, num_feat_dim, language_model: NumCatLanguageModel):
#         super(NumFeatureExtractor, self).__init__()
#         self.language_model = language_model
#         self.num_feat_dim = num_feat_dim
#         self.map_token_len = 1
#         self.language_model_hidden_dim = self.language_model.get_output_dim()
#         self.num_light_mapping = LightMapping((self.num_feat_dim,
#                                                # self.num_feat_dim * self.map_token_len // 2,
#                                                self.language_model_hidden_dim * self.map_token_len))
#         # self.num_light_mapping = nn.Linear(self.num_feat_dim, self.language_model_hidden_dim)
#
#         # self.bn = nn.BatchNorm1d(self.num_feat_dim)
#         self.ln_before = nn.LayerNorm(self.num_feat_dim)
#         # self.ln = nn.LayerNorm(self.language_model_hidden_dim)
#
#         self.dropout = nn.Dropout(0.1)  # hidden_dropout_prob in NumCatLanguageModelConfig
#
#     def forward(self, x):
#         x = self.ln_before(x)
#         mapped = self.num_light_mapping(x)
#         mapped = torch.reshape(mapped, (-1, self.map_token_len, self.language_model_hidden_dim))
#
#         # @@@@ try 1: Apply masking only during training
#         # if self.training:
#         #     # Create a mask with 50% probability of being 1 (keep) or 0 (mask)
#         #     mask = torch.bernoulli(torch.full(num_mapped.shape, 0.5)).to(num_mapped.device)
#         #     # Apply the mask to num_mapped by element-wise multiplication
#         #     num_mapped = num_mapped * mask
#
#         # mapped = self.ln(mapped)
#         mapped = self.dropout(mapped)
#         return mapped
#
#     def get_output_dim(self):
#         return self.language_model_hidden_dim

# @@@@ try 4
class IndependentFeatureLinearLayers(nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(IndependentFeatureLinearLayers, self).__init__()
        # 创建一个ModuleList，每个feature都有一个线性映射层
        self.linears = nn.ModuleList([nn.Linear(1, hidden_dim) for _ in range(num_features)])

    def forward(self, x):
        # x的维度应为(batch_size, num_features)
        batch_size, num_features = x.shape

        # 对每个特征应用相应的线性层，并将结果存储在一个列表中
        outputs = [self.linears[i](x[:, i:i + 1]).unsqueeze(1) for i in range(num_features)]

        # 将输出从列表形式转换为tensor形式，并在第二维度上拼接结果
        # 每个输出的size变为(batch_size, 1, hidden_dim)，拼接后结果size为(batch_size, num_features, hidden_dim)
        output = torch.cat(outputs, dim=1)

        return output


class NumFeatureExtractor(nn.Module):
    def __init__(self, num_feat_dim, language_model: NumCatLanguageModel):
        super(NumFeatureExtractor, self).__init__()
        self.language_model = language_model
        self.num_feat_dim = num_feat_dim
        self.language_model_hidden_dim = self.language_model.get_output_dim()
        self.ln_before = nn.LayerNorm(self.num_feat_dim)
        self.ifl = IndependentFeatureLinearLayers(self.num_feat_dim,
                                                  self.language_model_hidden_dim)
        self.embedding = language_model.bert.embeddings

        # @@@ below are I have tried:
        # # 初始化一个参数，这个参数会在训练中学习每个特征的重要性
        # self.feature_weights = nn.Parameter(torch.Tensor(num_feat_dim, 1))
        # nn.init.uniform_(self.feature_weights, 0, 1)
        # self.dropout = nn.Dropout(0.7)
        # self.dropout = AdaptiveDropout(0.5, 0.9, 0.5)  # hidden_dropout_prob in NumCatLanguageModelConfig

    # TODO disable position encoding
    def forward(self, x):
        x = self.ln_before(x)
        output = self.ifl(x)

        # @@@ try:
        # scores = torch.sigmoid(self.feature_weights)
        # output = output * scores

        output = self.embedding(inputs_embeds=output)
        # @@@ try:
        # output = self.dropout(output)
        return output

        # @@@ try:
        # return output[:, -1].unsqueeze(1)  # take last cls token feature as num feature!

    def get_output_dim(self):
        return self.language_model_hidden_dim
