import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from crmm.models.language_model import LanguageModel
from crmm.models.layer_utils import MLP


class LightMapping(nn.Module):
    """
    section 3.1 in paper: ClipCap: CLIP Prefix for Image Captioning
    """

    def forward(self, x):
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(LightMapping, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class JointFeatureExtractor(nn.Module):
    def __init__(self, modality_feat_dims, language_model: LanguageModel):
        super().__init__()
        self.language_model = language_model
        self.prefix_length = 1  # see paper: ClipCap: CLIP Prefix for Image Captioning
        self.language_model_hidden_dim = self.language_model.get_output_dim()
        self.num_light_mapping = LightMapping((modality_feat_dims['num'],
                                               modality_feat_dims['num'] * self.prefix_length // 2,
                                               self.language_model_hidden_dim * self.prefix_length))
        # self.encoder = MLP(input_dim=self.language_model_hidden_dim,  # num: 512 ,768
        #                    output_dim=self.get_output_dim(),
        #                    act='relu',
        #                    num_hidden_lyr=2,
        #                    dropout_prob=0.1,
        #                    return_layer_outs=False,
        #                    hidden_channels=[640, 576],
        #                    bn=True)
        # self.output_light_mapping = LightMapping((self.language_model_hidden_dim,
        #                                           self.language_model_hidden_dim - 128,
        #                                           self.get_output_dim()))

    def forward(self, inputs):
        num_mapped = self.num_light_mapping(inputs['num'])
        num_mapped = torch.reshape(num_mapped, (-1, self.prefix_length, self.language_model_hidden_dim))
        embeds = torch.cat([num_mapped, inputs['cat'], ], dim=1)
        output = self.language_model({'inputs_embeds': embeds})
        # output = self.output_light_mapping(output)
        return output

    def get_output_dim(self):
        return self.language_model_hidden_dim
