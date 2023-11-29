import torch.nn as nn

from crmm.models.language_model import LanguageModel
from crmm.models.layer_utils import MLP


class TextFeatureExtractor(nn.Module):
    def __init__(self, language_model: LanguageModel):
        super().__init__()
        # self.output_dim = 512
        self.language_model = language_model
        # self.encoder = MLP(input_dim=self.language_model.get_output_dim(),
        #                    output_dim=self.get_output_dim(),
        #                    act='relu',
        #                    num_hidden_lyr=2,
        #                    dropout_prob=0.5,
        #                    return_layer_outs=False,
        #                    hidden_channels=[640, 576],
        #                    bn=False)

    def forward(self, input):
        output = self.language_model(input)
        # output = self.encoder(output)
        return output

    def get_output_dim(self):
        # return self.output_dim
        return self.language_model.get_output_dim()
