import torch.nn as nn

from crmm.models.text_language_model import TextLanguageModel


class TextFeatureExtractor(nn.Module):
    def __init__(self, language_model: TextLanguageModel):
        super().__init__()
        self.language_model = language_model

        # @@@ try:
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
        # @@@ try:
        # output = self.encoder(output)
        return output

    def get_output_dim(self):
        return self.language_model.get_output_dim()
