import torch
import torch.nn as nn

from crmm.models.num_cat_language_model import NumCatLanguageModelConfig, NumCatLanguageModel


# @@@@ try 1
# class CatFeatureExtractor(nn.Module):
#     def __init__(self, language_model: TextLanguageModel):
#         super().__init__()
#         # self.output_dim = 512
#         self.language_model = language_model
#         # self.encoder = MLP(input_dim=self.language_model.get_output_dim(),
#         #                    output_dim=self.output_dim,
#         #                    act='relu',
#         #                    num_hidden_lyr=2,
#         #                    dropout_prob=0.5,
#         #                    return_layer_outs=False,
#         #                    hidden_channels=[640, 576],
#         #                    bn=False)
#
#     def forward(self, input):
#         output = self.language_model(input)
#         # output = self.encoder(output)
#         return output
#
#     def get_output_dim(self):
#         # return self.output_dim
#         return self.language_model.get_output_dim()

# @@@@ try 2
# class CatFeatureExtractor(nn.Module):
#     def __init__(self, language_model: TextLanguageModel):
#         super().__init__()
#         self.language_model = language_model
#         self.embedding = self.language_model.roberta.embeddings
#
#     def forward(self, inputs):
#         if 'attention_mask' in inputs:
#             inputs.pop('attention_mask')
#         # output = self.embedding(**inputs).mean(dim=1)
#         output = self.embedding(**inputs)
#         return output
#
#     def get_output_dim(self):
#         return self.language_model.get_output_dim()  # i.e., roberta.config.hidden_size


# @@@@ try 3: share the language model with joint network, means the embedding net weights are shared
class CatFeatureExtractor(nn.Module):
    def __init__(self, language_model: NumCatLanguageModel):
        super().__init__()
        self.embedding = language_model.bert.embeddings

        self.output_dim = language_model.get_output_dim()

    def forward(self, inputs):
        if 'attention_mask' in inputs:
            inputs.pop('attention_mask')
        # output = self.embedding(**inputs).mean(dim=1)
        output = self.embedding(**inputs)
        return output

    def get_output_dim(self):
        return self.output_dim

# @@@@ try 4: independent cat model
# class CatFeatureExtractor(nn.Module):
#     def __init__(self, num_cat_lm_config: NumCatLanguageModelConfig):
#         super().__init__()
#         self.word_embeddings = nn.Embedding(num_cat_lm_config.vocab_size,
#                                             num_cat_lm_config.hidden_size,
#                                             padding_idx=num_cat_lm_config.pad_token_id)
#         self.position_embeddings = nn.Embedding(num_cat_lm_config.max_position_embeddings,
#                                                 num_cat_lm_config.hidden_size)
#         self.LayerNorm = nn.LayerNorm(num_cat_lm_config.hidden_size, eps=num_cat_lm_config.layer_norm_eps)
#         self.dropout = nn.Dropout(num_cat_lm_config.hidden_dropout_prob)
#         self.position_embedding_type = num_cat_lm_config.position_embedding_type
#         self.register_buffer(
#             "position_ids", torch.arange(num_cat_lm_config.max_position_embeddings).expand((1, -1)), persistent=False
#         )
#
#         self.output_dim = num_cat_lm_config.hidden_size
#
#     def forward(self, inputs):
#         """
#         partially copied from transformers.models.bert.modeling_bert.BertEmbeddings.forward
#         """
#         embeddings = self.word_embeddings(inputs['input_ids'])
#         if self.position_embedding_type == "absolute":
#             seq_length = inputs['input_ids'].size()[1]
#             position_ids = self.position_ids[:, 0: seq_length + 0]
#             position_embeddings = self.position_embeddings(position_ids)
#             embeddings += position_embeddings
#         embeddings = self.LayerNorm(embeddings)
#         embeddings = self.dropout(embeddings)
#         return embeddings
#
#     def get_output_dim(self):
#         return self.output_dim
