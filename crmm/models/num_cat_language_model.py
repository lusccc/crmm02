import torch.nn as nn
from transformers import BertConfig, BertModel

from crmm.models.clncp_config import CLNCPConfig

class NumCatLanguageModelConfig(BertConfig):
    pass

class NumCatLanguageModel(nn.Module):
    def __init__(self, clncp_config: CLNCPConfig):
        super(NumCatLanguageModel, self).__init__()
        self.config = NumCatLanguageModelConfig(
            # hidden_size=256,
            hidden_size=512,
            num_hidden_layers=6,
            num_attention_heads=8,
            intermediate_size=2048,
            vocab_size=clncp_config.cat_vocab_size,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=1,  # is not sentence pair task, so set to 1
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=3,  # should be same value as cat_tokenizer in main_runner.py
            position_embedding_type="absolute",
            use_cache=True,
            classifier_dropout=None,
        )
        self.bert = BertModel(self.config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

    def forward(self, inputs):
        bert_outputs = self.bert(**inputs)
        # @@@@ try 1:
        # output = bert_outputs[0][:, 0, :]  # 提取[CLS]标记的输出用于分类 ？？？

        # @@@@ try 2: as BertForSequenceClassification did
        # pooled_output = bert_outputs[1]
        # pooled_output = self.dropout(pooled_output)

        # @@@@ try 3, Taking only CLS token for the prediction head (added by AppendCLSToken)
        output = bert_outputs[0][:, -1]

        # TODO for loop, each cat to go through wordemb
        # TODO 2,num cls token feature, and add cat cls toeken feature,

        return output

    def get_output_dim(self):
        return self.bert.config.hidden_size
