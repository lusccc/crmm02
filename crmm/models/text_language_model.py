from transformers.utils import logging
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig
from transformers.utils import logging

logger = logging.get_logger('transformers')


class TextLanguageModel(nn.Module):

    def __init__(self,
                 pretrained_model_name_or_path,
                 load_hf_pretrained=True,
                 local_files_only=False,
                 cache_dir=None,
                 max_seq_length=512,
                 freeze_params=False):
        super(TextLanguageModel, self).__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.load_hf_pretrained = load_hf_pretrained
        self.local_files_only = local_files_only
        self.cache_dir = cache_dir
        self.max_seq_length = max_seq_length
        self.freeze_params = freeze_params

        model_class = RobertaModel
        config_class = RobertaConfig

        # @@@ try 1:
        # model_class = BertModel
        # config_class = BertConfig

        # @@@ try 2:
        # model_class = GPT2Model
        # config_class = GPT2Config

        if self.load_hf_pretrained:
            self.lm = model_class.from_pretrained(self.pretrained_model_name_or_path,
                                                  local_files_only=self.local_files_only,
                                                  cache_dir=self.cache_dir)
        else:
            # the config we will not be modified, hence still load from hf!
            language_model_config = config_class.from_pretrained(self.pretrained_model_name_or_path,
                                                                 local_files_only=self.local_files_only,
                                                                 cache_dir=self.cache_dir)
            self.lm = model_class(config=language_model_config)

        if self.freeze_params:
            # @@@ try 1: freeze entire robert
            # freeze clip_text params, only train the classifier layer
            self.lm.requires_grad_(False)
            logger.warning("try 1: freeze entire robert but pooler!")

            # @@@ try 2: freeze all but embedding, add & layer norm, pooler
            # self.lm.requires_grad_(False)
            # self.lm.embeddings.requires_grad_(True)
            # self.lm.pooler.requires_grad_(True)
            # for roberta_layer in self.lm.encoder.layer:  # i.e., transformers.models.lm.modeling_roberta.RobertaLayer
            #     """
            #     encoder由多个attention组成，即多头。代码中是transformers.models.lm.modeling_roberta.RobertaLayer,
            #     每一个roberta_layer包含一个RobertaAttention，每一个RobertaAttention在代码中实际上是
            #     RobertaSelfAttention和RobertaSelfOutput组成，而RobertaSelfOutput，包含dense和add&layernorm
            #     说明，RobertaAttention代码中实际上attention包含了multiheadattentin和add&layernorm！
            #     然后回到roberta_layer，然后再是FFN即RobertaIntermediate，再是另一个add&layernorm即RobertaOutput！
            #     """
            #     roberta_layer.attention.output.LayerNorm.requires_grad_(True)
            #     roberta_layer.output.LayerNorm.requires_grad_(True)
            # logger.warning("try 2: freeze part robert!")
        else:
            self.lm.requires_grad_(True)
            logger.warning("text model params are now unfrozen!")

    def forward(self, inputs):
        outputs = self.lm(**inputs)
        """
        outputs[0] 取的是RoBERTa模型最后一层的hidden states,它的shape是 [batch_size, seq_len, hidden_size]。
        [:, 0, :] 表示取第0个token,也就是<s> token的表示,shape是[batch_size, hidden_size]。
        因为RoBERTa模型在输入序列最前面会添加<s> token,它起到和BERT中的[CLS] token类似的作用,可以用来做分类任务。
        所以这里通过取<s> token的表示,得到一个固定长度的向量,可以用于分类等任务中。
        所以这段代码的作用就是从RoBERTa模型输出中提取出<s> token对应的表示,相当于是得到了一个文本的整体表示,可以用来做分类等任务。
        """
        # take <s> token (equiv. to [CLS]),
        # also see: transformers.models.lm.modeling_roberta.RobertaClassificationHead
        sequence_output = outputs[0][:, 0, :]
        return sequence_output

    # @@@ try 2:
    # def forward(self, inputs):
    #     outputs = self.lm(**inputs)
    #     """
    #     outputs[0] 是GPT-2模型最后一层的hidden states，它的shape是 [batch_size, seq_len, hidden_size]。
    #     [:, -1, :] 表示取序列中最后一个token的表示，其shape是 [batch_size, hidden_size]。
    #     在GPT-2中，通常可以使用最后一个token的表示来进行分类任务，
    #     因为它被认为包含了整个序列的信息。
    #     """
    #     # take the last token representation
    #     sequence_output = outputs[0][:, -1, :]
    #     return sequence_output

    def get_output_dim(self):
        return self.lm.config.hidden_size
