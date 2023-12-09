from typing import Optional, Union, Tuple
from transformers.utils import logging
import torch
import torch.nn as nn
from transformers import CLIPTextModelWithProjection, CLIPTextConfig, RobertaModel, RobertaConfig
from transformers.models.clip.modeling_clip import CLIPTextModelOutput

logger = logging.get_logger('transformers')


class LanguageModel(nn.Module):

    def __init__(self,
                 pretrained_model_name_or_path,
                 load_hf_pretrained=True,
                 local_files_only=False,
                 cache_dir=None,
                 max_seq_length=512,
                 freeze_params=False):
        super(LanguageModel, self).__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.load_hf_pretrained = load_hf_pretrained
        self.local_files_only = local_files_only
        self.cache_dir = cache_dir
        self.max_seq_length = max_seq_length
        self.freeze_params = freeze_params

        if self.load_hf_pretrained:
            self.roberta = RobertaModel.from_pretrained(self.pretrained_model_name_or_path,
                                                        local_files_only=self.local_files_only,
                                                        cache_dir=self.cache_dir)
        else:
            # the config we will not be modified, hence still load from hf!
            language_model_config = RobertaConfig.from_pretrained(self.pretrained_model_name_or_path,
                                                                  local_files_only=self.local_files_only,
                                                                  cache_dir=self.cache_dir)
            self.roberta = RobertaModel(config=language_model_config)

        if self.freeze_params:
            # freeze clip_text params, only train the classifier layer
            self.roberta.requires_grad_(False)
            logger.warning("text model params are frozen!")
        else:
            self.roberta.requires_grad_(True)
            logger.warning("text model params are now unfrozen!")

    def forward(self, inputs):
        outputs = self.roberta(**inputs)
        """
        outputs[0] 取的是RoBERTa模型最后一层的hidden states,它的shape是 [batch_size, seq_len, hidden_size]。
        [:, 0, :] 表示取第0个token,也就是<s> token的表示,shape是[batch_size, hidden_size]。
        因为RoBERTa模型在输入序列最前面会添加<s> token,它起到和BERT中的[CLS] token类似的作用,可以用来做分类任务。
        所以这里通过取<s> token的表示,得到一个固定长度的向量,可以用于分类等任务中。
        所以这段代码的作用就是从RoBERTa模型输出中提取出<s> token对应的表示,相当于是得到了一个文本的整体表示,可以用来做分类等任务。
        """
        # take <s> token (equiv. to [CLS]),
        # also see: transformers.models.roberta.modeling_roberta.RobertaClassificationHead
        sequence_output = outputs[0][:, 0, :]
        return sequence_output

    def get_output_dim(self):
        return self.roberta.config.hidden_size
