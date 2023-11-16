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
            self.text_model = RobertaModel.from_pretrained(self.pretrained_model_name_or_path,
                                                           local_files_only=self.local_files_only,
                                                           cache_dir=self.cache_dir)
        else:
            # the config we will not be modified, hence still load from hf!
            text_model_config = RobertaConfig.from_pretrained(self.pretrained_model_name_or_path,
                                                              local_files_only=self.local_files_only,
                                                              cache_dir=self.cache_dir)
            self.text_model = RobertaModel(config=text_model_config)

        if self.freeze_params:
            # freeze clip_text params, only train the classifier layer
            self.text_model.requires_grad_(False)
            logger.warning("text model params are frozen!")

    def forward(self, input):
        outputs = self.text_model(**input)
        sequence_output = outputs[0][:, 0, :]  # take <s> token (equiv. to [CLS])
        return sequence_output

    def get_output_dim(self):
        return self.text_model.config.hidden_size
