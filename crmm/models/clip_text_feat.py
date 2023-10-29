from typing import Optional, Union, Tuple

import torch
import torch.nn as nn
from transformers import CLIPTextModelWithProjection, CLIPTextConfig
from transformers.models.clip.modeling_clip import CLIPTextModelOutput


class TextFeatureExtractor(nn.Module):
    def __init__(self, max_seq_length=512, freeze_clip_params=False):
        super(TextFeatureExtractor, self).__init__()
        self.max_seq_length = max_seq_length
        self.freeze_clip_params = freeze_clip_params

        clip_text_config = CLIPTextConfig(max_position_embeddings=self.max_seq_length)
        self.clip_text = CLIPTextModelWithProjection(config=clip_text_config)

        if self.freeze_clip_params:
            # freeze clip_text params, only train the classifier layer
            self.clip_text.requires_grad_(False)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CLIPTextModelOutput]:
        outputs = self.clip_text(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        text_embeds = outputs.text_embeds
        return text_embeds

    def get_output_dim(self):
        return self.clip_text.config.projection_dim
