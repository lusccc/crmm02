from dataclasses import dataclass
from typing import Optional, Tuple, Any

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import clip_loss
from transformers.utils import logging, ModelOutput

from crmm.models.clncp_config import CLNCPConfig
from crmm.models.feature_extractor_factory import FeatureExtractorFactory

logger = logging.get_logger(__name__)


# Contrastive Language–Numeric-Category Pretraining

@dataclass
class CLNCPOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits_per_num_cat: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    num_cat_joint_features: torch.FloatTensor = None
    text_features: torch.FloatTensor = None
    probs: torch.FloatTensor = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] for k in self.keys()
        )

class CLNCPPreTrainedModel(PreTrainedModel):
    config_class = CLNCPConfig
    base_model_prefix = "clncp"


class CLNCP(CLNCPPreTrainedModel):
    """
    use num cat text modality
    """

    config_class = CLNCPConfig

    def __init__(self, config: CLNCPConfig):
        super().__init__(config)
        self.config = config
        self.use_modality = self.config.use_modality
        self.n_modality = len(self.use_modality)
        self.n_labels = self.config.n_labels
        self.pretrained = self.config.pretrained

        self.feature_extractors_factory = FeatureExtractorFactory(clncp_config=self.config)
        self.feature_extractors = self.feature_extractors_factory.get_feature_extractors()
        self.feature_extractors = nn.ModuleDict(self.feature_extractors)

        self.clip_text_config = self.feature_extractors['text'].clip_text.config
        self.joint_projection = nn.Linear(self.feature_extractors['joint'].get_output_dim(),
                                          self.clip_text_config.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))  # value copied from configuration_clip.py

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, return_loss=True, **inputs, ):
        # should return_loss=True, or cause error in eval
        # `return_loss=` will be checked in transformers.utils.generic.can_return_loss

        # inputs: {'labels': ...,
        #         'text': {'input_ids',:..., 'attention_mask':...},
        #         'num': ...,
        #         'cat': ...}
        num_features = self.feature_extractors['num'](inputs['num'])
        cat_features = self.feature_extractors['cat'](inputs['cat'])
        num_cat_joint_features = self.feature_extractors['joint']({'num': num_features, 'cat': cat_features})
        num_cat_joint_features = self.joint_projection(num_cat_joint_features)
        text_features = self.feature_extractors['text'](**inputs['text'])

        # normalized features
        num_cat_joint_features = num_cat_joint_features / num_cat_joint_features.norm(p=2, dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_num_cat = torch.matmul(num_cat_joint_features, text_features.t()) * logit_scale
        logits_per_text = logits_per_num_cat.t()

        if return_loss and not self.pretrained:
            loss = clip_loss(logits_per_text)
        else:
            loss = torch.tensor(0.)

        probs = logits_per_num_cat.softmax(dim=1)  # we can take the softmax to get the label probabilities

        if not self.pretrained:
            return CLNCPOutput(
                loss=loss,
                logits_per_num_cat=logits_per_num_cat,
                logits_per_text=logits_per_text,
                num_cat_joint_features=num_cat_joint_features,
                text_features=text_features,
                probs=probs
            )
        else:
            return loss, probs
