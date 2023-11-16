from dataclasses import dataclass
from typing import Optional, Tuple, Any, List

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import clip_loss, contrastive_loss
from transformers.utils import logging, ModelOutput

from crmm.models.clncp_config import CLNCPConfig
from crmm.models.feature_extractor_factory import FeatureExtractorFactory

logger = logging.get_logger('transformers')


# the proposed CLNCP: Contrastive Language–Numeric-Category Pretraining


@dataclass
class CLNCPOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None

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
        self.fuse_modality = self.config.fuse_modality
        self.n_modality = len(self.use_modality)
        self.n_labels = self.config.n_labels
        self.pretrained = self.config.pretrained
        self.loss_weights = self.config.loss_weights

        self.feature_extractors_factory = FeatureExtractorFactory(clncp_config=self.config)
        self.feature_extractors = self.feature_extractors_factory.get_feature_extractors()
        self.feature_extractors = nn.ModuleDict(self.feature_extractors)

        self.joint_projection = nn.Linear(self.feature_extractors['joint'].get_output_dim(),
                                          512, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))  # value copied from configuration_clip.py

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, return_loss=True, **inputs, ):
        """
         should return_loss=True, or cause error in eval
         `return_loss=` will be checked in transformers.utils.generic.can_return_loss

         inputs: {'labels': ...,
                 'text': {'input_ids',:..., 'attention_mask':...},
                 'num': ...,
                 'cat': ...}
        """
        num_features = self.feature_extractors['num'](inputs['num'])
        cat_features = self.feature_extractors['cat'](inputs['cat'])
        text_features = self.feature_extractors['text'](inputs['text'])
        modality_features = {'num': num_features, 'cat': cat_features, 'text': text_features}
        joint_features = self.feature_extractors['joint']({m: modality_features[m] for m in self.fuse_modality})
        joint_features = self.joint_projection(joint_features)
        all_features = [num_features, cat_features, text_features, joint_features]

        # normalized features
        for i in range(len(all_features)):
            all_features[i] = all_features[i] / all_features[i].norm(p=2, dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_numcat = torch.matmul(joint_features, text_features.t()) * logit_scale
        logits_per_text = logits_per_numcat.t()

        if return_loss and not self.pretrained:
            loss = clip_loss(logits_per_text)
        else:
            loss = torch.tensor(0.)

        # logits_per_num = torch.matmul(num_features, text_features.t()) * logit_scale
        # logits_per_cat = torch.matmul(cat_features, text_features.t()) * logit_scale
        # logits_per_numcat = torch.matmul(joint_num_cat_features, text_features.t()) * logit_scale

        probs = logits_per_numcat.softmax(dim=1)
        # probs = logits_per_cat.softmax(dim=1)
        # probs = logits_per_num.softmax(dim=1)

        return loss, probs


def multi_clip_loss(feature_list: List[torch.Tensor], logit_scale: torch.Tensor) -> torch.Tensor:
    losses = []
    for i in range(len(feature_list)):
        for j in range(i + 1, len(feature_list)):
            logits_ij = torch.matmul(feature_list[i], feature_list[j].t()) * logit_scale
            logits_ji = logits_ij.t()
            loss_ij = contrastive_loss(logits_ij)
            loss_ji = contrastive_loss(logits_ji)
            losses.append((loss_ij + loss_ji) / 2.0)
    return sum(losses) / len(losses)
