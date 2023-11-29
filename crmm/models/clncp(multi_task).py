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
from crmm.models.layer_utils import get_classifier, hf_loss_func

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
        self.multitask = self.config.multitask
        self.loss_weights = self.config.loss_weights

        self.feature_extractors_factory = FeatureExtractorFactory(clncp_config=self.config)
        self.feature_extractors = self.feature_extractors_factory.get_feature_extractors()
        self.feature_extractors = nn.ModuleDict(self.feature_extractors)

        if self.feature_extractors['text'].get_output_dim() != self.feature_extractors['joint'].get_output_dim():
            # align output dims to calc cosine sim
            self.joint_projection_dim = self.feature_extractors['text'].get_output_dim()
            self.joint_projection = nn.Linear(self.feature_extractors['joint'].get_output_dim(),
                                              self.joint_projection_dim, bias=False)
            final_feature_dim = self.joint_projection_dim
        else:
            final_feature_dim = self.feature_extractors['text'].get_output_dim()
            self.joint_projection = None

        self.logit_scale = nn.Parameter(torch.tensor(2.6592))  # value copied from configuration_clip.py

        if self.pretrained:
            self.classifier = get_classifier(input_dim=final_feature_dim,
                                             n_class=self.n_labels) if self.pretrained else None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, return_loss=True, **inputs, ):
        """
         should return_loss=True, or cause error in eval
         `return_loss=` will be checked in transformers.utils.generic.can_return_loss

         inputs: {'labels': ...,
                 'text': ...,
                 'num': ...,
                 'cat': ...}
        """
        pair_match_prediction = False if len(inputs['num']) == len(inputs['text']['input_ids']) else True

        num_features = self.feature_extractors['num'](inputs['num'])
        cat_features = self.feature_extractors['cat'](inputs['cat'])
        text_features = self.feature_extractors['text'](inputs['text'])
        modality_features = {'num': num_features, 'cat': cat_features, 'text': text_features}
        joint_features = self.feature_extractors['joint']({m: modality_features[m] for m in self.fuse_modality})
        if self.joint_projection is not None:
            joint_features = self.joint_projection(joint_features)
        joint_features_normed = joint_features / joint_features.norm(p=2, dim=-1, keepdim=True)
        text_features_normed = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_numcat = torch.matmul(joint_features_normed, text_features_normed.t()) * logit_scale
        logits_per_text = logits_per_numcat.t()
        loss = 0 if pair_match_prediction else clip_loss(logits_per_text)
        logits = logits_per_numcat.softmax(dim=1)

        if (self.multitask or self.pretrained) and (not pair_match_prediction):
            cls_loss, logits, classifier_layer_outputs = hf_loss_func(joint_features,
                                                                      self.classifier,
                                                                      inputs['labels'],
                                                                      self.n_labels,
                                                                      None)
            loss += cls_loss if self.multitask else 0

        return loss, logits


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
