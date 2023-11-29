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


def normalize_features(features):
    return features / features.norm(p=2, dim=-1, keepdim=True)


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
        self.mode = self.config.mode
        self.loss_weights = self.config.loss_weights
        self.contrastive_targets = set(self.config.contrastive_targets) if self.config.contrastive_targets is not None \
            else None
        self.feature_extractors = self._get_feature_extractors()
        self.joint_projection, final_feature_dim = self._get_projection_and_feature_dim()
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))  # value copied from configuration_clip.py
        if self.mode in ['finetune', 'multi_task']:
            self.classifier = get_classifier(input_dim=final_feature_dim,
                                             n_class=self.n_labels)
        self.post_init()

    def _get_feature_extractors(self):
        feature_extractors_factory = FeatureExtractorFactory(clncp_config=self.config)
        feature_extractors = feature_extractors_factory.get_feature_extractors()
        return nn.ModuleDict(feature_extractors)

    def _get_projection_and_feature_dim(self):
        if self.feature_extractors['text'].get_output_dim() != self.feature_extractors['joint'].get_output_dim():
            joint_projection_dim = self.feature_extractors['text'].get_output_dim()
            joint_projection = nn.Linear(self.feature_extractors['joint'].get_output_dim(),
                                         joint_projection_dim, bias=False)
            return joint_projection, joint_projection_dim
        else:
            return None, self.feature_extractors['text'].get_output_dim()

    def forward(self, return_loss=True, **inputs):
        # Choose the correct step function based on the mode
        step_function = getattr(self, f"{self.mode}_step")
        return step_function(inputs)

    def pretrain_step(self, inputs):
        if self.contrastive_targets == {'num', 'cat'}:
            return self.contrastive_num_cat_step(inputs)
        elif self.contrastive_targets == {'joint', 'text'}:
            return self.contrastive_joint_text_step(inputs)
        elif self.contrastive_targets == {'text', 'num'}:
            return self.contrastive_num_text_step(inputs)

    def pair_match_prediction_step(self, inputs):
        return self.contrastive_joint_text_step(inputs)

    def finetune_step(self, inputs):
        return self.classification_step(inputs)

    def multi_task_step(self, inputs):
        joint_features = self._extract_joint_features(inputs)
        l1, _ = self.contrastive_step(joint_features, self.feature_extractors['text'](inputs['text']))
        l2, logits = self.classification_step(inputs, joint_features)
        return l1 + l2, logits

    def classification_step(self, inputs, joint_features=None):
        if joint_features is None:
            joint_features = self._extract_joint_features(inputs)
        loss, logits, classifier_layer_outputs = hf_loss_func(joint_features,
                                                              self.classifier,
                                                              inputs['labels'],
                                                              self.n_labels,
                                                              None)
        return loss, logits

    def _extract_joint_features(self, inputs):
        modality_features = {m: self.feature_extractors[m](inputs[m]) for m in self.use_modality}
        joint_features = self.feature_extractors['joint']({m: modality_features[m] for m in self.fuse_modality})
        if self.joint_projection is not None:
            joint_features = self.joint_projection(joint_features)
        return joint_features

    def contrastive_num_cat_step(self, inputs):
        num_features = self.feature_extractors['num'](inputs['num'])
        cat_features = self.feature_extractors['cat'](inputs['cat'])
        return self.contrastive_step(num_features, cat_features)

    def contrastive_num_text_step(self, inputs):
        num_features = self.feature_extractors['num'](inputs['num'])
        text_features = self.feature_extractors['text'](inputs['text'])
        return self.contrastive_step(num_features, text_features)

    def contrastive_joint_text_step(self, inputs):
        joint_features = self._extract_joint_features(inputs)
        text_features = self.feature_extractors['text'](inputs['text'])
        return self.contrastive_step(joint_features, text_features)

    def contrastive_step(self, m1_features, m2_features):
        m1_features = normalize_features(m1_features)
        m2_features = normalize_features(m2_features)
        logit_scale = self.logit_scale.exp()
        logits_per_m1 = torch.matmul(m1_features, m2_features.t()) * logit_scale
        logits_per_m2 = logits_per_m1.t()
        loss = clip_loss(logits_per_m2) if self.mode != 'pair_match_prediction' else torch.tensor(0.)
        logits = logits_per_m1.softmax(dim=1)
        return loss, logits

    # def forward___(self, return_loss=True, **inputs, ):
    #     """
    #      should return_loss=True, or cause error in eval
    #      `return_loss=` will be checked in transformers.utils.generic.can_return_loss
    #
    #      inputs: {'labels': ...,
    #              'text': ...,
    #              'num': ...,
    #              'cat': ...}
    #     """
    #     if not self.pretrained:
    #         modality_features = {}
    #         if {'num', 'cat'} == self.contrastive_targets:
    #             for modality in self.contrastive_targets:
    #                 modality_features[modality] = self.feature_extractors[modality](inputs[modality])
    #             return self.contrastive_step(modality_features['num'], modality_features['cat'])
    #         if {'text', 'joint'} == self.contrastive_targets:
    #             for modality in ['num', 'cat', 'text']:
    #                 modality_features[modality] = self.feature_extractors[modality](inputs[modality])
    #             joint_features = self.feature_extractors['joint']({m: modality_features[m] for m in self.fuse_modality})
    #             if self.joint_projection is not None:
    #                 joint_features = self.joint_projection(joint_features)
    #             modality_features['joint'] = joint_features
    #             return self.contrastive_step(modality_features['joint'], modality_features['text'])
    #     else:
    #
    #     pair_match_prediction = False if len(inputs['num']) == len(inputs['text']['input_ids']) else True
    #
    #     num_features = self.feature_extractors['num'](inputs['num'])
    #     cat_features = self.feature_extractors['cat'](inputs['cat'])
    #     text_features = self.feature_extractors['text'](inputs['text'])
    #     modality_features = {'num': num_features, 'cat': cat_features, 'text': text_features}
    #
    #     joint_features = self.feature_extractors['joint']({m: modality_features[m] for m in self.fuse_modality})
    #     if self.joint_projection is not None:
    #         joint_features = self.joint_projection(joint_features)
    #     joint_features_normed = joint_features / joint_features.norm(p=2, dim=-1, keepdim=True)
    #     text_features_normed = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    #
    #     logit_scale = self.logit_scale.exp()
    #     logits_per_numcat = torch.matmul(joint_features_normed, text_features_normed.t()) * logit_scale
    #     logits_per_text = logits_per_numcat.t()
    #     loss = 0 if pair_match_prediction else clip_loss(logits_per_text)
    #     logits = logits_per_numcat.softmax(dim=1)
    #
    #     if (self.multi_task or self.pretrained) and (not pair_match_prediction):
    #         cls_loss, logits, classifier_layer_outputs = hf_loss_func(joint_features,
    #                                                                   self.classifier,
    #                                                                   inputs['labels'],
    #                                                                   self.n_labels,
    #                                                                   None)
    #         loss += cls_loss if self.multi_task else 0
    #
    #     return loss, logits

# def multi_clip_loss(feature_list: List[torch.Tensor], logit_scale: torch.Tensor) -> torch.Tensor:
#     losses = []
#     for i in range(len(feature_list)):
#         for j in range(i + 1, len(feature_list)):
#             logits_ij = torch.matmul(feature_list[i], feature_list[j].t()) * logit_scale
#             logits_ji = logits_ij.t()
#             loss_ij = contrastive_loss(logits_ij)
#             loss_ji = contrastive_loss(logits_ji)
#             losses.append((loss_ij + loss_ji) / 2.0)
#     return sum(losses) / len(losses)
