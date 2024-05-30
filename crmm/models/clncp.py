import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.models.clip.modeling_clip import clip_loss
from transformers.utils import logging

from crmm.models.clncp_config import CLNCPConfig
from crmm.models.feature_extractor_factory import FeatureExtractorFactory
from crmm.models.layer_utils import create_classifier, hf_loss_func

logger = logging.get_logger('transformers')

# the proposed CLNCP: Contrastive Language–Numeric-Category Pretraining
# TODO  change to CLNC-MTL: Contrastive Language–Numeric-Category multitask learning
# TODO  change to CNCT-MTL Contrastive Numeric-Category-Text multitask learning

TASK_MODE_DICT = {
    'pretrain': 'contrastive',
    'pair_match_evaluation': 'contrastive_evaluation',
    'finetune_classification': 'classification',
    'finetune_classification_scratch': 'classification',
    'finetune_classification_evaluation': 'classification',
    'multi_task_classification': 'contrastive_and_classification',
}


# @@@ try 1: stacking
# TODO, calc Correlation coefficient with extracted feature?
class StackingEnsembleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 32)
        self.layer2 = nn.Linear(32, 2)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits1, logits2, labels):
        x = torch.cat([torch.sigmoid(logits1), torch.sigmoid(logits2)], dim=-1)
        # x = F.gelu(self.layer1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        loss = self.loss_fn(x, labels)
        return loss, x


# @@@ try 2: weighted avg
class WeightedAvgEnsembleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.ones(1))
        self.w2 = nn.Parameter(torch.ones(1))
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits1, logits2, labels):
        combined_logits = torch.sigmoid(self.w1)* logits1 + torch.sigmoid(self.w2)*logits2
        loss = self.loss_fn(combined_logits, labels)
        return loss, combined_logits


class CLNCPPreTrainedModel(PreTrainedModel):
    config_class = CLNCPConfig
    base_model_prefix = "clncp"


class CLNCP(CLNCPPreTrainedModel):
    config_class = CLNCPConfig

    def __init__(self, config: CLNCPConfig):
        super().__init__(config)
        self.config = config
        self.use_modality = self.config.use_modality  # indicate which modalities involved
        self.fuse_modality = self.config.fuse_modality  # can be num, cat, num&cat
        self.n_modality = len(self.use_modality)
        self.n_labels = self.config.n_labels
        self.mode = self.config.mode

        self.contrastive_targets = set(self.config.contrastive_targets) \
            if self.config.contrastive_targets is not None else None
        self.feature_extractors = nn.ModuleDict(FeatureExtractorFactory(clncp_config=self.config).auto_create_all())
        self.projection_layer = self.get_projection_layer()
        self.classifier = self.get_classifier()
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))  # value copied from configuration_clip.py
        self.stop_contrastive = False

        if self.config.clncp_ensemble_method == 'weighted_avg':
            self.ensemble_model = WeightedAvgEnsembleModel()
        elif self.config.clncp_ensemble_method == 'stacking':
            self.ensemble_model = StackingEnsembleModel()
        elif self.config.clncp_ensemble_method == 'no_ensemble':
            self.ensemble_model = None
        elif self.config.clncp_ensemble_method == 'only_nll_pair_match':
            self.ensemble_model = None
        else:
            raise ValueError('clncp_ensemble_method not specified')

        self.post_init()

    def get_classifier(self):
        classifier = None
        if self.mode not in ['contrastive', 'contrastive_evaluation']:
            if len(self.fuse_modality) == 1:
                classifier = create_classifier(
                    input_dim=self.feature_extractors[self.fuse_modality[0]].get_output_dim(),
                    n_class=self.n_labels)
            elif len(self.fuse_modality) == 2:
                classifier = create_classifier(
                    input_dim=self.feature_extractors['joint'].get_output_dim(),
                    n_class=self.n_labels
                )
            else:
                raise ValueError('Unrecognized')
        return classifier

    def get_projection_layer(self):
        """
        the projection layer is used to align the dim with semantic text model, to calc contrastive loss
        """
        joint_projection = None
        if self.mode in ['contrastive', 'contrastive_evaluation', 'contrastive_and_classification']:
            text_output_dim = self.feature_extractors['text'].get_output_dim()
            contrastive_targets = list(self.contrastive_targets)
            m1 = 'text'
            m2 = contrastive_targets[1] if contrastive_targets[0] == 'text' else contrastive_targets[0]
            if m2 == 'joint':
                num_or_cat_or_joint_output_dim = self.feature_extractors['joint'].get_output_dim()
            else:
                num_or_cat_or_joint_output_dim = self.feature_extractors[m2].get_output_dim()
            if num_or_cat_or_joint_output_dim != text_output_dim:
                joint_projection = nn.Linear(num_or_cat_or_joint_output_dim, text_output_dim, bias=False)
        return joint_projection

    def modality_features_step(self, inputs):
        modality_features = {'num': None, 'cat': None, 'text': None, 'joint': None, }

        def update_modality_features(modalities):
            for m in modalities:
                if m != 'joint' and modality_features[m] is None:
                    modality_features[m] = self.feature_extractors[m](inputs[m])

        def update_joint_feature():
            if (isinstance(self.fuse_modality, list)
                    and len(self.fuse_modality) > 1
                    and modality_features['joint'] is None):
                modality_features['joint'] = self.feature_extractors['joint'](
                    {m: modality_features[m] for m in self.fuse_modality}
                )

        if self.mode in ['contrastive', 'contrastive_evaluation', 'contrastive_and_classification']:
            update_modality_features(self.contrastive_targets)
            if 'joint' in self.contrastive_targets:
                update_modality_features(['num', 'cat'])
                update_joint_feature()

        if self.mode in ['classification', 'contrastive_and_classification']:
            update_modality_features(self.use_modality)
            update_joint_feature()

        return modality_features

    def forward(self, return_loss=True, **inputs):
        modality_features = self.modality_features_step(inputs)
        if self.mode in ['contrastive', 'contrastive_evaluation', 'contrastive_and_classification']:
            contrastive_targets = list(self.contrastive_targets)
            m1 = contrastive_targets[1] if contrastive_targets[0] == 'text' else contrastive_targets[0]
            m2 = 'text'
            m1_features = modality_features[m1]
            m2_features = modality_features[m2]
            if self.projection_layer is not None:
                m1_features = self.projection_layer(m1_features)
            cts_loss, cts_logits = self.contrastive_step(m1_features, m2_features)

        if self.mode in ['classification', 'contrastive_and_classification']:
            if len(self.fuse_modality) == 1:
                extracted_features = modality_features[self.fuse_modality[0]]
            elif len(self.fuse_modality) == 2:
                extracted_features = modality_features['joint']
            else:
                raise ValueError('wrong fuse_modality!')
            cls_loss, cls_logits, _ = hf_loss_func(extracted_features, self.classifier, inputs['labels'], self.n_labels)

        if self.mode in ['contrastive', 'contrastive_evaluation']:
            return cts_loss, cts_logits
        elif self.mode == 'classification':
            return cls_loss, cls_logits
        elif self.mode == 'contrastive_and_classification':
            # @@@@ try: initial idea
            # return cts_loss + cls_loss, cls_logits

            # @@@@ try 1:
            nll_features = self.feature_extractors['text'](inputs['nll'])
            _, nll_cts_logits = self.contrastive_step(m1_features, nll_features, calc_loss=False)

            # @@@ try 1.1:
            # cls_probs = torch.softmax(cls_logits, dim=1)
            # nll_cts_probs = torch.softmax(nll_cts_logits, dim=1)
            # expert_guide_loss = mse_loss(cls_probs, nll_cts_probs)

            # @@@ try 1.2:
            # # 使用 torch.max 获取两个 tensor 每个位置的最大值
            # ensemble_logits, _ = torch.max(torch.stack((cls_logits, nll_cts_logits)), dim=0)

            if self.ensemble_model is not None:
                ensemble_loss, ensemble_logits = self.ensemble_model(cls_logits, nll_cts_logits, inputs['labels'])
                return cts_loss + cls_loss + ensemble_loss, ensemble_logits
                # return cts_loss + ensemble_loss, ensemble_logits
            elif self.config.clncp_ensemble_method == 'no_ensemble':
                return cts_loss + cls_loss , cls_logits
            elif self.config.clncp_ensemble_method == 'only_nll_pair_match':
                return cts_loss, nll_cts_logits

    def contrastive_step(self, m1_features, m2_features, calc_loss=True):
        m1_features = normalize_features(m1_features)
        m2_features = normalize_features(m2_features)
        logit_scale = self.logit_scale.exp()
        logits_per_m1 = torch.matmul(m1_features, m2_features.t()) * logit_scale
        logits_per_m2 = logits_per_m1.t()
        loss = clip_loss(logits_per_m2) if self.mode != 'contrastive_evaluation' and calc_loss else torch.tensor(0.)
        logits = logits_per_m1.softmax(dim=1)
        return loss, logits


def normalize_features(features):
    return features / features.norm(p=2, dim=-1, keepdim=True)
