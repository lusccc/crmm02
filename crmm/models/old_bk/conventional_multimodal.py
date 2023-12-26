from torch import nn
from transformers import PreTrainedModel

from crmm.models.clncp_config import CLNCPConfig
from crmm.models.feature_extractor_factory import FeatureExtractorFactory
from crmm.models.layer_utils import hf_loss_func, create_classifier


class CMMConfig(CLNCPConfig):

    def __init__(self, n_labels=0, num_feat_dim=0, use_modality=None, fuse_modality=None, modality_fusion_method=None,
                 max_text_length=None, loss_weights=None, language_model_cache_dir=None, language_model_name=None,
                 language_model_local_files_only=None, load_hf_pretrained=True, freeze_language_model_params=False,
                 pretrained=False, multitask=False, **kwargs):
        super().__init__(n_labels, num_feat_dim, use_modality, fuse_modality, modality_fusion_method, max_text_length,
                         loss_weights, language_model_cache_dir, language_model_name, language_model_local_files_only,
                         load_hf_pretrained, freeze_language_model_params, pretrained, multitask, **kwargs)


class CMMPreTrainedModel(PreTrainedModel):
    config_class = CMMConfig
    base_model_prefix = "CMM"


class ConventionalMultimodalClassification(CMMPreTrainedModel):
    config_class = CMMConfig

    def __init__(self, config: CMMConfig):
        super().__init__(config)
        self.config = config
        self.n_labels = self.config.n_labels

        self.feature_extractors_factory = FeatureExtractorFactory(clncp_config=self.config)
        self.feature_extractors = self.feature_extractors_factory.get_feature_extractors()
        self.feature_extractors = nn.ModuleDict(self.feature_extractors)

        self.classifier = create_classifier(input_dim=self.feature_extractors['joint'].get_output_dim(),
                                            n_class=self.n_labels)

    def forward(self, return_loss=True, **inputs, ):
        num_features = self.feature_extractors['num'](inputs['num'])
        cat_features = self.feature_extractors['cat'](inputs['cat'])
        joint_features = self.feature_extractors['joint']({'num': num_features, 'cat': cat_features})
        loss, logits, classifier_layer_outputs = hf_loss_func(joint_features,
                                                              self.classifier,
                                                              inputs['labels'],
                                                              self.n_labels,
                                                              None)
        return loss, logits, classifier_layer_outputs
