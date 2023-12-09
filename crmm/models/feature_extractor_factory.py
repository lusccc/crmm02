from .cat_feature_extractor import CatFeatureExtractor
from .language_model import LanguageModel
from .text_feature_extractor import TextFeatureExtractor
from .clncp import CLNCPConfig
from .joint_feature_extractor import JointFeatureExtractor
from .num_feature_extractor import NumFeatureExtractor


class FeatureExtractorFactory:

    def __init__(self, clncp_config: CLNCPConfig):
        self.clncp_config = clncp_config
        self.feature_extractors = None

    def get_feature_extractors(self, dropout=.2):
        use_modality = self.clncp_config.use_modality
        self.feature_extractors = {}
        if 'num' in use_modality:
            self.feature_extractors['num'] = NumFeatureExtractor(
                input_dim=self.clncp_config.num_feat_dim,
                hidden_dims=[768, 768, 768],  # hidden size in res block
                dropout=dropout
            )

        language_model = LanguageModel(
            pretrained_model_name_or_path=self.clncp_config.language_model_name,
            load_hf_pretrained=self.clncp_config.load_hf_pretrained,
            local_files_only=self.clncp_config.language_model_local_files_only,
            cache_dir=self.clncp_config.language_model_cache_dir,
            max_seq_length=self.clncp_config.max_text_length,
            freeze_params=self.clncp_config.freeze_language_model_params
        )

        if 'cat' in use_modality or 'text' in use_modality:
            if 'cat' in use_modality:
                self.feature_extractors['cat'] = CatFeatureExtractor(language_model)
            if 'text' in use_modality:
                self.feature_extractors['text'] = TextFeatureExtractor(language_model)

        fuse_modality = self.clncp_config.fuse_modality
        if isinstance(fuse_modality, list) and len(fuse_modality) > 1:
            modality_feat_dims = {
                m: self.feature_extractors[m].get_output_dim()
                for m in fuse_modality
            }
            joint_feature_extractor = JointFeatureExtractor(
                modality_feat_dims=modality_feat_dims,
                hidden_dims=[512, 512],
                dropout=dropout,
                modality_fusion_method=self.clncp_config.modality_fusion_method
            )
            self.feature_extractors['joint'] = joint_feature_extractor

        return self.feature_extractors
