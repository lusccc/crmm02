from .clip_text_feat import TextFeatureExtractor
from .clncp import CLNCPConfig
from .emb_cat_feat import CatFeatureExtractor
from .joint_feat import JointFeatureExtractor
from .resnet_num_feat import NumFeatureExtractor


class FeatureExtractorFactory:

    def __init__(self, clncp_config: CLNCPConfig):
        self.clncp_config = clncp_config
        self.feature_extractors = None

    def get_feature_extractors(self):
        self.feature_extractors = {
            m: self._create_feature_extractor_for(m, dropout=.1)
            for m in self.clncp_config.use_modality
        }
        if 'num' in self.clncp_config.use_modality and 'cat' in self.clncp_config.use_modality:
            modality_feat_dims = {
                m: self.feature_extractors[m].get_output_dim()
                for m in self.clncp_config.use_modality
            }
            self.feature_extractors['joint'] = JointFeatureExtractor(
                modality_feat_dims=modality_feat_dims,
                hidden_dims=[512, 512],
                dropout=.1,
                modality_fusion_method=self.clncp_config.modality_fusion_method
            )

        return self.feature_extractors

    def _create_feature_extractor_for(self, modality, dropout=.3):
        if modality == 'num':
            feature_extractor = NumFeatureExtractor(
                input_dim=self.clncp_config.num_feat_dim,
                hidden_dims=[512, 512, 512],  # hidden size in res block
                dropout=dropout
            )
        elif modality == 'cat':
            feature_extractor = CatFeatureExtractor(
                num_embeddings=self.clncp_config.nunique_cat_nums,
                embedding_dims=self.clncp_config.cat_emb_dims,
                hidden_dim=max(self.clncp_config.cat_emb_dims),
                dropout=dropout
            )
        elif modality == 'text':
            feature_extractor = TextFeatureExtractor(
                max_seq_length=self.clncp_config.max_text_length,
                freeze_clip_params=self.clncp_config.freeze_clip_text_params
            )
        else:
            raise ValueError(f"Invalid modality: {modality}")
        return feature_extractor

    def get_rbm_output_dim_for_classification(self):
        if len(self.clncp_config.use_modality) == 1:
            return self.feature_extractors[self.clncp_config.use_modality[0]].get_output_dim()
        elif len(self.mm_model_config.use_modality) > 1:
            return self.rbms['joint'].encoder.get_output_dim()
        else:
            raise ValueError(f'number of modality {len(self.mm_model_config.use_modality)} not supported')
