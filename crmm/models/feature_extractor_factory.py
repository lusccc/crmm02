from .cat_feature_extractor import CatFeatureExtractor
from .clncp import CLNCPConfig
from .joint_feature_extractor import JointFeatureExtractor
from .num_cat_language_model import NumCatLanguageModel
from .num_feature_extractor import NumFeatureExtractor
from .text_feature_extractor import TextFeatureExtractor
from .text_language_model import TextLanguageModel


class FeatureExtractorFactory:

    def __init__(self, clncp_config: CLNCPConfig):
        self.clncp_config = clncp_config

    def _create_num_feature_extractor(self, semantic_language_model):
        # @@@ try at beginning:
        # return NumFeatureExtractor(
        #     input_dim=self.clncp_config.num_feat_dim,
        #     hidden_dims=[768, 768, 768],  # hidden size in res block
        #     dropout=.2
        # )

        return NumFeatureExtractor(self.clncp_config.num_feat_dim, semantic_language_model)

    def _create_semantic_language_model(self):
        return TextLanguageModel(
            pretrained_model_name_or_path=self.clncp_config.language_model_name,
            load_hf_pretrained=self.clncp_config.load_hf_pretrained,
            local_files_only=self.clncp_config.language_model_local_files_only,
            cache_dir=self.clncp_config.language_model_cache_dir,
            max_seq_length=self.clncp_config.max_text_length,
            freeze_params=self.clncp_config.freeze_language_model_params
        )

    def _create_text_feature_extractor(self, semantic_language_model):
        return TextFeatureExtractor(semantic_language_model)

    def _create_num_cat_language_model(self):
        return NumCatLanguageModel(self.clncp_config)

    def _create_cat_feature_extractor(self, num_cat_language_model: NumCatLanguageModel):
        return CatFeatureExtractor(num_cat_language_model)

        # try 4 in cat_feature_extractor.py
        # return CatFeatureExtractor(num_cat_language_model.config)

    def _creat_joint_feature_extractor(self, modality_feature_extractors, num_cat_language_model):
        # joint_feature_extractor = JointFeatureExtractor(
        #     modality_feat_dims=modality_feat_dims,
        #     hidden_dims=[512, 512],
        #     dropout=dropout,
        #     modality_fusion_method=self.clncp_config.modality_fusion_method
        # )
        return JointFeatureExtractor(
            language_model=num_cat_language_model
        )

    def auto_create_all(self):
        feature_extractors = {'num': None, 'cat': None, 'text': None, 'joint': None}
        num_cat_language_model = None  # is used for cat, and num&cat joint

        def update_feature_extractors(modalities):
            nonlocal num_cat_language_model
            if 'num' in modalities and feature_extractors['num'] is None:
                if num_cat_language_model is None:
                    num_cat_language_model = self._create_num_cat_language_model()
                feature_extractors['num'] = self._create_num_feature_extractor(num_cat_language_model)
            if 'cat' in modalities and feature_extractors['cat'] is None:
                if num_cat_language_model is None:
                    num_cat_language_model = self._create_num_cat_language_model()
                feature_extractors['cat'] = self._create_cat_feature_extractor(num_cat_language_model)
            if 'text' in modalities and feature_extractors['text'] is None:
                feature_extractors['text'] = self._create_text_feature_extractor(self._create_semantic_language_model())

        def update_joint_feature_extractor():
            nonlocal num_cat_language_model
            if isinstance(self.clncp_config.fuse_modality, list) and len(self.clncp_config.fuse_modality) > 1:
                if num_cat_language_model is None:
                    num_cat_language_model = self._create_num_cat_language_model()
                if feature_extractors['joint'] is None:
                    feature_extractors['joint'] = self._creat_joint_feature_extractor(feature_extractors,
                                                                                      num_cat_language_model)

        if self.clncp_config.mode in ['contrastive', 'contrastive_evaluation', 'contrastive_and_classification']:
            contrastive_targets = set(self.clncp_config.contrastive_targets)
            update_feature_extractors(contrastive_targets)
            if 'joint' in contrastive_targets:
                update_feature_extractors(['num', 'cat'])
                update_joint_feature_extractor()

        if self.clncp_config.mode in ['classification', 'contrastive_and_classification']:
            update_feature_extractors(self.clncp_config.use_modality)
            update_joint_feature_extractor()

        if self.clncp_config.mode not in ['contrastive', 'contrastive_evaluation', 'classification',
                                          'contrastive_and_classification']:
            raise ValueError('Unrecognized')

        return feature_extractors

    # def get_feature_extractors(self, dropout=.2):
    #     use_modality = self.clncp_config.use_modality
    #     self.feature_extractors = {}
    #     if 'num' in use_modality:
    #         self.feature_extractors['num'] = NumFeatureExtractor(
    #             input_dim=self.clncp_config.num_feat_dim,
    #             hidden_dims=[768, 768, 768],  # hidden size in res block
    #             dropout=dropout
    #         )
    #
    #     semantic_language_model = TextLanguageModel(
    #         pretrained_model_name_or_path=self.clncp_config.language_model_name,
    #         load_hf_pretrained=self.clncp_config.load_hf_pretrained,
    #         local_files_only=self.clncp_config.language_model_local_files_only,
    #         cache_dir=self.clncp_config.language_model_cache_dir,
    #         max_seq_length=self.clncp_config.max_text_length,
    #         freeze_params=self.clncp_config.freeze_language_model_params
    #     ) if self.clncp_config.mode in ['pretrain', 'multi_task'] else None
    #
    #     num_cat_language_model = NumCatLanguageModel() \
    #         if self.clncp_config.mode in ['finetune_for_classification', 'multi_task'] else None
    #
    #     if 'cat' in use_modality or 'text' in use_modality:
    #         if 'cat' in use_modality:
    #             # self.feature_extractors['cat'] = CatFeatureExtractor(semantic_language_model)
    #             self.feature_extractors['cat'] = CatFeatureExtractor(num_cat_language_model)
    #         if 'text' in use_modality and self.clncp_config.mode in ['pretrain', 'multi_task']:
    #             self.feature_extractors['text'] = TextFeatureExtractor(semantic_language_model)
    #
    #     fuse_modality = self.clncp_config.fuse_modality
    #     if isinstance(fuse_modality, list) and len(fuse_modality) > 1:
    #         modality_feat_dims = {
    #             m: self.feature_extractors[m].get_output_dim()
    #             for m in fuse_modality
    #         }
    #         # joint_feature_extractor = JointFeatureExtractor(
    #         #     modality_feat_dims=modality_feat_dims,
    #         #     hidden_dims=[512, 512],
    #         #     dropout=dropout,
    #         #     modality_fusion_method=self.clncp_config.modality_fusion_method
    #         # )
    #         joint_feature_extractor = JointFeatureExtractor(
    #             modality_feat_dims=modality_feat_dims,
    #             language_model=num_cat_language_model
    #         )
    #         self.feature_extractors['joint'] = joint_feature_extractor
    #
    #     return self.feature_extractors
