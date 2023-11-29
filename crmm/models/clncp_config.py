from transformers import PretrainedConfig


class CLNCPConfig(PretrainedConfig):

    def __init__(self,
                 n_labels=0,
                 num_feat_dim=0,
                 use_modality=None,
                 fuse_modality=None,
                 modality_fusion_method=None,
                 max_text_length=None,
                 loss_weights=None,
                 language_model_cache_dir=None,
                 language_model_name=None,
                 language_model_local_files_only=None,
                 load_hf_pretrained=True,
                 freeze_language_model_params=False,
                 mode=None,
                 contrastive_targets=None,
                 **kwargs):
        super().__init__(**kwargs)
        # unique label count
        self.n_labels = n_labels
        # numerical feature dimension after num_transformer in dataset.crmm_data.MultimodalData.transform_features
        self.num_feat_dim = num_feat_dim
        self.use_modality = use_modality
        self.fuse_modality = fuse_modality
        self.modality_fusion_method = modality_fusion_method
        self.loss_weights = loss_weights
        self.max_text_length = max_text_length
        self.language_model_cache_dir = language_model_cache_dir
        self.language_model_name = language_model_name
        self.language_model_local_files_only = language_model_local_files_only
        self.load_hf_pretrained = load_hf_pretrained
        self.freeze_language_model_params = freeze_language_model_params
        self.mode = mode
        self.contrastive_targets=contrastive_targets
