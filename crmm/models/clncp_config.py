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
                 text_model_cache_dir=None,
                 text_model_name=None,
                 text_model_local_files_only=None,
                 load_hf_pretrained=True,
                 freeze_text_params=False,
                 pretrained=False,
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
        self.text_model_cache_dir = text_model_cache_dir
        self.text_model_name = text_model_name
        self.text_model_local_files_only = text_model_local_files_only
        self.load_hf_pretrained = load_hf_pretrained
        self.freeze_text_params = freeze_text_params
        self.pretrained = pretrained