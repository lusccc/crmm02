from transformers import PretrainedConfig


class CLNCPConfig(PretrainedConfig):

    def __init__(self,
                 n_labels=0,
                 num_feat_dim=0,
                 nunique_cat_nums=0,
                 cat_emb_dims=0,
                 use_modality=None,
                 modality_fusion_method=None,
                 max_text_length=None,
                 clip_text_model_cache_dir=None,
                 clip_text_model_name=None,
                 clip_text_model_local_files_only=None,
                 use_hf_pretrained_clip_text=False,
                 freeze_clip_text_params=False,
                 pretrained=False,
                 **kwargs):
        super().__init__(**kwargs)
        # unique label count
        self.n_labels = n_labels
        # numerical feature dimension after num_transformer in dataset.crmm_data.MultimodalData.transform_features
        self.num_feat_dim = num_feat_dim
        # nunqiue of each cat col
        self.nunique_cat_nums = nunique_cat_nums
        # category feature embedding dimension of each cat col,
        # used in Embedding layer in models.emb_cat_feat.CatFeatureExtractor
        self.cat_emb_dims = cat_emb_dims
        self.use_modality = use_modality
        self.modality_fusion_method = modality_fusion_method
        self.max_text_length = max_text_length
        self.clip_text_model_cache_dir = clip_text_model_cache_dir
        self.clip_text_model_name = clip_text_model_name
        self.clip_text_model_local_files_only = clip_text_model_local_files_only
        self.freeze_clip_text_params = freeze_clip_text_params
        self.pretrained = pretrained