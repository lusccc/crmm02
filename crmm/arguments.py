import os
from dataclasses import dataclass, field
from typing import Optional, Union, List

from transformers import TrainingArguments, IntervalStrategy

os.environ["WANDB_DISABLED"] = "true"
TASK = ['pretrain', 'fine_tune', 'fine_tune_from_scratch']
MODALITY = ['num', 'cat', 'text']


@dataclass
class CrmmTrainingArguments(TrainingArguments):
    # @@@@ 1. our args
    # report_to: str = field(default='wandb')
    root_dir: str = field(default='./exps', metadata={"help": "parent dir of output_dir"})
    pretrained_model_dir: str = field(default='', metadata={"help": ""})
    auto_create_model_dir: bool = field(default=True, metadata={"help": "auto create model dir in root_dir"})
    save_excel_path: str = field(default='./excel/res.xlsx', metadata={"help": ""})
    task: str = field(default="eval_prediction_after_pretraining", metadata={"help": "", })
    # task: str = field(default="pretrain", metadata={"help": "", })
    use_modality: str = field(default="num,cat,text", metadata={"help": ""})
    modality_fusion_method: str = field(default="conv", metadata={"help": ""})
    patience: int = field(default=1000, metadata={"help": ""})
    label_names: Optional[List[str]] = field(
        default=None, metadata={"help": "The list of keys in your dictionary of inputs that correspond to the labels."}
    )
    # @@@@ 2. huggingface args
    # output_dir and logging_dir will be auto set in runner_setup.setup
    output_dir: str = field(default=None, metadata={
        "help": "The output directory where the model predictions and checkpoints will be written."}, )
    logging_dir: Optional[str] = field(default=None, metadata={"help": "Tensorboard log dir."})
    overwrite_output_dir: bool = field(default=True, metadata={
        "help": ("Overwrite the content of the output directory. "
                 "Use this to continue training if output_dir points to a checkpoint directory.")})
    do_train: bool = field(default=True, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=True, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})
    # remove_unused_columns should be False for models.multi_modal_dbn.MultiModalDBN.forward
    remove_unused_columns: Optional[bool] = field(
        default=False, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )
    num_train_epochs: float = field(default=100, metadata={"help": "Total number of training epochs to perform."})
    logging_steps: int = field(default=10, metadata={"help": "Log every X updates steps."})
    no_cuda: bool = field(default=False, metadata={"help": "Do not use CUDA even when it is available"})
    per_device_train_batch_size: int = field(default=200, metadata={
        "help": "Batch size per GPU/TPU core/CPU for training."})
    evaluation_strategy: Union[IntervalStrategy, str] = field(default="epoch", metadata={
        "help": "The evaluation strategy to use."}, )  # metric will be checked in early-stopping
    dataloader_num_workers: int = field(default=16, metadata={
        "help": ("Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded"
                 " in the main process.")})
    save_strategy: Union[IntervalStrategy, str] = field(default="epoch", metadata={
        "help": "The checkpoint save strategy to use."})
    load_best_model_at_end: Optional[bool] = field(default=True, metadata={
        "help": "Whether or not to load the best model found during training at the end of training."})
    save_total_limit: Optional[int] = field(default=1, metadata={
        "help": ("Limit the total amount of checkpoints. "
                 "Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints")})
    auto_find_batch_size: bool = field(default=True, metadata={
        "help": ("Whether to automatically decrease the batch size in half and rerun the training loop again each time"
                 " a CUDA Out-of-Memory was reached")})
    # metric_for_best_model: str = field(default='acc')  # used for early stopping;
    # greater_is_better: bool = field(default=False)  # used for early stopping
    fp16: bool = field(
        default=not no_cuda,
        metadata={"help": "Whether to use fp16 (mixed) precision instead of 32-bit"},
    )

    # PyTorch 2.0 specifics
    # bf16: bool = field(default=True, metadata={})
    # torch_compile: bool = field(default=False, metadata={})
    # optim: str = field(default='adamw_torch_fused', metadata={})

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.use_modality, str):
            self.use_modality = [m.strip() for m in self.use_modality.split(',')]

        if self.task == 'pretrain':
            self.metric_for_best_model = 'loss'
            self.greater_is_better = False
            self.evaluation_strategy = 'no'
        else:
            self.metric_for_best_model = 'acc'
            self.greater_is_better = True

        if self.output_dir is not None and self.logging_dir is not None:
            self.auto_create_model_dir = False

        # TODO  below I make compatible with code `has_labels = False if len(self.label_names) == 0
        #  else all(inputs.get(k) is not None for k in self.label_names)` in
        #  transformers.trainer.Trainer.prediction_step, but why?
        self.label_names = ['labels']


@dataclass
class CLIPTextModelArguments:
    # actually only used for tokenizer!
    clip_text_model_name: str = field(default='openai/clip-vit-base-patch32', metadata={
        "help": "Path to pretrained model or model identifier from huggingface.co/models"})
    freeze_clip_text_params: bool = field(default=True, metadata={
        "help": "Whether to freeze the clip text model parameters"})
    max_seq_length: int = field(default=512, metadata={
        "help": "The maximum length (in number of tokens) for the inputs to the CLIP text model"})
    # actually only used for tokenizer!
    cache_dir: Optional[str] = field(default='./exps/model_config_cache', metadata={
        "help": "Where do you want to store the pretrained models downloaded from s3"})
    # actually only used for tokenizer!
    load_hf_model_from_cache: bool = field(default=True, )


@dataclass
class MultimodalDataArguments:
    dataset_name: str = field(default='cr', metadata={"help": ""})
    dataset_info: str = field(default='', metadata={"help": ""})
    data_path: str = field(default=f'./data/cr', metadata={
        'help': 'the path to the csv file containing the dataset'})
    use_val: bool = field(default=True)
    feature_transform_res_dir: str = field(default=None, metadata={
        "help": "the path to the directory containing the feature transformation results from Autogluon"})
    label_col: str = field(default='binaryRating', metadata={
        'help': 'the name of the label column'})
    text_cols: str = field(default='GPT_description', metadata={
        'help': 'the name of the text column'})
    numerical_transformer_method: str = field(default='yeo_johnson', metadata={
        'help': 'sklearn numerical transformer to preprocess numerical data',
        'choices': ['yeo_johnson', 'box_cox', 'quantile_normal', 'standard', 'none']})
    natural_language_labels: str = field(
        # default='Financially struggling company, credit-challenged enterprise, struggling business with weak credit, '
        #         'company with a poor credit rating, underperforming business with credit issues, '
        #         'enterprise with subpar creditworthiness, company facing credit difficulties,'
        #         ' business with a tarnished credit history, struggling company with credit limitations,'
        #         ' enterprise experiencing credit challenges. @ Financially sound company, '
        #         'creditworthy enterprise, strong business with good credit, company with a solid credit rating, '
        #         'profitable business with strong credit, enterprise with excellent creditworthiness, '
        #         'company with a clean credit history, successful company with strong credit, '
        #         'business with robust credit standing, enterprise with a solid track record of credit.',
        default='Poor credit@Good credit',
        metadata={
            'help': ''})
