import os
from dataclasses import dataclass, field
from typing import Optional, Union

from transformers import TrainingArguments, IntervalStrategy, SchedulerType

os.environ["WANDB_DISABLED"] = "true"
TASK = ['pretrain', 'fine_tune', 'fine_tune_from_scratch']
MODALITY = ['num', 'cat', 'text']


@dataclass
class CrmmTrainingArguments(TrainingArguments):
    # @@@@ 1. our args
    # report_to: str = field(default='wandb')·
    root_dir: str = field(default='./exps', metadata={"help": "parent dir of output_dir"})
    pretrained_model_dir: str = field(default=None, metadata={"help": ""})
    finetuned_model_dir: str = field(default=None, metadata={"help": "can be load for only evaluation"})
    auto_create_model_dir: bool = field(default=True, metadata={"help": "auto create model dir in root_dir"})
    save_excel_path: str = field(default='./excel/res.xlsx', metadata={"help": ""})
    save_hist_eval_csv_path: str = field(default=None, metadata={"help": ""})
    task: str = field(default=None, metadata={"help": "", })
    use_modality: str = field(default=None, metadata={"help": ""})
    fuse_modality: str = field(default=None, metadata={"help": ""})
    modality_fusion_method: str = field(default=None, metadata={"help": ""})
    contrastive_targets: str = field(default=None, metadata={"help": ""})
    contrastive_early_stopping_epoch: int = field(default=None, metadata={"help": ""})
    patience: int = field(default=1000, metadata={"help": ""})

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
    # remove_unused_columns should be False in our custom model!
    remove_unused_columns: Optional[bool] = field(
        default=False, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )
    num_train_epochs: float = field(default=100, metadata={"help": "Total number of training epochs to perform."})
    logging_steps: int = field(default=10, metadata={"help": "Log every X updates steps."})
    no_cuda: bool = field(default=False, metadata={"help": "Do not use CUDA even when it is available"})
    per_device_train_batch_size: int = field(default=200, metadata={
        "help": "Batch size per GPU/TPU core/CPU for training."})
    evaluation_strategy: Union[IntervalStrategy, str] = field(default=IntervalStrategy.EPOCH, metadata={
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
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    lr_scheduler_type: Union[SchedulerType, str] = field(
        default="linear",
        metadata={"help": "The scheduler type to use."},
    )
    fp16: bool = field(
        default=not no_cuda,
        metadata={"help": "Whether to use fp16 (mixed) precision instead of 32-bit"},
    )

    # PyTorch 2.0 specifics
    # bf16: bool = field(default=True, metadata={})
    # torch_compile: bool = field(default=False, metadata={})
    optim: str = field(default='adamw_torch_fused' if not no_cuda else 'adamw_hf', metadata={})

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.use_modality, str):
            self.use_modality = [m.strip() for m in self.use_modality.split(',')]
        if isinstance(self.fuse_modality, str):
            self.fuse_modality = [m.strip() for m in self.fuse_modality.split(',')]
        if isinstance(self.contrastive_targets, str):
            self.contrastive_targets = [m.strip() for m in self.contrastive_targets.split(',')]

        if self.task == 'pretrain' or self.task == 'continue_pretrain':
            self.metric_for_best_model = 'loss'
            self.greater_is_better = False
            self.evaluation_strategy = 'no'
        else:
            # # self.metric_for_best_model = 'loss'
            # # self.greater_is_better = False
            # # self.metric_for_best_model = 'acc'

            self.metric_for_best_model = 'acc'
            self.greater_is_better = True

            # @@@ try:
            if self.save_hist_eval_csv_path is not None:
                self.metric_for_best_model = 'loss'
                self.greater_is_better = False
                self.evaluation_strategy = 'no'

        if self.output_dir is not None and self.logging_dir is not None:
            self.auto_create_model_dir = False

        # TODO  below I make compatible with code `has_labels = False if len(self.label_names) == 0
        #  else all(inputs.get(k) is not None for k in self.label_names)` in
        #  transformers.trainer.Trainer.prediction_step, but why?
        self.label_names = ['labels']


@dataclass
class LanguageModelArguments:
    language_model_name: str = field(default='', metadata={
        "help": "Path to pretrained model or model identifier from huggingface.co/models"})
    freeze_language_model_params: bool = field(default=True, metadata={
        "help": "Whether to freeze the clip text model parameters"})
    max_seq_length: int = field(default=512, metadata={
        "help": "The maximum length (in number of tokens) for the inputs to the CLIP text model"})
    cache_dir: Optional[str] = field(default='./exps/model_config_cache', metadata={
        "help": "Where do you want to store the pretrained models downloaded from s3"})
    load_hf_model_from_cache: bool = field(default=True, )
    load_hf_pretrained: bool = field(default=True, )


@dataclass
class MultimodalDataArguments:
    dataset_name: str = field(default=None, metadata={"help": ""})
    dataset_info: str = field(default='', metadata={"help": ""})
    data_path: str = field(default=None, metadata={
        'help': 'the path to the csv file containing the dataset'})
    dataset_split_strategy: str = field(default='random', metadata={"help": ""})
    train_years: str = field(default=None, metadata={"help": ""})
    test_years: str = field(default=None, metadata={"help": ""})
    num_train_samples: int = field(default=None, metadata={
        "help": "if not specified, for random will be 8:2 split,"
                " for rolling window will be the number of samples in the train window"})
    use_val: bool = field(default=True)
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

    def __post_init__(self):
        if isinstance(self.train_years, str):
            self.train_years = [int(m.strip()) for m in self.train_years.split(',')]
        if isinstance(self.test_years, str):
            self.test_years = [int(m.strip()) for m in self.test_years.split(',')]
        if isinstance(self.natural_language_labels, str):
            self.natural_language_labels = [m.strip() for m in self.natural_language_labels.split('@')]
