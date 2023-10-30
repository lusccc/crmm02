import glob
import json
import os
import random
import re
from datetime import datetime

import numpy as np
import openpyxl
import pandas as pd
import torch
import torchinfo
from torch.utils.data import ConcatDataset
from transformers import AutoTokenizer, Trainer, EarlyStoppingCallback, HfArgumentParser
from transformers.utils import logging

from crmm import runner_setup
from crmm.arguments import CLIPTextModelArguments, MultimodalDataArguments, CrmmTrainingArguments
from crmm.dataset.multimodal_data import MultimodalData
from crmm.dataset.multimodal_dataset import MultimodalPretrainCollator, MultimodalClassificationCollator
from crmm.metrics import calc_classification_metrics
from crmm.models.clncp import CLNCPConfig, CLNCP

logger = logging.get_logger('transformers')

os.environ['COMET_MODE'] = 'DISABLED'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["WANDB_DISABLED"] = "true"

# 设置随机种子
seed = 3407
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def main(clip_text_model_args: CLIPTextModelArguments,
         data_args: MultimodalDataArguments,
         training_args: CrmmTrainingArguments):
    training_args.seed = seed
    task = training_args.task
    # @@@@ 1. TABULAR COLUMNS
    # note: the num and cat cols will be automatically inferred in `data.crmm_data.MultimodalData`
    label_col = data_args.label_col  # 'Rating'
    text_cols = data_args.text_cols.split(',')  # ['GPT_description']
    label_list = [0, 1]

    # @@@@ 2. DATASET
    mm_data = MultimodalData(data_args,
                             label_col=label_col,
                             label_list=label_list,
                             text_cols=text_cols,
                             num_transform_method=data_args.numerical_transformer_method)
    train_dataset, test_dataset, val_dataset = mm_data.get_datasets()
    n_labels = len(label_list)
    if task == 'pretrain':
        # if pretrain task, concat train and val to unsupervised train!
        train_dataset = ConcatDataset([train_dataset, val_dataset])
        val_dataset = None
    nunique_cat_nums, cat_emb_dims = mm_data.get_nunique_cat_nums_and_emb_dim(equal_dim=None)

    # @@@@ 3. MODEL
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=clip_text_model_args.clip_text_model_name,
                                              cache_dir=clip_text_model_args.cache_dir,
                                              local_files_only=clip_text_model_args.load_hf_model_from_cache)
    default_model_config_params = dict(n_labels=n_labels,
                                       num_feat_dim=test_dataset.numerical_feats.shape[1],
                                       nunique_cat_nums=nunique_cat_nums,
                                       cat_emb_dims=cat_emb_dims,
                                       use_modality=training_args.use_modality,
                                       modality_fusion_method=training_args.modality_fusion_method,
                                       max_text_length=clip_text_model_args.max_seq_length,
                                       clip_text_model_cache_dir=clip_text_model_args.cache_dir,
                                       clip_text_model_name=clip_text_model_args.clip_text_model_name,
                                       clip_text_model_local_files_only=clip_text_model_args.load_hf_model_from_cache)
    if task == 'pretrain':
        model_config = CLNCPConfig(**default_model_config_params,
                                   freeze_clip_text_params=clip_text_model_args.freeze_clip_text_params,
                                   pretrained=False)
        if {'text', 'num', 'cat'} == set(training_args.use_modality):
            model = CLNCP(model_config)
    elif task == 'eval_prediction_after_pretraining':
        # prediction_after_pretraining, model load from saved dir
        model_config = CLNCPConfig.from_pretrained(training_args.pretrained_model_dir)
        model = CLNCP.from_pretrained(training_args.pretrained_model_dir, config=model_config)

    else:
        raise ValueError(f'Unknown task: {task}')
    logger.info(f'\n{model}')
    logger.info(f'\n{torchinfo.summary(model, verbose=0)}')

    # @@@@ 4. TRAINING
    get_trainer = lambda model: Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset if 'prediction' in task else val_dataset,  # can be None in pretrain
        compute_metrics=calc_classification_metrics if 'prediction' in task else None,
        callbacks=[EarlyStoppingCallback(training_args.patience)] if 'fine_tune' in task else None,
        data_collator=MultimodalPretrainCollator(tokenizer, clip_text_model_args.max_seq_length) if
        'pretrain' == task else MultimodalClassificationCollator(tokenizer, clip_text_model_args.max_seq_length,
                                                                 # note bad or good label is split with @
                                                                 data_args.natural_language_labels.split('@')),
    )
    trainer = get_trainer(model)
    if task == 'pretrain':
        trainer.train()
        # should set pretrained to True. Hence, we can load in later tasks.
        model.config.pretrained = True
        trainer.save_model()

    best_model_checkpoint = trainer.state.best_model_checkpoint
    best_step = int(best_model_checkpoint.split("-")[-1]) if best_model_checkpoint else None
    logger.info(f"Best model path: {best_model_checkpoint}")
    logger.info(f"Best metric value: {trainer.state.best_metric}")

    # @@@@ 5. EVALUATION
    if task != 'pretrain':
        val_best_results = trainer.evaluate(eval_dataset=val_dataset)
        test_results = trainer.evaluate(eval_dataset=test_dataset)
        if task == 'eval_prediction_after_pretraining':
            with open(os.path.join(training_args.pretrained_model_dir, 'training_arguments.json')) as file:
                training_arguments = json.load(file)
            pretrain_batch_size = training_arguments['per_device_train_batch_size']
            pretrain_epoch = training_arguments['num_train_epochs']
            pretrain_best_step = int(
                # to get dir name like 'checkpoint-2500' to infer best step
                re.search('checkpoint-(\d+)', glob.glob(f'{training_args.pretrained_model_dir}/*/')[0]).group(1))
            pretrain_best_epoch = pretrain_best_step / pretrain_batch_size

        basic_info = {
            'dataset': data_args.dataset_name,
            'dataset_info': data_args.dataset_info,
            'data_path': data_args.data_path,
            'clip_text_model': None,  # we train with our credit description data! did not use the openai pretrained one
            'numerical': 'num' in training_args.use_modality,
            'category': 'cat' in training_args.use_modality,
            'text': 'text' in training_args.use_modality,
            'modality_fusion_method': model_config.modality_fusion_method,
            'this_exp_path': training_args.output_dir,
            'natural_language_labels': data_args.natural_language_labels if task == 'eval_prediction_after_pretraining' else None,
            'pretrain_model_path': training_args.pretrained_model_dir if task == 'eval_prediction_after_pretraining' else None,
            'pretrain_batch_size': pretrain_batch_size if task == 'eval_prediction_after_pretraining' else None,
            'pretrain_epoch': pretrain_epoch if task == 'eval_prediction_after_pretraining' else None,
            'pretrain_model_best_epoch': pretrain_best_epoch if task == 'eval_prediction_after_pretraining' else None,
            'classification_train_best_step': None if task == 'eval_prediction_after_pretraining' else best_step,
            'classification_train_batch_size': None if task == 'eval_prediction_after_pretraining' else training_args.per_device_train_batch_size,
            'classification_train_epoch': None if task == 'eval_prediction_after_pretraining' else training_args.num_train_epochs,
        }

        save_excel(val_best_results, test_results, basic_info, training_args.save_excel_path)


def save_excel(val_best_results, test_results, basic_info, excel_path):
    logger.info(f'** save results to {excel_path}')
    val_data = {f'val_{k}': v for k, v in val_best_results.items()}
    test_data = {f'test_{k}': v for k, v in test_results.items()}

    if os.path.exists(excel_path):
        book = openpyxl.load_workbook(excel_path)
        writer = pd.ExcelWriter(excel_path, engine='openpyxl')
        writer._book = book

        if 'Sheet1' in book.sheetnames:
            startrow = writer.sheets['Sheet1'].max_row
        else:
            startrow = 0

    else:
        writer = pd.ExcelWriter(excel_path, engine='openpyxl')
        startrow = 0

    data = {**{'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
            **basic_info, **val_data, **test_data, }
    df = pd.DataFrame([data])

    df.to_excel(writer, index=False, header=(startrow == 0), startrow=startrow)

    writer._save()
    writer.close()


if __name__ == '__main__':
    parser = HfArgumentParser([CLIPTextModelArguments, MultimodalDataArguments, CrmmTrainingArguments])
    _bert_model_args, _data_args, _training_args = parser.parse_args_into_dataclasses()
    _training_args = runner_setup.setup(_training_args, _data_args, _bert_model_args)
    logger.info(f'training_args: {_training_args}')

    main(_bert_model_args, _data_args, _training_args)
