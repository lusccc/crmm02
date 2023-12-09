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
from crmm.arguments import LanguageModelArguments, MultimodalDataArguments, CrmmTrainingArguments
from crmm.dataset.multimodal_data import MultimodalData
from crmm.dataset.multimodal_dataset import MultimodalNormalCollator, MultimodalClipPairMatchCollator
from crmm.metrics import calc_classification_metrics
from crmm.models.clncp import CLNCPConfig, CLNCP
from crmm.models.conventional_multimodal import CMMConfig, ConventionalMultimodalClassification

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


def main(language_model_args: LanguageModelArguments,
         data_args: MultimodalDataArguments,
         training_args: CrmmTrainingArguments):
    training_args.seed = seed
    task = training_args.task

    # @@@@ 2. DATASET
    mm_data = MultimodalData(data_args)
    train_dataset, test_dataset, val_dataset = mm_data.get_datasets()
    n_labels = len(mm_data.label_values)
    if task == 'pretrain':
        # if pretrain task, concat train and val to unsupervised train!
        train_dataset = ConcatDataset([train_dataset, val_dataset])
        val_dataset = None

    # @@@@ 3. MODEL
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=language_model_args.language_model_name,
                                              cache_dir=language_model_args.cache_dir,
                                              local_files_only=language_model_args.load_hf_model_from_cache)
    default_model_config_params = dict(n_labels=n_labels,
                                       num_feat_dim=test_dataset.numerical_feats.shape[1],
                                       use_modality=training_args.use_modality,
                                       fuse_modality=training_args.fuse_modality,
                                       modality_fusion_method=training_args.modality_fusion_method,
                                       max_text_length=language_model_args.max_seq_length,
                                       loss_weights=training_args.loss_weights,
                                       language_model_cache_dir=language_model_args.cache_dir,
                                       language_model_name=language_model_args.language_model_name,
                                       language_model_local_files_only=language_model_args.load_hf_model_from_cache,
                                       load_hf_pretrained=True)
    if task == 'pretrain':
        model_config = CLNCPConfig(**default_model_config_params,
                                   freeze_language_model_params=language_model_args.freeze_language_model_params,
                                   mode='pretrain',
                                   contrastive_targets=training_args.contrastive_targets)

        model = CLNCP(model_config) if training_args.pretrained_model_dir is None else \
            CLNCP.from_pretrained(training_args.pretrained_model_dir, config=model_config)
    elif task == 'continue_pretrain':
        model_config = CLNCPConfig.from_pretrained(training_args.pretrained_model_dir)
        model = CLNCP.from_pretrained(training_args.pretrained_model_dir, config=model_config)
    elif task in ['clip_pair_match_prediction', 'finetune_for_classification']:
        model_config = CLNCPConfig.from_pretrained(training_args.pretrained_model_dir)
        # note: freeze_language_model_params is overridden by training_args.freeze_language_model_params in run command!
        model_config.freeze_language_model_params = language_model_args.freeze_language_model_params
        model_config.mode = 'pair_match_prediction' if task == 'clip_pair_match_prediction' else 'finetune'
        model = CLNCP.from_pretrained(training_args.pretrained_model_dir, config=model_config)
    elif task == 'finetune_classification_evaluate':
        model_config = CLNCPConfig.from_pretrained(training_args.finetuned_model_dir)
        model = CLNCP.from_pretrained(training_args.finetuned_model_dir, config=model_config)
    elif task in ['finetune_for_classification_from_scratch', 'clip_multi_task_classification']:
        model_config = CLNCPConfig(**default_model_config_params,
                                   freeze_language_model_params=language_model_args.freeze_language_model_params,
                                   mode='finetune')
        if 'multi_task' in task:
            model_config.mode = 'multi_task'
            model = CLNCP(model_config) if training_args.pretrained_model_dir is None else \
                CLNCP.from_pretrained(training_args.pretrained_model_dir, config=model_config)
        else:
            model = CLNCP(model_config)
    elif task == 'zero_shot_prediction_from_another_dataset_pretraining':
        # load model pretrained on another dataset
        pretrained_model_config = CLNCPConfig.from_pretrained(training_args.pretrained_model_dir)
        pretrained_model = CLNCP.from_pretrained(training_args.pretrained_model_dir, config=pretrained_model_config)
        # create model for current dataset
        model_config = CLNCPConfig(**default_model_config_params,
                                   freeze_language_model_params=language_model_args.freeze_language_model_params,
                                   pretrained=False)
        model = CLNCP(model_config)
        model.pretrained = True  # should manually set to pretrained
        # use pretrained clip text weights
        model.feature_extractors['text'] = pretrained_model.feature_extractors['text']
    elif task in ['conventional_multimodal_classification', 'no_clip_classification']:
        model_config = CMMConfig(**default_model_config_params)
        model = ConventionalMultimodalClassification(model_config)
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
        compute_metrics=calc_classification_metrics if 'prediction' in task or 'classification' in task else None,
        callbacks=[EarlyStoppingCallback(training_args.patience)] if 'classification' in task else None,
        data_collator=MultimodalNormalCollator(tokenizer, language_model_args.max_seq_length)
        if task in
           ['pretrain',
            'continue_pretrain',
            'clip_multi_task_classification',
            'finetune_for_classification',
            'finetune_for_classification_from_scratch',
            'no_clip_classification']
        else MultimodalClipPairMatchCollator(tokenizer,
                                             language_model_args.max_seq_length,
                                             data_args.natural_language_labels),
    )
    trainer = get_trainer(model)
    if 'prediction' not in task and not training_args.only_eval:
        trainer.train()

    if task in ['pretrain', 'continue_pretrain']:
        # should set pretrained to True. Hence, we can load in later tasks.
        model.config.pretrained = True

    trainer.save_model()  # manually save at end

    best_model_checkpoint = trainer.state.best_model_checkpoint
    best_step = int(best_model_checkpoint.split("-")[-1]) if best_model_checkpoint else None
    total_step = trainer.state.max_steps
    logger.info(f"Best model path: {best_model_checkpoint}")
    logger.info(f"Best metric value: {trainer.state.best_metric}")

    # @@@@ 5. EVALUATION
    if 'prediction' in task or 'classification' in task:
        val_best_results = trainer.evaluate(eval_dataset=val_dataset)
        test_results = trainer.evaluate(eval_dataset=test_dataset)
        tasks_need_pretrain = ['clip_pair_match_prediction', 'finetune_for_classification']
        if task in tasks_need_pretrain + ['zero_shot_prediction_from_another_dataset_pretraining']:
            with open(os.path.join(training_args.pretrained_model_dir, 'training_arguments.json')) as file:
                training_arguments = json.load(file)

            ckpt_dir = glob.glob(f'{training_args.pretrained_model_dir}/*/')[0]
            with open(os.path.join(ckpt_dir, 'trainer_state.json')) as file:
                trainer_state = json.load(file)

            pretrain_final_step = trainer_state['global_step']
            pretrain_max_steps = trainer_state['max_steps']
            pretrain_epoch = training_arguments['num_train_epochs']
            pretrain_model_final_ckpt_epoch = pretrain_final_step / pretrain_max_steps * pretrain_epoch
            pretrain_steps_per_epoch = pretrain_max_steps / pretrain_epoch

        basic_info = {
            'dataset': data_args.dataset_name,
            'dataset_info': data_args.dataset_info,
            'data_path': data_args.data_path,
            'task': task,
            'language_model': language_model_args.language_model_name,
            'freeze_language_model_params': model_config.freeze_language_model_params,
            'numerical': 'num' in model_config.use_modality,
            'category': 'cat' in model_config.use_modality,
            'text': 'text' in model_config.use_modality,
            'modality_fusion_method': model_config.modality_fusion_method,
            'this_exp_path': training_args.output_dir,
            'natural_language_labels': data_args.natural_language_labels if task == 'clip_pair_match_prediction' else None,
            'pretrain_model_path': training_args.pretrained_model_dir if task in tasks_need_pretrain else None,
            'pretrain_epoch': pretrain_epoch if task in tasks_need_pretrain else None,
            'pretrain_model_final_ckpt_epoch': pretrain_model_final_ckpt_epoch if task in tasks_need_pretrain else None,
            'pretrain_steps_per_epoch': pretrain_steps_per_epoch if task in tasks_need_pretrain else None,
            'classification_train_best_step': None if task == 'clip_pair_match_prediction' else best_step,
            'classification_train_total_step': None if task == 'clip_pair_match_prediction' else total_step,
            'classification_train_batch_size': None if task == 'clip_pair_match_prediction' else training_args.per_device_train_batch_size,
            'classification_train_epoch': None if task == 'clip_pair_match_prediction' else training_args.num_train_epochs,
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
    parser = HfArgumentParser([LanguageModelArguments, MultimodalDataArguments, CrmmTrainingArguments])
    _bert_model_args, _data_args, _training_args = parser.parse_args_into_dataclasses()
    _training_args = runner_setup.setup(_training_args, _data_args, _bert_model_args)
    logger.info(f'training_args: {_training_args}')

    main(_bert_model_args, _data_args, _training_args)
