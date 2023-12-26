import csv
import os

import pandas as pd
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.utils import logging

logger = logging.get_logger('transformers')


class TrainerLoggerCallback(TrainerCallback):

    def on_log(self, args, state, control, logs=None, **kwargs):
        # A bare [`TrainerCallback`] that just prints the logs.
        _ = logs.pop("total_flos", None)
        # if state.is_local_process_zero:
        #     logger.info(logs)


class CrmmTrainerCallback(TrainerCallback):

    def __init__(self, runner) -> None:
        self.runner = runner

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        super().on_epoch_end(args, state, control, **kwargs)
        self.runner.predict_on_test()


class ContrastiveEarlyStoppingCallBack(TrainerCallback):
    def __init__(self, epoch_to_trigger):
        self.epoch_to_trigger = epoch_to_trigger

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.epoch_to_trigger is not None and state.epoch == self.epoch_to_trigger:
            model = kwargs['model']  # 获取Trainer实例
            model.stop_contrastive = True
            logger.info(f"stop contrastive after epoch {state.epoch}!")


class EvaluateEveryNthEpoch(TrainerCallback):
    def __init__(self, eval_dataset, test_dataset, n_epochs, csv_path):
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.n_epochs = n_epochs
        self.csv_path = csv_path
        self.trainer = None

    def write_metrics_to_csv(self, val_metrics, test_metrics, epoch):
        # 检查文件是否存在，如果不存在则创建并添加标题行
        file_exists = os.path.isfile(self.csv_path)
        with open(self.csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                # 如果是新创建的文件，写入标题行
                headers = ['epoch'] + [f'val_{key}' for key in val_metrics.keys()] + [f'test_{key}' for key in test_metrics.keys()]
                writer.writerow(headers)
            # 写入数据行
            row = [epoch] + list(val_metrics.values()) + list(test_metrics.values())
            writer.writerow(row)

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch and int(state.epoch) % self.n_epochs == 0:
            # 评估验证集
            val_metrics = self.trainer.evaluate(eval_dataset=self.eval_dataset)
            logger.info(f"Evaluation on validation set after {state.epoch} epochs:")
            logger.info(val_metrics)

            # 评估测试集
            test_metrics = self.trainer.evaluate(eval_dataset=self.test_dataset)
            logger.info(f"Evaluation on test set after {state.epoch} epochs:")
            logger.info(test_metrics)

            # 将验证集和测试集的结果写入同一行
            self.write_metrics_to_csv(val_metrics, test_metrics, state.epoch)
