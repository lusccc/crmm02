import os
import random
import string
import subprocess
from datetime import datetime


def create_exp_dirs(root_dir, task):
    if not os.path.exists(root_dir):
        raise FileNotFoundError(
            f"Root directory '{root_dir}', where the directory of the experiment will be created, must exist")

    formatted_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(root_dir, task)
    rand_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=3))
    output_dir = output_dir + "_" + formatted_timestamp + "_" + rand_suffix
    exp_args_output_dir = os.path.join(output_dir, "output")
    exp_args_logging_dir = os.path.join(output_dir, "logging")

    os.makedirs(exp_args_output_dir, exist_ok=True)
    os.makedirs(exp_args_logging_dir, exist_ok=True)

    return exp_args_output_dir, exp_args_logging_dir


class CommandRunner:
    def __init__(self, root_dir, task, per_device_train_batch_size, num_train_epochs, data_path, dataset_name,
                 dataset_split_strategy, modality_fusion_method, freeze_language_model_params, use_modality,
                 fuse_modality, language_model_name, load_hf_model_from_cache, contrastive_targets=None,
                 pretrained_model_dir=None, save_excel_path=None, natural_language_labels=None,
                 num_train_samples=None, train_years=None, test_years=None, contrastive_early_stopping_epoch=None):
        self.root_dir = root_dir
        self.task = task
        self.per_device_train_batch_size = per_device_train_batch_size
        self.num_train_epochs = num_train_epochs
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.dataset_split_strategy = dataset_split_strategy
        self.modality_fusion_method = modality_fusion_method
        self.freeze_language_model_params = freeze_language_model_params
        self.use_modality = use_modality
        self.fuse_modality = fuse_modality
        self.language_model_name = language_model_name
        self.load_hf_model_from_cache = load_hf_model_from_cache
        self.contrastive_targets = contrastive_targets
        self.pretrained_model_dir = pretrained_model_dir
        self.save_excel_path = save_excel_path
        self.natural_language_labels = natural_language_labels
        self.num_train_samples = num_train_samples
        self.train_years = train_years
        self.test_years = test_years
        self.contrastive_early_stopping_epoch = contrastive_early_stopping_epoch

        exp_dirs = create_exp_dirs(self.root_dir, self.task)
        self.output_dir, self.logging_dir = exp_dirs

    def to_command(self):
        command = (f"python main_runner.py "
                   f"--root_dir {self.root_dir} "
                   f"--output_dir {self.output_dir} "
                   f"--logging_dir {self.logging_dir} "
                   f"--task {self.task} "
                   f"--per_device_train_batch_size {self.per_device_train_batch_size} "
                   f"--num_train_epochs {self.num_train_epochs} "
                   f"--data_path {self.data_path} "
                   f"--dataset_name {self.dataset_name} "
                   f"--dataset_split_strategy {self.dataset_split_strategy} "
                   f"--modality_fusion_method {self.modality_fusion_method} "
                   f"--freeze_language_model_params {self.freeze_language_model_params} "
                   f"--use_modality {self.use_modality} --fuse_modality {self.fuse_modality} "
                   f"--language_model_name {self.language_model_name} "
                   f"--load_hf_model_from_cache {self.load_hf_model_from_cache}")
        if self.contrastive_targets is not None:
            command += f" --contrastive_targets {self.contrastive_targets}"
        if self.pretrained_model_dir is not None:
            command += f" --pretrained_model_dir {self.pretrained_model_dir}"
        if self.save_excel_path is not None:
            command += f" --save_excel_path {self.save_excel_path}"
        if self.natural_language_labels is not None:
            command += f" --natural_language_labels {self.natural_language_labels}"
        if self.num_train_samples is not None:
            command += f" --num_train_samples {self.num_train_samples}"
        if self.train_years is not None:
            command += f" --train_years {self.train_years}"
        if self.test_years is not None:
            command += f" --test_years {self.test_years}"
        if self.contrastive_early_stopping_epoch is not None:
            command += f" --contrastive_early_stopping_epoch {self.contrastive_early_stopping_epoch}"
        return command

    def run(self):
        command = self.to_command()
        print(command)
        process = subprocess.Popen(command, shell=True)
        process.wait()


# def run_pre_epoch(repeat=1):
#     dataset = 'cr'
#     pre_batch_size = 115
#     finetune_batch_size = 240
#     # pre_epochs = [10, 20, 50, 60, 100, 120, 150, 170, 200, 220, 250, 300]
#     for i in range(repeat):
#         pre_epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 30, 40]
#         for epoch in pre_epochs:
#             # @@@@ 1. pretrain
#             pre_command_runner = CommandRunner(
#                 root_dir='./exps',
#                 task='pretrain',
#                 per_device_train_batch_size=str(pre_batch_size),
#                 num_train_epochs=str(epoch),
#                 data_path=f'./data/{dataset}',
#                 dataset_name=dataset,
#                 modality_fusion_method='conv',
#                 freeze_language_model_params='True',
#                 use_modality='num,cat,text',
#                 fuse_modality='num,cat',
#                 language_model_name='mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis',
#                 load_hf_model_from_cache='True',
#                 contrastive_targets='num,text'
#             )
#             pre_command_runner.run()
#
#             # @@@@ 2. finetune
#             finetune_command_runer = CommandRunner(
#                 root_dir='./exps',
#                 task='finetune_for_classification',  # !
#                 per_device_train_batch_size=str(finetune_batch_size),
#                 num_train_epochs=100,  # !
#                 data_path=f'./data/{dataset}',
#                 dataset_name=dataset,
#                 modality_fusion_method='conv',
#                 freeze_language_model_params='False',  # !
#                 use_modality='num,cat,text',
#                 fuse_modality='num,cat',
#                 language_model_name='mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis',
#                 load_hf_model_from_cache='True',
#                 pretrained_model_dir=pre_command_runner.output_dir,  # !
#                 save_excel_path=f'./excel/{dataset}_res_pre_{epoch}_rep_{repeat}.xlsx',
#             )
#             finetune_command_runer.run()
#
#
# def run_scratch(repeat=1):
#     dataset = 'cr'
#     pre_batch_size = 115
#     finetune_batch_size = 350
#     for i in range(repeat):
#         command_runer = CommandRunner(
#             root_dir='./exps',
#             task='finetune_for_classification_from_scratch',  # !
#             per_device_train_batch_size=str(finetune_batch_size),
#             num_train_epochs=100,  # !
#             data_path=f'./data/{dataset}',
#             dataset_name=dataset,
#             modality_fusion_method='conv',
#             freeze_language_model_params='False',  # !
#             use_modality='num,cat,text',
#             fuse_modality='num,cat',
#             language_model_name='mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis',
#             load_hf_model_from_cache='True',
#             save_excel_path=f'./excel/{dataset}_res_scratch_rep_{repeat}.xlsx',
#         )
#         command_runer.run()


def run_pre_fine_rolling_window(repeat=1, only_scratch=False, suffix=''):
    dataset = 'cr'
    pre_batch_size = 150
    finetune_batch_size = 14
    pretrain_epoch = 20
    finetune_epoch = 100
    num_train_samples = None
    train_years_list = [
        [2010, 2011, 2012],
        [2011, 2012, 2013],
        [2012, 2013, 2014],
        [2013, 2014, 2015],
    ]
    test_year_List = [
        [2013],
        [2014],
        [2015],
        [2016],
    ]
    for i in range(repeat):
        for train_years, test_years in zip(train_years_list, test_year_List):
            train_years = ','.join(map(str, train_years))
            test_years = ','.join(map(str, test_years))
            save_excel_path = (f'./excel/{dataset}_res_rolling_window_'
                               f'#{train_years}#_#{test_years}#_'
                               f'scratch_and_'
                               f'pre_{pretrain_epoch}_fine_{finetune_epoch}_rep_{repeat}'
                               f'{suffix}.xlsx') \
                if not only_scratch else (f'./excel/{dataset}_res_rolling_window_'
                                          f'#{train_years}#_#{test_years}#_only_scratch_epoch_{finetune_epoch}'
                                          f'{suffix}.xlsx')
            # ######### scratch ################
            scratch_command_runer = CommandRunner(
                root_dir='./exps',
                task='finetune_for_classification_from_scratch',  # !
                per_device_train_batch_size=str(finetune_batch_size),
                num_train_epochs=finetune_epoch,  # !
                data_path=f'./data/{dataset}',
                dataset_name=dataset,
                dataset_split_strategy='rolling_window',
                modality_fusion_method='conv',
                freeze_language_model_params='False',  # !
                use_modality='num,cat,text',
                fuse_modality='num,cat',
                language_model_name='mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis',
                load_hf_model_from_cache='True',
                train_years=train_years,
                test_years=test_years,
                num_train_samples=num_train_samples,
                save_excel_path=save_excel_path,
            )
            scratch_command_runer.run()

            if only_scratch:
                continue
            # ######### pre fine ################
            # @@@@ 1. pretrain
            pre_command_runner = CommandRunner(
                root_dir='./exps',
                task='pretrain',
                per_device_train_batch_size=str(pre_batch_size),
                num_train_epochs=str(pretrain_epoch),
                data_path=f'./data/{dataset}',
                dataset_name=dataset,
                dataset_split_strategy='rolling_window',
                modality_fusion_method='conv',
                freeze_language_model_params='True',
                use_modality='num,cat,text',
                fuse_modality='num,cat',
                language_model_name='mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis',
                load_hf_model_from_cache='True',
                contrastive_targets='num,text',
                train_years=train_years,
                test_years=test_years,
                num_train_samples=num_train_samples,
            )
            pre_command_runner.run()
            # @@@@ 2. finetune
            finetune_command_runer = CommandRunner(
                root_dir='./exps',
                task='finetune_for_classification',  # !
                per_device_train_batch_size=str(finetune_batch_size),
                num_train_epochs=finetune_epoch,  # !
                data_path=f'./data/{dataset}',
                dataset_name=dataset,
                dataset_split_strategy='rolling_window',
                modality_fusion_method='conv',
                freeze_language_model_params='False',  # !
                use_modality='num,cat,text',
                fuse_modality='num,cat',
                language_model_name='mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis',
                load_hf_model_from_cache='True',
                train_years=train_years,
                test_years=test_years,
                num_train_samples=num_train_samples,
                pretrained_model_dir=pre_command_runner.output_dir,  # !
                save_excel_path=save_excel_path,
            )
            finetune_command_runer.run()


def run_pre_fine_random_split(repeat=1, only_scratch=False, suffix=''):
    dataset = 'cr2'
    pre_batch_size = 300
    finetune_batch_size = 1280
    pretrain_epoch = 20
    finetune_epoch = 100
    num_train_samples = None
    for i in range(repeat):
        save_excel_path = (f'./excel/{dataset}_res_random_split_'
                           f'scratch_and_'
                           f'pre_{pretrain_epoch}_fine_{finetune_epoch}_rep_{repeat}'
                           f'{suffix}.xlsx') \
            if not only_scratch else (f'./excel/{dataset}_res_random_split_'
                                      f'only_scratch_epoch_{finetune_epoch}'
                                      f'{suffix}.xlsx')
        # ######### scratch ################
        scratch_command_runer = CommandRunner(
            root_dir='./exps',
            task='finetune_for_classification_from_scratch',  # !
            per_device_train_batch_size=str(finetune_batch_size),
            num_train_epochs=finetune_epoch,  # !
            data_path=f'./data/{dataset}',
            dataset_name=dataset,
            dataset_split_strategy='random',
            modality_fusion_method='conv',
            freeze_language_model_params='True',  # ! TODO note: True for_unfreeze_emb_pooler_addnorm
            use_modality='num,cat,text',
            fuse_modality='num,cat',
            language_model_name='mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis',
            load_hf_model_from_cache='True',
            num_train_samples=num_train_samples,
            save_excel_path=save_excel_path,
        )
        scratch_command_runer.run()

        if only_scratch:
            continue
        # ######### pre fine ################
        # @@@@ 1. pretrain
        pre_command_runner = CommandRunner(
            root_dir='./exps',
            task='pretrain',
            per_device_train_batch_size=str(pre_batch_size),
            num_train_epochs=str(pretrain_epoch),
            data_path=f'./data/{dataset}',
            dataset_name=dataset,
            dataset_split_strategy='random',
            modality_fusion_method='conv',
            freeze_language_model_params='True',
            use_modality='num,cat,text',
            fuse_modality='num,cat',
            language_model_name='mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis',
            load_hf_model_from_cache='True',
            contrastive_targets='num,text',
            num_train_samples=num_train_samples,
        )
        pre_command_runner.run()
        # @@@@ 2. finetune
        finetune_command_runer = CommandRunner(
            root_dir='./exps',
            task='finetune_for_classification',  # !
            per_device_train_batch_size=str(finetune_batch_size),
            num_train_epochs=finetune_epoch,  # !
            data_path=f'./data/{dataset}',
            dataset_name=dataset,
            dataset_split_strategy='random',
            modality_fusion_method='conv',
            freeze_language_model_params='False',  # !
            use_modality='num,cat,text',
            fuse_modality='num,cat',
            language_model_name='mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis',
            load_hf_model_from_cache='True',
            num_train_samples=num_train_samples,
            pretrained_model_dir=pre_command_runner.output_dir,  # !
            save_excel_path=save_excel_path,
        )
        finetune_command_runer.run()


def run_multitask_rolling_window(repeat=1, suffix=''):
    dataset = 'cr2'
    finetune_batch_size = 160
    finetune_epoch = 100
    num_train_samples = None
    train_years_list = [
        [2010, 2011, 2012],
        [2011, 2012, 2013],
        [2012, 2013, 2014],
        [2013, 2014, 2015],
    ]
    test_year_List = [
        [2013],
        [2014],
        [2015],
        [2016],
    ]
    for i in range(repeat):
        for train_years, test_years in zip(train_years_list, test_year_List):
            train_years = ','.join(map(str, train_years))
            test_years = ','.join(map(str, test_years))
            save_excel_path = (f'./excel/{dataset}_res_rolling_window_'
                               f'#{train_years}#_#{test_years}#_'
                               f'multitask_epoch_{finetune_epoch}_rep_{repeat}{suffix}.xlsx')
            command_runer = CommandRunner(
                root_dir='./exps',
                task='clip_multi_task_classification',  # !
                per_device_train_batch_size=str(finetune_batch_size),
                num_train_epochs=finetune_epoch,  # !
                data_path=f'./data/{dataset}',
                dataset_name=dataset,
                dataset_split_strategy='rolling_window',
                modality_fusion_method='conv',
                freeze_language_model_params='True',  # ! TODO note: True for_unfreeze_emb_pooler_addnorm
                use_modality='num,cat,text',
                fuse_modality='num,cat',
                language_model_name='mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis',
                load_hf_model_from_cache='True',
                train_years=train_years,
                test_years=test_years,
                num_train_samples=num_train_samples,
                save_excel_path=save_excel_path,
            )
            command_runer.run()
            # print(command_runer.to_command())


def run_multitask_random_split(repeat=1, suffix=''):
    dataset = 'cr2'
    finetune_batch_size = 80
    finetune_epoch = 100
    num_train_samples = None
    for i in range(repeat):
        save_excel_path = (f'./excel/{dataset}_res_random_split_'
                           f'multitask_epoch_{finetune_epoch}_rep_{repeat}{suffix}.xlsx')
        # ######### scratch ################
        command_runer = CommandRunner(
            root_dir='./exps',
            task='clip_multi_task_classification',  # !
            per_device_train_batch_size=str(finetune_batch_size),
            num_train_epochs=finetune_epoch,  # !
            data_path=f'./data/{dataset}',
            dataset_name=dataset,
            dataset_split_strategy='random',
            modality_fusion_method='conv',
            freeze_language_model_params='True',  # ! TODO note: True for_unfreeze_emb_pooler_addnorm
            use_modality='num,cat,text',
            fuse_modality='num,cat',
            # language_model_name='mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis',
            # language_model_name='ProsusAI/finbert',
            language_model_name='gpt2',
            load_hf_model_from_cache='True',
            num_train_samples=num_train_samples,
            save_excel_path=save_excel_path,
        )
        command_runer.run()
        # print(command_runer.to_command())


def run_ptt_benchmark_rolling_window(repeat=10, dataset=None):
    train_years_list = [
        [2010, 2011, 2012],
        [2011, 2012, 2013],
        [2012, 2013, 2014],
        [2013, 2014, 2015],
    ]
    test_year_List = [
        [2013],
        [2014],
        [2015],
        [2016],
    ]
    for i in range(repeat):
        for train_years, test_years in zip(train_years_list, test_year_List):
            train_years = ','.join(map(str, train_years))
            test_years = ','.join(map(str, test_years))
            save_excel_path = (f'./excel/{dataset}_res_ptt_benchmark_rolling_window_'
                               f'#{train_years}#_#{test_years}#_rep_{i}.xlsx')
            command = (f"python pytorch_tabular_model_comparison.py"
                       f" --dataset_name {dataset}"
                       f" --data_path ./data/{dataset}"
                       f" --excel_path {save_excel_path}"
                       f" --dataset_split_strategy rolling_window"
                       f" --train_years {train_years}"
                       f" --test_years {test_years}")
            print(command)
            process = subprocess.Popen(command, shell=True)
            process.wait()


def run_ptt_benchmark_random_split(repeat=1, dataset=None):
    for i in range(repeat):
        save_excel_path = (f'./excel/{dataset}_res_ptt_benchmark_random_split_'
                           f'rep_{i}.xlsx')
        command = (f"python pytorch_tabular_model_comparison.py"
                   f" --dataset_name {dataset}"
                   f" --data_path ./data/{dataset}"
                   f" --excel_path {save_excel_path}"
                   f" --dataset_split_strategy random")
        print(command)
        process = subprocess.Popen(command, shell=True)
        process.wait()


def run_benchmark_rolling_window(repeat=10, dataset=None):
    train_years_list = [
        [2010, 2011, 2012],
        [2011, 2012, 2013],
        [2012, 2013, 2014],
        [2013, 2014, 2015],
    ]
    test_year_List = [
        [2013],
        [2014],
        [2015],
        [2016],
    ]
    for i in range(repeat):
        for train_years, test_years in zip(train_years_list, test_year_List):
            train_years = ','.join(map(str, train_years))
            test_years = ','.join(map(str, test_years))
            save_excel_path = (f'./excel/{dataset}_res_benchmark_rolling_window_'
                               f'#{train_years}#_#{test_years}#_rep_{i}.xlsx')
            command = (f"python ml_benchmark_model_comparison.py"
                       f" --dataset_name {dataset}"
                       f" --data_path ./data/{dataset}"
                       f" --excel_path {save_excel_path}"
                       f" --dataset_split_strategy rolling_window"
                       f" --train_years {train_years}"
                       f" --test_years {test_years}")
            print(command)
            process = subprocess.Popen(command, shell=True)
            process.wait()


def run_benchmark_random_split(repeat=10, dataset=None):
    for i in range(repeat):
        save_excel_path = (f'./excel/{dataset}_res_benchmark_random_split_'
                           f'rep_{i}.xlsx')
        command = (f"python ml_benchmark_model_comparison.py"
                   f" --dataset_name {dataset}"
                   f" --data_path ./data/{dataset}"
                   f" --excel_path {save_excel_path}"
                   f" --dataset_split_strategy random")
        print(command)
        process = subprocess.Popen(command, shell=True)
        process.wait()


if __name__ == '__main__':
    run_multitask_rolling_window(2, '_freezeall_roberta_catemb_numtextcontrastive_promptconcat_tinystruct_onvalacc')
