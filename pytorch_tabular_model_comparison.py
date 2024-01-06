import random
from dataclasses import dataclass, field

import pandas as pd
from pytorch_tabular import TabularModel
from pytorch_tabular.models import (
    CategoryEmbeddingModelConfig,
    FTTransformerConfig,
    TabNetModelConfig,
    GatedAdditiveTreeEnsembleConfig,
    TabTransformerConfig,
    AutoIntConfig
)
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig, ExperimentConfig
from pytorch_tabular.models.common.heads import LinearHeadConfig
from transformers import HfArgumentParser

from crmm.arguments import MultimodalDataArguments
from crmm.dataset.data_info import FEATURE_COLS
from crmm.dataset.multimodal_data import MultimodalData
from crmm.metrics import calc_classification_metrics_benchmark


# to run this file, one should activate the env for `pytorch_tabular`!

@dataclass
class PytorchTabularBenchmarkArguments(MultimodalDataArguments):
    excel_path: str = field(default=None, metadata={"help": "Path to the Excel file to save the results."})
    epochs: int = field(default=50, metadata={"help": "Number of epoch"})
    batch_size: int = field(default=300, metadata={"help": "Number of batch_size"})

class PytorchTabularBenchmark:
    def __init__(self, data_args: PytorchTabularBenchmarkArguments):
        self.data_args = data_args
        benchmark_data = MultimodalData(data_args, preprocess=False)
        (self.train_data,
         self.val_data,
         self.test_data) = (benchmark_data.train_data.drop(['GPT_description'], axis=1),
                            benchmark_data.val_data.drop(['GPT_description'], axis=1),
                            benchmark_data.test_data.drop(['GPT_description'], axis=1))

        dataset_cols = FEATURE_COLS[data_args.dataset_name]
        self.data_config = DataConfig(
            target=[dataset_cols['label']],  # target should always be a list.
            continuous_cols=dataset_cols['num'],
            categorical_cols=dataset_cols['cat'],
        )
        self.seed = random.randint(0, 10000)
        self.trainer_config = TrainerConfig(
            #     auto_lr_find=True, # Runs the LRFinder to automatically derive a learning rate
            batch_size=self.data_args.batch_size,
            max_epochs=self.data_args.epochs,
            early_stopping="valid_loss",  # Monitor valid_loss for early stopping
            early_stopping_mode="min",  # Set the mode as min because for val_loss, lower is better
            early_stopping_patience=self.data_args.epochs,  # No. of epochs of degradation training will wait before terminating
            checkpoints="valid_loss",  # Save best checkpoint monitoring val_loss
            load_best=True,  # After training, load the best checkpoint,
            seed=self.seed  # TODO actually not work, is a bug of pytorch tabular!
        )

        self.optimizer_config = OptimizerConfig()

        self.head_config = LinearHeadConfig(
            layers="",  # No additional layer in head, just a mapping layer to output_dim
            dropout=0.1,
            initialization="kaiming"
        ).__dict__  # Convert to dict to pass to the model config (OmegaConf doesn't accept objects)

        self.model_configs = [
            CategoryEmbeddingModelConfig(
                task="classification",
                layers="64-32",  # Number of nodes in each layer
                activation="ReLU",  # Activation between each layers
                learning_rate=1e-3,
                head="LinearHead",  # Linear Head
                head_config=self.head_config,  # Linear Head Config
            ),
            GatedAdditiveTreeEnsembleConfig(
                task="classification",
                learning_rate=1e-3,
                head="LinearHead",  # Linear Head
                head_config=self.head_config,  # Linear Head Config
            ),
            # GatedAdditiveTreeEnsembleConfig(
            #     task="classification",
            #     learning_rate=1e-3,
            #     head="LinearHead",  # Linear Head
            #     head_config=self.head_config,  # Linear Head Config
            #     gflu_stages=4,
            #     num_trees=30,
            #     tree_depth=5,
            #     chain_trees=False
            # ),
            FTTransformerConfig(
                task="classification",
                learning_rate=1e-3,
                head="LinearHead",  # Linear Head
                head_config=self.head_config,  # Linear Head Config
            ),
            TabTransformerConfig(
                task="classification",
                learning_rate=1e-3,
                head="LinearHead",  # Linear Head
                head_config=self.head_config,  # Linear Head Config
            ),
            AutoIntConfig(
                task="classification",
                learning_rate=1e-3,
                head="LinearHead",  # Linear Head
                head_config=self.head_config,  # Linear Head Config
            ),
            TabNetModelConfig(
                task="classification",
                learning_rate=1e-3,
                head="LinearHead",  # Linear Head
                head_config=self.head_config,  # Linear Head Config
            )

        ]

    def train_and_eval(self):
        # Create a DataFrame to store all the results
        results_list = []
        for model_config in self.model_configs:
            tabular_model = TabularModel(
                data_config=self.data_config,
                model_config=model_config,
                optimizer_config=self.optimizer_config,
                trainer_config=self.trainer_config,
            )
            tabular_model.fit(train=self.train_data, validation=self.val_data,
                              # TODO actually seed work here, is a bug of pytorch tabular!
                              seed=self.seed)
            tabular_model.evaluate(self.test_data)
            pred = tabular_model.predict(self.test_data)
            y_true = self.test_data[self.data_config.target]
            y_pred = pred['prediction']
            y_pred_prob = pred['1_probability']
            result = calc_classification_metrics_benchmark(model_config._model_name,
                                                           y_true.values.squeeze(),
                                                           y_pred.values.squeeze(),
                                                           y_pred_prob.values.squeeze())
            results_list.append(result)

        # Write the results to an Excel file
        results_df = pd.DataFrame(results_list)
        results_df.to_excel(self.data_args.excel_path, index=False)


if __name__ == '__main__':
    parser = HfArgumentParser([PytorchTabularBenchmarkArguments, ])
    args: PytorchTabularBenchmarkArguments = parser.parse_args_into_dataclasses()[0]

    benchmark = PytorchTabularBenchmark(args)
    benchmark.train_and_eval()
