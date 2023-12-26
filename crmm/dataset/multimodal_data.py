import os.path

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer, OrdinalEncoder, OneHotEncoder
from transformers.utils import logging

from crmm.arguments import MultimodalDataArguments
from .data_info import FEATURE_COLS
from .multimodal_dataset import MultimodalDataset

logger = logging.get_logger('transformers')


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names]


class MultimodalData:
    # `preprocess=False` is adapted and used for pytorch_tabular_model_comparison.PytorchTabularBenchmark
    def __init__(self, data_args: MultimodalDataArguments, preprocess=True):
        self.data_args = data_args
        self.dataset_name = self.data_args.dataset_name
        self.has_val = data_args.use_val
        self.label_values = FEATURE_COLS[self.dataset_name]['label_values']

        (self.train_data,
         self.test_data,
         self.val_data) = self.load_and_preprocess_data(FEATURE_COLS[self.dataset_name], preprocess)

        self.train_dataset = MultimodalDataset(
            self.train_data.loc[:, FEATURE_COLS[self.dataset_name]['text']],
            self.train_data.loc[:, FEATURE_COLS[self.dataset_name]['cat']],
            self.train_data.loc[:, FEATURE_COLS[self.dataset_name]['num']],
            self.train_data.loc[:, FEATURE_COLS[self.dataset_name]['label']]
        )
        self.test_dataset = MultimodalDataset(
            self.test_data.loc[:, FEATURE_COLS[self.dataset_name]['text']],
            self.test_data.loc[:, FEATURE_COLS[self.dataset_name]['cat']],
            self.test_data.loc[:, FEATURE_COLS[self.dataset_name]['num']],
            self.test_data.loc[:, FEATURE_COLS[self.dataset_name]['label']]
        )
        self.val_dataset = MultimodalDataset(
            self.val_data.loc[:, FEATURE_COLS[self.dataset_name]['text']],
            self.val_data.loc[:, FEATURE_COLS[self.dataset_name]['cat']],
            self.val_data.loc[:, FEATURE_COLS[self.dataset_name]['num']],
            self.val_data.loc[:, FEATURE_COLS[self.dataset_name]['label']]
        ) if self.has_val else None

    def load_and_preprocess_data(self, feature_cols, preprocess=True):
        all_df = pd.read_csv(os.path.join(self.data_args.data_path, 'all(with_description_col).csv'))
        if self.data_args.dataset_split_strategy == 'random':
            if self.data_args.num_train_samples is not None:
                train_df = all_df[:self.data_args.num_train_samples]
                if self.has_val:
                    # 剩余数据的10%作为验证集，然后其余的作为测试集
                    remaining_df = all_df[self.data_args.num_train_samples:]
                    val_df = remaining_df[:int(0.1 * len(remaining_df))]
                    test_df = remaining_df[int(0.1 * len(remaining_df)):]
                else:
                    # 剩余数据作为测试集
                    test_df = all_df[self.data_args.num_train_samples:]
                    val_df = None
            else:
                train_df = all_df[:int(0.8 * len(all_df))]
                test_df = all_df[int(0.8 * len(all_df)):]
                if self.has_val:
                    val_df = train_df[:int(0.1 * len(train_df))]
                    train_df = train_df[int(0.1 * len(train_df)):]
                else:
                    val_df = None
        elif self.data_args.dataset_split_strategy == 'rolling_window':
            date_col = 'Rating Date' if 'Rating Date' in all_df.columns else 'Date'
            all_df[date_col] = pd.to_datetime(all_df[date_col])
            all_df['Rating Year'] = all_df[date_col].dt.year.astype(int)
            train_df = all_df[all_df['Rating Year'].isin(self.data_args.train_years)]
            test_df = all_df[all_df['Rating Year'].isin(self.data_args.test_years)]
            if self.data_args.num_train_samples is not None:
                if self.has_val:
                    val_df = train_df[self.data_args.num_train_samples:]
                train_df = train_df[:self.data_args.num_train_samples]
            else:
                if self.has_val:
                    val_df = train_df[:int(0.1 * len(train_df))]
                    train_df = train_df[int(0.1 * len(train_df)):]
                else:
                    val_df = None
        else:
            raise ValueError(f'Unknown dataset_split_strategy: {self.data_args.dataset_split_strategy}')

        test_df[feature_cols['text']] = ''  # to avoid data leak, and make code not throw error!
        val_df[feature_cols['text']] = ''

        # convert cat data who are float to int, then to str. thus can be tokenized as text!
        cat_float_cols = ['CIK', 'SIC Code']
        for df in [train_df, test_df, val_df]:
            if df is not None:
                for col in cat_float_cols:
                    if col in df.columns:
                        df[col] = df[col].fillna(-1).astype(int).astype(str)

        if not preprocess:  # adapt for pytorch_tabular_model_comparison.PytorchTabularBenchmark
            return train_df, test_df, val_df

        num_pipeline = Pipeline([
            ('selector', DataFrameSelector(feature_cols['num'])),
            ('imputer', SimpleImputer(strategy="mean")),
            ('std_scaler', self.get_num_transformer(self.data_args.numerical_transformer_method)),
        ])

        cat_pipeline = Pipeline([
            ('selector', DataFrameSelector(feature_cols['cat'])),
            ('imputer', SimpleImputer(strategy="most_frequent")),
            # for benchmark condition
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            if self.data_args.cat_encoder == 'ordinal' else OneHotEncoder(handle_unknown='ignore', sparse=False))
        ]) if hasattr(self.data_args, 'cat_encoder') else Pipeline([
            ('selector', DataFrameSelector(feature_cols['cat'])),
            ('imputer', SimpleImputer(strategy="most_frequent")),
        ])

        text_pipeline = Pipeline([
            ('selector', DataFrameSelector(feature_cols['text'])),
        ])

        preprocess_pipeline = ColumnTransformer([
            ('num', num_pipeline, feature_cols['num']),
            ('cat', cat_pipeline, feature_cols['cat']),
            ('text', text_pipeline, feature_cols['text']),
        ])

        preprocess_pipeline.fit(train_df)
        train_preprocessed = preprocess_pipeline.transform(train_df)
        test_preprocessed = preprocess_pipeline.transform(test_df)
        val_preprocessed = preprocess_pipeline.transform(val_df) if self.has_val else None

        cols = feature_cols['num'] + feature_cols['cat'] + feature_cols['text']
        train_preprocessed = pd.DataFrame(train_preprocessed, columns=cols).infer_objects()
        test_preprocessed = pd.DataFrame(test_preprocessed, columns=cols).infer_objects()
        val_preprocessed = pd.DataFrame(val_preprocessed, columns=cols).infer_objects() if self.has_val else None

        train_preprocessed[feature_cols['label']] = train_df[feature_cols['label']].values
        test_preprocessed[feature_cols['label']] = test_df[feature_cols['label']].values
        val_preprocessed[feature_cols['label']] = val_df[feature_cols['label']].values

        return train_preprocessed, test_preprocessed, val_preprocessed

    def get_datasets(self):
        return self.train_dataset, self.test_dataset, self.val_dataset

    def get_num_transformer(self, num_transform_method):
        if num_transform_method != 'none':
            if num_transform_method == 'yeo_johnson':
                num_transformer = PowerTransformer(method='yeo-johnson')
            elif num_transform_method == 'box_cox':
                num_transformer = PowerTransformer(method='box-cox')
            elif num_transform_method == 'quantile_normal':
                num_transformer = QuantileTransformer(output_distribution='normal')
            elif num_transform_method == 'standard':
                num_transformer = StandardScaler()
            else:
                raise ValueError(f'preprocessing transformer method '
                                 f'{num_transform_method} not implemented')
        else:
            num_transformer = None
        return num_transformer
