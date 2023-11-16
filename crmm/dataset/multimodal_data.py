import os.path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer
from transformers.utils import logging

from crmm.arguments import MultimodalDataArguments
from .multimodal_dataset import MultimodalDataset

logger = logging.get_logger('transformers')

FEATURE_COLS = {
    'cr': {
        'num': ['currentRatio', 'quickRatio', 'cashRatio',
                'daysOfSalesOutstanding', 'netProfitMargin', 'pretaxProfitMargin',
                'grossProfitMargin', 'operatingProfitMargin', 'returnOnAssets',
                'returnOnCapitalEmployed', 'returnOnEquity', 'assetTurnover',
                'fixedAssetTurnover', 'debtEquityRatio', 'debtRatio',
                'effectiveTaxRate', 'freeCashFlowOperatingCashFlowRatio',
                'freeCashFlowPerShare', 'cashPerShare', 'companyEquityMultiplier',
                'ebitPerRevenue', 'enterpriseValueMultiple',
                'operatingCashFlowPerShare', 'operatingCashFlowSalesRatio',
                'payablesTurnover'],
        'cat': ['Name', 'Symbol', 'Rating Agency Name', 'Sector', 'CIK'],
        'text': ['GPT_description'],
        'label': 'binaryRating',
        'label_values': [0, 1]
    },
    'cr2': {
        'num': ['Current Ratio',
                'Long-term Debt / Capital', 'Debt/Equity Ratio', 'Gross Margin',
                'Operating Margin', 'EBIT Margin', 'EBITDA Margin',
                'Pre-Tax Profit Margin', 'Net Profit Margin', 'Asset Turnover',
                'ROE - Return On Equity', 'Return On Tangible Equity',
                'ROA - Return On Assets', 'ROI - Return On Investment',
                'Operating Cash Flow Per Share', 'Free Cash Flow Per Share'],
        'cat': ['Rating Agency', 'Corporation', 'CIK', 'SIC Code', 'Sector', 'Ticker'],
        'text': ['GPT_description'],
        'label': 'Binary Rating',
        'label_values': [0, 1]
    }
}


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names]


class MultimodalData:
    def __init__(self, data_args: MultimodalDataArguments):
        self.data_args = data_args
        self.dataset_name = self.data_args.dataset_name
        self.has_val = data_args.use_val
        self.label_values = FEATURE_COLS[self.dataset_name]['label_values']

        (self.train_preprocessed,
         self.test_preprocessed,
         self.val_preprocessed) = self.load_and_preprocess_data(FEATURE_COLS[self.dataset_name])

        self.train_dataset = MultimodalDataset(
            self.train_preprocessed.loc[:, FEATURE_COLS[self.dataset_name]['text']],
            self.train_preprocessed.loc[:, FEATURE_COLS[self.dataset_name]['cat']],
            self.train_preprocessed.loc[:, FEATURE_COLS[self.dataset_name]['num']],
            self.train_preprocessed.loc[:, FEATURE_COLS[self.dataset_name]['label']]
        )
        self.test_dataset = MultimodalDataset(
            self.test_preprocessed.loc[:, FEATURE_COLS[self.dataset_name]['text']],
            self.test_preprocessed.loc[:, FEATURE_COLS[self.dataset_name]['cat']],
            self.test_preprocessed.loc[:, FEATURE_COLS[self.dataset_name]['num']],
            self.test_preprocessed.loc[:, FEATURE_COLS[self.dataset_name]['label']]
        )
        self.val_dataset = MultimodalDataset(
            self.val_preprocessed.loc[:, FEATURE_COLS[self.dataset_name]['text']],
            self.val_preprocessed.loc[:, FEATURE_COLS[self.dataset_name]['cat']],
            self.val_preprocessed.loc[:, FEATURE_COLS[self.dataset_name]['num']],
            self.val_preprocessed.loc[:, FEATURE_COLS[self.dataset_name]['label']]
        ) if self.has_val else None

    def load_and_preprocess_data(self, feature_cols):
        train_df = pd.read_csv(os.path.join(self.data_args.data_path, 'train(with_description_col).csv'))
        test_df = pd.read_csv(os.path.join(self.data_args.data_path, 'test.csv'))
        val_df = pd.read_csv(os.path.join(self.data_args.data_path, 'val(with_description_col).csv')) \
            if self.has_val else None

        test_df[feature_cols['text']] = ''  # only to make code not throw error!

        # convert cat data who are float to int, then to str. thus can be tokenized as text!
        cat_float_cols = ['CIK', 'SIC Code']
        for df in [train_df, test_df, val_df]:
            if df is not None:
                for col in cat_float_cols:
                    if col in df.columns:
                        df[col] = df[col].fillna(-1).astype(int).astype(str)

        num_pipeline = Pipeline([
            ('selector', DataFrameSelector(feature_cols['num'])),
            ('imputer', SimpleImputer(strategy="mean")),
            ('std_scaler', self.get_num_transformer(self.data_args.numerical_transformer_method)),
        ])

        cat_pipeline = Pipeline([
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
        train_preprocessed = pd.DataFrame(train_preprocessed, columns=cols)
        test_preprocessed = pd.DataFrame(test_preprocessed, columns=cols)
        val_preprocessed = pd.DataFrame(val_preprocessed, columns=cols) if self.has_val else None

        train_preprocessed[feature_cols['label']] = train_df[feature_cols['label']]
        test_preprocessed[feature_cols['label']] = test_df[feature_cols['label']]
        val_preprocessed[feature_cols['label']] = val_df[feature_cols['label']]

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
