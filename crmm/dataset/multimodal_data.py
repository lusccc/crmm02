import os.path
from functools import partial

import pandas as pd
from autogluon.common.features.types import R_FLOAT, R_CATEGORY
from autogluon.core import TabularDataset
from autogluon.tabular import TabularPredictor
from fastai.tabular.core import TabularPandas, Categorify
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler
from transformers.utils import logging

from crmm.arguments import MultimodalDataArguments
from .data_utils import agg_text_columns_func
from .multimodal_dataset import MultimodalDataset

logger = logging.get_logger('__name__')

"""
In autogluon:
Types of features in original data (raw dtype, special dtypes):
		('float', [])                      : 25 | ['currentRatio', 'quickRatio', 'cashRatio', 'daysOfSalesOutstanding', 'netProfitMargin', ...]
		('int', [])                        :  2 | ['Id', 'CIK']
		('object', [])                     :  4 | ['Name', 'Symbol', 'Rating Agency Name', 'Sector']
		('object', ['datetime_as_object']) :  1 | ['Date']
		('object', ['text'])               :  2 | ['secText', 'secKeywords']
	Types of features in processed data (raw dtype, special dtypes):
		('category', [])                    :    4 | ['Name', 'Symbol', 'Rating Agency Name', 'Sector']
		('category', ['text_as_category'])  :    2 | ['secText', 'secKeywords']
		('float', [])                       :   25 | ['currentRatio', 'quickRatio', 'cashRatio', 'daysOfSalesOutstanding', 'netProfitMargin', ...]
		('int', [])                         :    2 | ['Id', 'CIK']
		('int', ['binned', 'text_special']) :   29 | ['secText.char_count', 'secText.word_count', 'secText.capital_ratio', 'secText.lower_ratio', 'secText.digit_ratio', ...]
		('int', ['datetime_as_int'])        :    5 | ['Date', 'Date.year', 'Date.month', 'Date.day', 'Date.dayofweek']
		('int', ['text_ngram'])             : 2856 | ['__nlp__.000', '__nlp__.10', '__nlp__.10 for', '__nlp__.10 for the', '__nlp__.10 million', ...]
"""


class MultimodalData:
    def __init__(self,
                 data_args: MultimodalDataArguments,
                 label_col='Rating',
                 label_list=None,
                 text_cols=None,
                 num_transform_method='quantile_normal', ) -> None:
        self.data_args = data_args
        self.label_col = label_col
        self.label_list = label_list
        self.text_cols = text_cols
        self.num_transform_method = num_transform_method

        self.has_val = data_args.use_val

        self.raw_train_data, self.raw_test_data, self.raw_val_data = self.load_data()

        self.train_dataset, self.test_dataset, self.val_dataset = None, None, None

        self.transform_features()

    def load_data(self):
        train_data = TabularDataset(os.path.join(self.data_args.data_path, 'train(with_description_col).csv'))
        test_data = TabularDataset(os.path.join(self.data_args.data_path, 'test.csv'))
        test_data['GPT_description'] = ''  # to avoid error in  later predictor.transform_features
        val_data = TabularDataset(
            os.path.join(self.data_args.data_path, 'val(with_description_col).csv')) if self.has_val else None
        return train_data, test_data, val_data

    def transform_features(self):
        train_labels = self.raw_train_data[self.label_col]
        test_labels = self.raw_test_data[self.label_col]
        val_labels = self.raw_val_data[self.label_col] if self.has_val else None

        # TODO manually construct FeatureMetadata
        # note 'RF': {} is just a placeholder, we don't actually use RF. but we want to transform feature after `fit`
        # also note that RF in autogluon.tabular.models.rf.rf_model.RFModel._preprocess is not normalized!
        # predictor = TabularPredictor(label=self.label_col).fit(self.raw_train_data, hyperparameters={'RF': {}})
        # predictor = TabularPredictor.load("AutogluonModels/ag-20231026_024542/")
        # predictor = TabularPredictor.load("AutogluonModels/ag-20231030_092803")
        autogluon_res_dir = self.data_args.feature_transform_res_dir
        predictor = TabularPredictor.load(autogluon_res_dir) if autogluon_res_dir else TabularPredictor(
            label=self.label_col).fit(self.raw_train_data, hyperparameters={'RF': {}})
        tfm_train_feats = predictor.transform_features(self.raw_train_data)
        tfm_test_feats = predictor.transform_features(self.raw_test_data)
        tfm_val_feats = predictor.transform_features(self.raw_val_data) if self.has_val else None

        # -----NUM FEATURES
        # ['currentRatio', 'quickRatio', 'cashRatio', 'daysOfSalesOutstanding', 'netProfitMargin',
        # 'pretaxProfitMargin', 'grossProfitMargin', 'operatingProfitMargin', 'returnOnAssets',
        # 'returnOnCapitalEmployed', 'returnOnEquity', 'assetTurnover', 'fixedAssetTurnover', 'debtEquityRatio',
        # 'debtRatio', 'effectiveTaxRate', 'freeCashFlowOperatingCashFlowRatio', 'freeCashFlowPerShare',
        # 'cashPerShare', 'companyEquityMultiplier', 'ebitPerRevenue', 'enterpriseValueMultiple',
        # 'operatingCashFlowPerShare', 'operatingCashFlowSalesRatio', 'payablesTurnover']
        num_transformer = self.get_num_transformer()
        num_cols = self.get_cols_of('num', predictor)
        get_num_preprocessed = lambda df: df.loc[:, num_cols].fillna(df.loc[:, num_cols].median())

        train_num_feats = get_num_preprocessed(tfm_train_feats)
        test_num_feats = get_num_preprocessed(tfm_test_feats)
        val_num_feats = get_num_preprocessed(tfm_val_feats) if self.has_val else None

        train_num_feats_cp1 = pd.DataFrame(train_num_feats)

        if num_transformer:
            # https://github.com/georgian-io/Multimodal-Toolkit
            num_transformer.fit(train_num_feats)
            self.train_num_feats = num_transformer.transform(train_num_feats)
            self.test_num_feats = num_transformer.transform(test_num_feats)
            self.val_num_feats = num_transformer.transform(val_num_feats) if self.has_val else None

        train_num_feats_cp2 = pd.DataFrame(train_num_feats)

        # -----CAT FEATURES
        #   0~350  0~350              1~4          1~12
        # ['Name', 'Symbol', 'Rating Agency Name', 'Sector', 'CIK']
        cat_cols = self.get_cols_of('cat', predictor)
        get_categorized = lambda df: TabularPandas(df.loc[:, cat_cols], procs=[Categorify], cat_names=cat_cols)

        self.train_cat_feats = get_categorized(tfm_train_feats)
        self.test_cat_feats = get_categorized(tfm_test_feats)
        self.val_cat_feats = get_categorized(tfm_val_feats) if self.has_val else None

        # ----- TEXT FEATURES
        # ['secText', 'secKeywords']
        text_cols = self.get_cols_of('text', predictor)
        get_texts_arrged = lambda df: self.aggregate_txt_on(df.loc[:, text_cols])
        # note: text should be read from raw data! because tfm data is transformed!
        self.train_text_feats = get_texts_arrged(self.raw_train_data)
        # self.test_text_feats = get_texts_arrged(self.raw_test_data)
        self.val_text_feats = get_texts_arrged(self.raw_val_data) if self.has_val else None

        # @@@@@ CREATE DATASET
        self.train_dataset = self.create_dataset(self.train_text_feats, self.train_cat_feats, self.train_num_feats,
                                                 train_labels, None, self.label_list)
        self.test_dataset = self.create_dataset(None, self.test_cat_feats, self.test_num_feats,
                                                test_labels, None, self.label_list)
        self.val_dataset = self.create_dataset(self.val_text_feats, self.val_cat_feats, self.val_num_feats, val_labels,
                                               None, self.label_list) if self.has_val else None

    def create_dataset(self, texts_list, categorical_feats, numerical_feats, labels, data_df, label_list):
        dt = MultimodalDataset(texts_list, categorical_feats.xs.values,
                               numerical_feats, labels.values, data_df, label_list)
        return dt

    def get_datasets(self):
        return self.train_dataset, self.test_dataset, self.val_dataset

    #  nunqiue of each cat col, and emb dim of each cat col
    def get_nunique_cat_nums_and_emb_dim(self, equal_dim=None):
        all_cat_feats = pd.concat([self.train_cat_feats.items, self.test_cat_feats.items, self.val_cat_feats.items])
        nunique_cat_nums = list(all_cat_feats.nunique().values)
        if equal_dim:
            cat_emb_dims = [equal_dim for _ in nunique_cat_nums]
        else:
            cat_emb_dims = []
            for cat_num in nunique_cat_nums:
                if cat_num >= 300:
                    emb_dim = 32
                elif 100 <= cat_num < 300:
                    emb_dim = 16
                else:
                    emb_dim = 8
                cat_emb_dims.append(emb_dim)
        # return int, rather than numpy int64
        return list(map(int, nunique_cat_nums)), list(map(int, cat_emb_dims))

    def get_cols_of(self, modality, predictor=None):
        # TODO In future work, feature selection can be more scientific
        feat_meta = predictor.feature_metadata
        # the num and cat cols are inferred by autogluon predictor
        if modality == 'num':
            cols = feat_meta.get_features(valid_raw_types=[R_FLOAT])
            self.num_cols = cols
        elif modality == 'cat':
            cols = feat_meta.get_features(valid_raw_types=[R_CATEGORY])
            # TODO CIK should be treated as category feature, but it is considered as R_INT in autogluon
            # cols += ['CIK']

            # text cols are considered as category feature in autogluon, however, should be avoided
            for tc in self.text_cols:
                if tc in cols:
                    cols.remove(tc)
            self.cat_cols = cols
        elif modality == 'text':
            cols = self.text_cols
        else:
            cols = None
        return cols

    def get_num_transformer(self):
        num_transform_method = self.num_transform_method
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
        self.num_transformer = num_transformer
        return num_transformer

    def aggregate_txt_on(self, data):
        empty_text_values = ['nan', 'None']
        # should be '[SEP]'!
        sep_text_token_str = '[SEP]'
        agg_func = partial(agg_text_columns_func, empty_text_values, None)
        texts_list = data[self.text_cols].agg(agg_func, axis=1).tolist()
        for i, text in enumerate(texts_list):
            texts_list[i] = f' {sep_text_token_str} '.join(text)
        logger.info(f'a raw text sample: {texts_list[0]}')
        return texts_list
