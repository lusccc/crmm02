import argparse
import time

import numpy as np
import pandas as pd
from autogluon.core.models import AbstractModel
from autogluon.features.generators import LabelEncoderFeatureGenerator
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.tabular.models import KNNModel
from imblearn.metrics import geometric_mean_score
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, OrdinalEncoder


class CustomSVMModel(AbstractModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None
        self._scaler = None

    def _preprocess(self, X: pd.DataFrame, is_train=False, **kwargs) -> np.ndarray:
        print(f'Entering the `_preprocess` method: {len(X)} rows of data (is_train={is_train})')
        X = super()._preprocess(X, **kwargs)

        if is_train:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
        X = X.fillna(0).to_numpy(dtype=np.float32)
        # 数据标准化
        if is_train:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)
        else:
            X = self._scaler.transform(X)

        return X

    def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        print('Entering the `_fit` method')

        from sklearn.svm import SVC, SVR

        if self.problem_type in ['regression', 'softclass']:
            model_cls = SVR
        else:
            model_cls = SVC

        X = self.preprocess(X, is_train=True)
        params = self._get_model_params()
        print(f'Hyperparameters: {params}')
        self.model = model_cls(**params)
        self.model.fit(X, y)
        print('Exiting the `_fit` method')

    def _set_default_params(self):
        default_params = {
            'C': 1.0,
            'kernel': 'rbf',
            'degree': 3,
            'gamma': 'scale',
            'coef0': 0.0,
            'shrinking': True,
            'probability': True,
            'tol': 1e-3,
            'cache_size': 200,
            'class_weight': None,
            'verbose': False,
            'max_iter': -1,
            'decision_function_shape': 'ovr',
            'break_ties': False,
            'random_state': None,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            get_features_kwargs=dict(
                valid_raw_types=['int', 'float', 'category'],
            ),
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params


class CustomMLPModel(AbstractModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None
        self._scaler = None

    def _preprocess(self, X: pd.DataFrame, is_train=False, **kwargs) -> np.ndarray:
        print(f'Entering the `_preprocess` method: {len(X)} rows of data (is_train={is_train})')
        X = super()._preprocess(X, **kwargs)

        if is_train:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
        X = X.fillna(0).to_numpy(dtype=np.float32)
        # 数据标准化
        if is_train:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)
        else:
            X = self._scaler.transform(X)

        return X

    def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        print('Entering the `_fit` method')

        from sklearn.neural_network import MLPClassifier, MLPRegressor

        if self.problem_type in ['regression', 'softclass']:
            model_cls = MLPRegressor
        else:
            model_cls = MLPClassifier

        X = self.preprocess(X, is_train=True)
        params = self._get_model_params()
        print(f'Hyperparameters: {params}')
        self.model = model_cls(**params)
        self.model.fit(X, y)
        print('Exiting the `_fit` method')

    def _set_default_params(self):
        default_params = {
            'hidden_layer_sizes': (256,),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'batch_size': 'auto',
            'learning_rate': 'constant',
            'learning_rate_init': 0.001,
            'power_t': 0.5,
            'max_iter': 1000,
            'shuffle': True,
            'random_state': None,
            'tol': 1e-4,
            'verbose': False,
            'warm_start': False,
            'momentum': 0.9,
            'nesterovs_momentum': True,
            'early_stopping': False,
            'validation_fraction': 0.1,
            'beta_1': 0.9,
            'beta_2': 0.999,
            'epsilon': 1e-8,
            'n_iter_no_change': 10,
            'max_fun': 15000,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            get_features_kwargs=dict(
                valid_raw_types=['int', 'float', 'category'],
            ),
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params


class CustomKNNModel(KNNModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._scaler = None

    def _preprocess(self, X: pd.DataFrame, is_train=False, **kwargs) -> np.ndarray:
        print(f'Entering the `_preprocess` method: {len(X)} rows of data (is_train={is_train})')
        X = super()._preprocess(X, **kwargs)

        # 数据标准化
        if is_train:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)
        else:
            X = self._scaler.transform(X)

        return X

    def _fit(self,
             X,
             y,
             num_cpus=-1,
             time_limit=None,
             sample_weight=None,
             **kwargs):
        time_start = time.time()
        X = self.preprocess(X, is_train=True)
        params = self._get_model_params()
        if 'n_jobs' not in params:
            params['n_jobs'] = num_cpus

        num_rows_max = len(X)
        # FIXME: v0.1 Must store final num rows for refit_full or else will use everything! Worst case refit_full could train far longer than the original model.
        if time_limit is None or num_rows_max <= 10000:
            self.model = self._get_model_type()(**params).fit(X, y)
        else:
            self.model = self._fit_with_samples(X=X, y=y, model_params=params,
                                                time_limit=time_limit - (time.time() - time_start))


class CustomAdaBoostModel(CustomMLPModel):
    def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        print('Entering the `_fit` method')

        model_cls = AdaBoostClassifier

        X = self.preprocess(X, is_train=True)
        params = self._get_model_params()
        print(f'Hyperparameters: {params}')
        self.model = model_cls(**params)
        self.model.fit(X, y)
        print('Exiting the `_fit` method')

    def _set_default_params(self):
        default_params = {
            'base_estimator': None,
            'n_estimators': 50,
            'learning_rate': 1.0,
            'algorithm': 'SAMME.R',
            'random_state': None,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)


class CustomGBDTModel(CustomMLPModel):
    def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        print('Entering the `_fit` method')

        model_cls = GradientBoostingClassifier

        X = self.preprocess(X, is_train=True)
        params = self._get_model_params()
        print(f'Hyperparameters: {params}')
        self.model = model_cls(**params)
        self.model.fit(X, y)
        print('Exiting the `_fit` method')

    def _set_default_params(self):
        default_params = {
            'loss': 'log_loss',
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 1.0,
            'criterion': 'friedman_mse',
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'min_weight_fraction_leaf': 0.0,
            'max_depth': 3,
            'min_impurity_decrease': 0.0,
            'init': None,
            'random_state': None,
            'max_features': None,
            'verbose': 0,
            'max_leaf_nodes': None,
            'warm_start': False,
            'validation_fraction': 0.1,
            'n_iter_no_change': None,
            'tol': 1e-4,
            'ccp_alpha': 0.0,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)


class CustomBaggingModel(CustomMLPModel):
    def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        print('Entering the `_fit` method')

        model_cls = BaggingClassifier

        X = self.preprocess(X, is_train=True)
        params = self._get_model_params()
        print(f'Hyperparameters: {params}')
        self.model = model_cls(**params)
        self.model.fit(X, y)
        print('Exiting the `_fit` method')

    def _set_default_params(self):
        default_params = {
            'base_estimator': None,
            'n_estimators': 10,
            'max_samples': 1.0,
            'max_features': 1.0,
            'bootstrap': True,
            'bootstrap_features': False,
            'oob_score': False,
            'warm_start': False,
            'n_jobs': None,
            'random_state': None,
            'verbose': 0,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)


class Benchmark:

    def __init__(self, data_path, excel_path) -> None:
        super().__init__()
        self.train_data = TabularDataset(f'{data_path}/train.csv')
        self.val_data = TabularDataset(f'{data_path}/val.csv')
        self.test_data = TabularDataset(f'{data_path}/test.csv')
        self.excel_path = excel_path

        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        for i, data in enumerate([self.train_data, self.val_data, self.test_data]):
            cols_to_drop = ['Id', 'Rating']
            data.drop(columns=[col for col in cols_to_drop if col in data.columns], inplace=True)
            # to label encoding multi cat cols a time
            if 'cr2' in data_path:  # for cr2 dataset
                cat_cols = ['Rating Agency', 'Corporation', 'CIK', 'SIC Code', 'Sector', 'Ticker']
            else:  # for cr dataset
                cat_cols = ['Name', 'Symbol', 'Rating Agency Name', 'Sector', 'CIK']

            if i == 0:
                oe.fit(data[cat_cols])
            data[cat_cols] = oe.transform(data[cat_cols])

    def train_and_eval(self):
        output_path = 'benchmark_res'
        label_column = 'binaryRating' if 'binaryRating' in self.train_data.columns else 'Binary Rating'
        model_performance = {}
        # delete labels from test data since we wouldn't have them in practice
        test_data = self.test_data.drop(labels=[label_column], axis=1)
        y_true = self.test_data[label_column]

        # @@@@ 1. ag models
        ag_hyperparameters = {
            # 'GBM': {},  # LightGBM
            # 'CAT': {},  # CatBoost
            'XGB': {},  # XGBoost
            'RF': {},  # Random Forest
            'XT': {},  # Extremely Randomized Trees
            # 'KNN': {},  # K-Nearest Neighbors
            'LR': {},  # Linear Regression
            # 'NN_TORCH': {},  # Neural Network (PyTorch)
            # 'FASTAI': {},  # Neural Network (FastAI)
            # 'AG_AUTOMM': {},  # MultimodalPredictor (requires GPU)
        }
        ag_predictor = TabularPredictor(label=label_column, path=output_path).fit(train_data=self.train_data,
                                                                                  tuning_data=self.val_data,
                                                                                  hyperparameters=ag_hyperparameters,
                                                                                  num_cpus=24,
                                                                                  save_space=True, )
        for model_name in ag_predictor.get_model_names():
            y_pred_proba = ag_predictor.predict_proba(test_data, model=model_name).values[:, 1]
            y_pred = ag_predictor.predict(test_data, model=model_name)
            metrics = self.compute_metrics(y_true, y_pred, y_pred_proba)
            model_performance[model_name] = metrics

        # @@@@ 2. my custom models
        custom_hyperparameters = {
            CustomSVMModel: {},
            CustomMLPModel: {},
            CustomKNNModel: {},
            CustomGBDTModel: {},
            CustomBaggingModel: {},
            CustomAdaBoostModel: {}
        }
        custom_predictor = TabularPredictor(label=label_column, path=output_path).fit(train_data=self.train_data,
                                                                                      tuning_data=self.val_data,
                                                                                      num_cpus=24,
                                                                                      save_space=True,
                                                                                      hyperparameters=custom_hyperparameters)
        for model_name in custom_predictor.get_model_names():
            y_pred_proba = custom_predictor.predict_proba(test_data, model=model_name).values[:, 1]
            y_pred = custom_predictor.predict(test_data, model=model_name)
            metrics = self.compute_metrics(y_true, y_pred, y_pred_proba)
            model_performance[model_name] = metrics

        df = pd.DataFrame(model_performance).T
        df.to_excel(self.excel_path)

    def compute_metrics(self, y_true, y_pred, y_proba):
        acc = accuracy_score(y_true, y_pred)
        if y_proba is not None:
            auc = roc_auc_score(y_true, y_proba)
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            ks = max(tpr - fpr)
        else:
            auc = None
            ks = None
        g_mean = geometric_mean_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        type1_acc = tp / (tp + fn)
        type2_acc = tn / (tn + fp)

        return {
            'Acc': acc,
            'AUC': auc,
            'KS': ks,
            'G-mean': g_mean,
            'Type-I Acc': type1_acc,
            'Type-II Acc': type2_acc,
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default='./data/cr2_cls2_mixed_st10_kw20')
    parser.add_argument('--excel_path', type=str, default='./result.xlsx')
    args = parser.parse_args()

    benchmark = Benchmark(args.data_path, args.excel_path)
    benchmark.train_and_eval()
