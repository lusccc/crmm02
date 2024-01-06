import argparse
import os
from dataclasses import dataclass, field


import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from transformers import HfArgumentParser
from xgboost import XGBClassifier

from crmm.arguments import MultimodalDataArguments
from crmm.dataset.multimodal_data import MultimodalData
from crmm.metrics import calc_classification_metrics_benchmark


@dataclass
class MLBenchmarkDataArguments(MultimodalDataArguments):
    excel_path: str = field(default=None, metadata={"help": "Path to the Excel file to save the results."})
    cat_encoder: str = field(default="ordinal", metadata={"help": "Encoding method for categorical features."})


class MLBenchmark:

    def __init__(self, data_args: MLBenchmarkDataArguments):
        self.data_args = data_args
        benchmark_data = MultimodalData(data_args)
        (self.train_data,
         self.val_data,
         self.test_data) = (benchmark_data.train_data.drop(['GPT_description'], axis=1),
                            benchmark_data.val_data.drop(['GPT_description'], axis=1),
                            benchmark_data.test_data.drop(['GPT_description'], axis=1))
        print()

    def train_and_eval(self):
        models = [
            ('LogR', LogisticRegression()),
            ('SVM', SVC(probability=True)),  # probability=True to enable predict_proba for AUC
            ('KNN', KNeighborsClassifier()),
            ('DT', DecisionTreeClassifier()),
            ('MLP', MLPClassifier()),
            ('Adaboost', AdaBoostClassifier()),
            ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
            # disable the default label encoder and use logloss as eval metric
            ('GBDT', GradientBoostingClassifier()),
            ('RF', RandomForestClassifier()),
        ]

        label_column = 'binaryRating' if 'binaryRating' in self.train_data.columns else 'Binary Rating'
        X_train = self.train_data.drop(label_column, axis=1)
        y_train = self.train_data[label_column]
        X_val = self.val_data.drop(label_column, axis=1)
        y_val = self.val_data[label_column]
        X_test = self.test_data.drop(label_column, axis=1)
        y_test = self.test_data[label_column]

        # Standardize the features for models like SVM and KNN
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)


        results_list = []
        for name, model in models:
            if name in ['SVM', 'KNN', 'LogR']:  # Use scaled data for SVM and KNN
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]  # Probability of positive class
            elif name == 'XGBoost':  # Use validation data for early stopping in XGBoost
                model.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_val, y_val)], verbose=False)
                y_pred = model.predict(X_test)
                y_pred_prob = model.predict_proba(X_test)[:, 1]  # Probability of positive class
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_prob = model.predict_proba(X_test)[:, 1]  # Probability of positive class

            result = calc_classification_metrics_benchmark(name, y_test, y_pred, y_pred_prob)
            results_list.append(result)

        # Write the results to an Excel file
        results_df = pd.DataFrame(results_list)
        results_df.to_excel(self.data_args.excel_path, index=False)




if __name__ == '__main__':
    parser = HfArgumentParser([MLBenchmarkDataArguments, ])
    args: MLBenchmarkDataArguments = parser.parse_args_into_dataclasses()[0]

    benchmark = MLBenchmark(args)
    benchmark.train_and_eval()
