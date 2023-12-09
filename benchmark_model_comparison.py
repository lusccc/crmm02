import argparse
import os
from dataclasses import dataclass, field
from math import sqrt

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


@dataclass
class BenchmarkDataArguments(MultimodalDataArguments):
    excel_path: str = field(default=None, metadata={"help": "Path to the Excel file to save the results."})
    cat_encoder: str = field(default="ordinal", metadata={"help": "Encoding method for categorical features."})


class Benchmark:

    def __init__(self, data_args: BenchmarkDataArguments):
        self.data_args = data_args
        benchmark_data = MultimodalData(data_args)
        (self.train_data,
         self.val_data,
         self.test_data) = (benchmark_data.train_preprocessed.drop(['GPT_description'], axis=1),
                            benchmark_data.val_preprocessed.drop(['GPT_description'], axis=1),
                            benchmark_data.test_preprocessed.drop(['GPT_description'], axis=1))
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

        # Create a DataFrame to store all the results
        res_cols = ['Model', 'Acc', 'AUC', 'KS', 'G-mean', 'Type-I Acc', 'Type-II Acc']
        results_df = pd.DataFrame(columns=res_cols)

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

            acc, auc, ks, g_mean, type_1_acc, type_2_acc = self.evaluate_model(y_test, y_pred, y_pred_prob)
            results_df = pd.concat(
                [results_df, pd.DataFrame([[name, acc, auc, ks, g_mean, type_1_acc, type_2_acc]], columns=res_cols)]
            )

        # Write the results to an Excel file
        results_df.to_excel(self.data_args.excel_path, index=False)


    def evaluate_model(self, y_true, y_pred, y_pred_prob):
        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred_prob)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
        ks = max(tpr - fpr)
        cm = confusion_matrix(y_true, y_pred)
        type_1_acc = cm[0, 0] / cm[0, :].sum()
        type_2_acc = cm[1, 1] / cm[1, :].sum()
        g_mean = sqrt(type_1_acc * type_2_acc)

        return acc, auc, ks, g_mean, type_1_acc, type_2_acc


if __name__ == '__main__':
    parser = HfArgumentParser([BenchmarkDataArguments, ])
    args: BenchmarkDataArguments = parser.parse_args_into_dataclasses()[0]

    benchmark = Benchmark(args)
    benchmark.train_and_eval()
