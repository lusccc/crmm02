import argparse
import os
from math import sqrt

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


class Benchmark:

    def __init__(self, data_path, excel_path, dataset_split_strategy, train_years=None, test_years=None) -> None:
        super().__init__()
        self.data_path = data_path
        self.excel_path = excel_path
        self.dataset_split_strategy = dataset_split_strategy
        self.train_years = train_years
        self.test_years = test_years
        self.num_train_samples = None
        self.has_val = True

        self.train_data, self.val_data, self.test_data = self.load_and_preprocess_data()

    def load_and_preprocess_data(self):
        # @@@@ 1. read data
        all_df = pd.read_csv(os.path.join(self.data_path, 'all(with_description_col).csv'))

        if self.dataset_split_strategy == 'random':
            if self.num_train_samples is not None:
                train_df = all_df[:self.num_train_samples]
                if self.has_val:
                    # 剩余数据的10%作为验证集，然后其余的作为测试集
                    remaining_df = all_df[self.num_train_samples:]
                    val_df = remaining_df[:int(0.1 * len(remaining_df))]
                    test_df = remaining_df[int(0.1 * len(remaining_df)):]
                else:
                    # 剩余数据作为测试集
                    test_df = all_df[self.num_train_samples:]
                    val_df = None
            else:
                train_df = all_df[:int(0.8 * len(all_df))]
                test_df = all_df[int(0.8 * len(all_df)):]
                if self.has_val:
                    val_df = train_df[:int(0.1 * len(train_df))]
                    train_df = train_df[int(0.1 * len(train_df)):]
                else:
                    val_df = None
        elif self.dataset_split_strategy == 'rolling_window':
            date_col = 'Rating Date' if 'Rating Date' in all_df.columns else 'Date'
            all_df[date_col] = pd.to_datetime(all_df[date_col])
            all_df['Rating Year'] = all_df[date_col].dt.year.astype(int)
            train_df = all_df[all_df['Rating Year'].isin(self.train_years)]
            test_df = all_df[all_df['Rating Year'].isin(self.test_years)]
            if self.num_train_samples is not None:
                if self.has_val:
                    val_df = train_df[self.num_train_samples:]
                train_df = train_df[:self.num_train_samples]
            else:
                if self.has_val:
                    val_df = train_df[:int(0.1 * len(train_df))]
                    train_df = train_df[int(0.1 * len(train_df)):]
                else:
                    val_df = None
        else:
            raise ValueError(f'Unknown dataset_split_strategy: {self.dataset_split_strategy}')

        # @@@@ 2. preprocess
        cols_to_drop = ['Id', 'Rating', 'GPT_description']

        # to label encoding multi cat cols a time
        if 'cr2' in self.data_path:  # for cr2 dataset
            cat_cols = ['Rating Agency', 'Corporation', 'CIK', 'SIC Code', 'Sector', 'Ticker']
        else:  # for cr dataset
            cat_cols = ['Name', 'Symbol', 'Rating Agency Name', 'Sector', 'CIK']
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

        def process_dataframe(df, is_train):
            # 1) label encoding
            df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
            if is_train:
                oe.fit(df[cat_cols])
            df.loc[:, cat_cols] = oe.transform(df[cat_cols])

            # 2) date to int
            date_col = 'Rating Date' if 'Rating Date' in df.columns else 'Date'
            # Convert timestamp to datetime object
            df[date_col] = pd.to_datetime(df[date_col])
            # Convert datetime object to int
            df[date_col] = df[date_col].apply(lambda x: int(x.timestamp()))

            # 3) fill nan
            df.fillna(0, inplace=True)

            df = df.infer_objects()

            return df

        # Apply the function to the datasets
        train_df = process_dataframe(train_df, True)
        val_df = process_dataframe(val_df, False)
        test_df = process_dataframe(test_df, False)

        return train_df, val_df, test_df

    def train_and_eval(self):
        models = [
            ('LogR', LogisticRegression()),
            ('SVM', SVC(probability=True)),  # probability=True to enable predict_proba for AUC
            ('KNN', KNeighborsClassifier()),
            ('DT', DecisionTreeClassifier()),
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
            if name in ['SVM', 'KNN']:  # Use scaled data for SVM and KNN
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
            # results_df = results_df.append({
            #     'Model': name,
            #     'Acc': acc,
            #     'AUC': auc,
            #     'KS': ks,
            #     'G-mean': g_mean,
            #     'Type-I Acc': type_1_acc,
            #     'Type-II Acc': type_2_acc,
            # }, ignore_index=True)

        # Write the results to an Excel file
        results_df.to_excel(self.excel_path, index=False)

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
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--excel_path', type=str, default=None)
    parser.add_argument('--dataset_split_strategy', type=str, default=None)
    parser.add_argument('--train_years', type=str, default=None)
    parser.add_argument('--test_years', type=str, default=None)
    args = parser.parse_args()

    if isinstance(args.train_years, str):
        args.train_years = [int(m.strip()) for m in args.train_years.split(',')]
    if isinstance(args.test_years, str):
        args.test_years = [int(m.strip()) for m in args.test_years.split(',')]

    benchmark = Benchmark(args.data_path, args.excel_path, args.dataset_split_strategy, args.train_years,
                          args.test_years)
    benchmark.train_and_eval()
