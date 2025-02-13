import argparse
import asyncio
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, balanced_accuracy_score
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from transformers import BertForSequenceClassification, BertTokenizer
model_path = 'manueldeprada/FactCC'

tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model = model.to(device)


def batched_FactCC(text_l, summary_l, max_length=512):
    input_dict = tokenizer(text_l, summary_l, max_length=max_length, padding='max_length', truncation='only_first',
                           return_tensors='pt')
    input_dict = {key: value.to(device) for key, value in input_dict.items()}
    with torch.no_grad():
        logits = model(**input_dict).logits
        preds = logits.argmax(dim=1)

    return logits, preds


def main():
    tabular_strings = []
    GPT_descriptions = []
    for i in range(num_sample):
        row = my_data.iloc[i]
        if dataset == 'cr':
            Name = row['Name']
            Date = row['Date']
            label = row['binaryRating']
            # print(f'{i}: {Name} {Date} {label}')
            tabular_string = '\n '.join([f"{key}: {value}" for key, value in row.items() if key not in ['binaryRating',
                                                                                                       'GPT_description']])

        elif dataset == 'cr2':
            Corporation = row['Corporation']
            RatingDate = row['Rating Date']
            label = row['Binary Rating']
            # print(f'{i}: {Corporation} {RatingDate} {label}')
            tabular_string = '\n '.join([f"{key}: {value}" for key, value in row.items() if key not in ['Binary Rating',
                                                                                                       'GPT_description']])
        credit_str = 'Good credit' if label == 1 else 'Poor credit'
        tabular_string = tabular_string + f', Binary credit rating: {credit_str}'
        GPT_description = row['GPT_description']

        tabular_strings.append(tabular_string)
        GPT_descriptions.append(GPT_description)

    preds = []
    batch_size = 160
    for i in tqdm(range(0, len(tabular_strings), batch_size)):
        batch_texts = GPT_descriptions[i:i + batch_size]
        batch_claims = tabular_strings[i:i + batch_size]
        _, pred = batched_FactCC(batch_texts, batch_claims)
        preds.extend(pred.tolist())

    #"CORRECT": 0,
    # "INCORRECT": 1
    labels = np.zeros(len(tabular_strings))
    print(f"F1 micro: {f1_score(labels, preds, average='micro')}")
    print(f"Balanced accuracy: {balanced_accuracy_score(labels, preds)}")
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='My Dataset Parser')
    parser.add_argument('--dataset', type=str, default='cr', help='Dataset name')
    parser.add_argument('--data_type', type=str, default='train', help='Data type')
    args = parser.parse_args()
    dataset = args.dataset
    data_type = args.data_type

    data = pd.read_csv(f'./data/{dataset}/{data_type}(with_description_col).csv', header=0)
    num_sample = len(data)

    if dataset == 'cr':
        use_cols = ['binaryRating', 'Name', 'Date', 'currentRatio', 'quickRatio', 'cashRatio',
                    'daysOfSalesOutstanding', 'netProfitMargin', 'pretaxProfitMargin',
                    'grossProfitMargin', 'operatingProfitMargin', 'returnOnAssets',
                    'returnOnCapitalEmployed', 'returnOnEquity', 'assetTurnover',
                    'fixedAssetTurnover', 'debtEquityRatio', 'debtRatio',
                    'effectiveTaxRate', 'freeCashFlowOperatingCashFlowRatio',
                    'freeCashFlowPerShare', 'cashPerShare', 'companyEquityMultiplier',
                    'ebitPerRevenue', 'enterpriseValueMultiple',
                    'operatingCashFlowPerShare', 'operatingCashFlowSalesRatio',
                    'payablesTurnover', 'GPT_description']
    elif dataset == 'cr2':
        use_cols = ['Corporation', 'Rating Date', 'Binary Rating', 'Current Ratio',
                    'Long-term Debt / Capital', 'Debt/Equity Ratio', 'Gross Margin',
                    'Operating Margin', 'EBIT Margin', 'EBITDA Margin',
                    'Pre-Tax Profit Margin', 'Net Profit Margin', 'Asset Turnover',
                    'ROE - Return On Equity', 'Return On Tangible Equity',
                    'ROA - Return On Assets', 'ROI - Return On Investment',
                    'Operating Cash Flow Per Share', 'Free Cash Flow Per Share', 'GPT_description']

    my_data = data[use_cols]
    main()
