import argparse
import asyncio
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, balanced_accuracy_score
from tqdm import tqdm

# 引入评估指标相关的库
import nltk
from rouge_score import rouge_scorer  # 用于ROUGE
from bert_score import score as bert_score  # 用于BERTScore
from nltk.translate.bleu_score import sentence_bleu  # 用于BLEU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 下载nltk所需的资源
nltk.download('punkt')


def evaluate_nlg(tabular_strings, GPT_descriptions):
    """
    评估GPT生成的描述是否准确反映了tabular_string中的信息
    使用BLEU、ROUGE、BERTScore来评估
    :param tabular_strings: 表格数据转换的字符串列表
    :param GPT_descriptions: GPT生成的描述列表
    :return: BLEU, ROUGE, BERTScore 平均分
    """
    bleu_scores = []
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    bert_scores = []

    # 初始化ROUGE评分器
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # 遍历每一对tabular_string和GPT_description
    for tabular_string, GPT_description in tqdm(zip(tabular_strings, GPT_descriptions), total=len(tabular_strings), desc="Processing"):
        # 1. 计算 BLEU 分数
        reference = nltk.word_tokenize(tabular_string)
        candidate = nltk.word_tokenize(GPT_description)
        bleu = sentence_bleu([reference], candidate)
        bleu_scores.append(bleu)

        # 2. 计算 ROUGE 分数
        rouge_result = rouge.score(tabular_string, GPT_description)
        rouge_scores['rouge1'].append(rouge_result['rouge1'].fmeasure)
        rouge_scores['rouge2'].append(rouge_result['rouge2'].fmeasure)
        rouge_scores['rougeL'].append(rouge_result['rougeL'].fmeasure)

        # 3. 计算 BERTScore
        # P, R, F1 = bert_score([GPT_description], [tabular_string], lang="en", rescale_with_baseline=True)
        # bert_scores.append(F1.mean().item())

    # 计算BLEU、ROUGE、BERTScore的平均分
    avg_bleu = np.mean(bleu_scores)
    avg_rouge1 = np.mean(rouge_scores['rouge1'])
    avg_rouge2 = np.mean(rouge_scores['rouge2'])
    avg_rougeL = np.mean(rouge_scores['rougeL'])
    # avg_bert_score = np.mean(bert_scores)

    # return avg_bleu, avg_rouge1, avg_rouge2, avg_rougeL, avg_bert_score
    return avg_bleu, avg_rouge1, avg_rouge2, avg_rougeL, 0


def main():
    tabular_strings = []
    GPT_descriptions = []

    for i in range(num_sample):
        row = my_data.iloc[i]
        if dataset == 'cr':
            label = row['binaryRating']
            tabular_string = '\n '.join([f"{key}: {value}" for key, value in row.items() if key not in ['binaryRating',
                                                                                                        'GPT_description']])

        elif dataset == 'cr2':
            label = row['Binary Rating']
            tabular_string = '\n '.join([f"{key}: {value}" for key, value in row.items() if key not in ['Binary Rating',
                                                                                                        'GPT_description']])
        credit_str = 'Good credit' if label == 1 else 'Poor credit'
        tabular_string = tabular_string + f', Binary credit rating: {credit_str}'
        GPT_description = row['GPT_description']

        tabular_strings.append(tabular_string)
        GPT_descriptions.append(GPT_description)

    # 调用NLG评估函数
    avg_bleu, avg_rouge1, avg_rouge2, avg_rougeL, avg_bert_score = evaluate_nlg(tabular_strings, GPT_descriptions)

    # 输出评估结果
    print(f"BLEU 平均分: {avg_bleu * 100:.2f}%")
    print(f"ROUGE-1 平均分: {avg_rouge1 * 100:.2f}%")
    print(f"ROUGE-2 平均分: {avg_rouge2 * 100:.2f}%")
    print(f"ROUGE-L 平均分: {avg_rougeL * 100:.2f}%")
    print(f"BERTScore 平均分: {avg_bert_score * 100:.2f}%")


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