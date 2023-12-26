import argparse
import asyncio
import time

import pandas as pd
from fastapi_poe.client import get_bot_response
from fastapi_poe.types import ProtocolMessage


async def chat(prompt):
    response_content = []
    message = ProtocolMessage(role="user", content=prompt)
    while True:
        try:
            async for response in get_bot_response(
                    messages=[message],
                    bot_name="GPT-3.5-Turbo",
                    api_key='AX7j2wByF-f2OY2i7L17p8rIAWyfraEr4MX8a6GFie4'
            ):
                response_content.append(response.text)
            break
        except Exception as e:
            print(f"Error occurred: {e}")
            print("Retrying in 5 seconds...")
            time.sleep(5)
    return response_content


# 存储对话历史
# history = []
# async def chat_h(prompt):
#     global history
#
#     # 添加用户输入到历史记录
#     history.append(ProtocolMessage(role="user", content=prompt))
#
#     response_content = []
#     async for response in get_bot_response(
#             messages=history,  # 传入全部对话历史
#             bot_name="GPT-3.5-Turbo",
#             api_key='AX7j2wByF-f2OY2i7L17p8rIAWyfraEr4MX8a6GFie4'
#     ):
#         response_content.append(response.text)
#
#         # 添加bot响应到历史记录
#         history.append(ProtocolMessage(role="bot", content=response.text))
#
#     return response_content


async def main():
    for i in range(num_sample):
        print('*' * 60)
        row = query_data.iloc[i]
        row_string = ', '.join([f"{key}: {value}" for key, value in row.items()])
        if dataset == 'cr':
            Name = row['Name']
            Date = row['Date']
            label = row['binaryRating']
            print(f'{i}: {Name} {Date} {label}')
        elif dataset == 'cr2':
            Corporation = row['Corporation']
            RatingDate = row['Rating Date']
            label = row['Binary Rating']
            print(f'{i}: {Corporation} {RatingDate} {label}')
        credit_str = 'Good credit' if label == 1 else 'Poor credit'
        query = row_string + ' ' + (prompt_template % credit_str)
        print(f'@@QUERY@@: {query}')
        print()
        result = await chat(query)
        result = ''.join(result)
        print(f'@@RESPONSE@@: {result}')

        if dataset == 'cr':
            df = pd.DataFrame({'Name': [Name], 'Date': Date, 'Query': query, 'GPT_description': [result]})
        elif dataset == 'cr2':
            df = pd.DataFrame(
                {'Corporation': [Corporation], 'Rating Date': RatingDate, 'Query': query, 'GPT_description': [result]})

        with open(f'./data/{dataset}/{data_type}_description.csv', 'a') as f:
            df.to_csv(f, header=f.tell() == 0, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='My Dataset Parser')
    parser.add_argument('--dataset', type=str, default='cr', help='Dataset name')
    parser.add_argument('--data_type', type=str, default='train', help='Data type')
    args = parser.parse_args()
    dataset = args.dataset
    data_type = args.data_type

    data = pd.read_csv(f'./data/{dataset}/{data_type}.csv', header=0)
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
                    'payablesTurnover']
        # prompt_template = "上面是该企业的具体金融指标和信用评价，其中binaryRating是信用好坏的评价结果。要求：（1）根据这些数据，需要你对这个企业的总体信用状况进行“归纳总结！”（2）描述时尽量避免出现类似“2.324803263”的具体数值！（3）注意将驼峰形式的指标名称拆开来写！如companyEquityMultiplier应为company equity multiplier。（4）给出英语的“一段话”，200词左右。"
        # prompt_template = "上面是该企业的具体金融指标结果，其评级结果是“%s”。要求你做：（1）根据这些数据，需要你对这个企业的总体信用状况进行“归纳总结！”（2）描述时尽量避免出现类似“2.324803263”的具体数值！（3）注意将驼峰形式的指标名称拆开来写！如companyEquityMultiplier应为company equity multiplier。（4）给出英语的“一段话”，200词左右。"
        prompt_template = 'The above is the specific financial indicator results of the company, and its rating result is "%s". You are required to do: (1) According to these data, you need to make a summary of the overall credit status of the enterprise! (2) Try to avoid specific values similar to "2.324803263" when describing! (3) Pay attention to the hump form of indicator name to write apart! For example, "companyEquityMultiplier" should be "company equity multiplier". (4) Give English "a paragraph", about 200 words.'
    elif dataset == 'cr2':
        use_cols = ['Corporation', 'Rating Date', 'Binary Rating', 'Current Ratio',
                    'Long-term Debt / Capital', 'Debt/Equity Ratio', 'Gross Margin',
                    'Operating Margin', 'EBIT Margin', 'EBITDA Margin',
                    'Pre-Tax Profit Margin', 'Net Profit Margin', 'Asset Turnover',
                    'ROE - Return On Equity', 'Return On Tangible Equity',
                    'ROA - Return On Assets', 'ROI - Return On Investment',
                    'Operating Cash Flow Per Share', 'Free Cash Flow Per Share']
        # prompt_template = "上面是该企业的具体金融指标结果，其评级结果是“%s”。要求你做：（1）根据这些数据，需要你对这个企业的总体信用状况进行“归纳总结！”（2）描述时尽量避免出现类似“2.324803263”的具体数值！（3）给出英语的“一段话”，200词左右。"
        prompt_template = 'The above is the specific financial indicator results of the company, and its rating result is "%s". You are required to do: (1) According to these data, you need to make a summary of the overall credit status of the enterprise! (2) Try to avoid specific values similar to "2.324803263" when describing! (3) Give the English "paragraph", about 200 words.'

    query_data = data[use_cols]
    asyncio.run(main())
