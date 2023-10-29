import asyncio
import time

import pandas as pd
from fastapi_poe.client import get_bot_response
from fastapi_poe.types import ProtocolMessage

dataset = 'cr2'
train_data = pd.read_csv(f'./data/{dataset}/train.csv', header=0)

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
    prompt_template = "上面是该企业的具体金融指标和信用评价，其中binaryRating是信用好坏的评价结果。要求：（1）根据这些数据，需要你对这个企业的总体信用状况进行“归纳总结！”（2）描述时尽量避免出现类似“2.324803263”的具体数值！（3）注意将驼峰形式的指标名称拆开来写！如companyEquityMultiplier应为company equity multiplier。（4）给出英语的“一段话”，200词左右。"

elif dataset == 'cr2':
    use_cols = ['Corporation', 'Rating Date', 'Binary Rating', 'Current Ratio',
                'Long-term Debt / Capital', 'Debt/Equity Ratio', 'Gross Margin',
                'Operating Margin', 'EBIT Margin', 'EBITDA Margin',
                'Pre-Tax Profit Margin', 'Net Profit Margin', 'Asset Turnover',
                'ROE - Return On Equity', 'Return On Tangible Equity',
                'ROA - Return On Assets', 'ROI - Return On Investment',
                'Operating Cash Flow Per Share', 'Free Cash Flow Per Share']
    prompt_template = "上面是该企业的具体金融指标和信用评价，其中Binary Rating是信用好坏的评价结果。要求：（1）根据这些数据，需要你对这个企业的总体信用状况进行“归纳总结！”（2）描述时尽量避免出现类似“2.324803263”的具体数值！（3）给出英语的“一段话”，200词左右。"

query_data = train_data[use_cols]
print()


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
    for i in range(537, 7805):
        print('*' * 60)
        row = query_data.iloc[i]
        row_string = ', '.join([f"{key}: {value}" for key, value in row.items()])
        if dataset == 'cr':
            Name = row['Name']
            Date = row['Date']
            print(f'{i}: {Name} {Date}')
        elif dataset == 'cr2':
            Corporation = row['Corporation']
            RatingDate = row['Rating Date']
            print(f'{i}: {Corporation} {RatingDate}')

        query = row_string + ' ' + prompt_template
        result = await chat(query)
        result = ''.join(result)
        print(result)

        if dataset == 'cr':
            df = pd.DataFrame({'Name': [Name], 'Date': Date, 'Query': query, 'GPT_description': [result]})
        elif dataset == 'cr2':
            df = pd.DataFrame(
                {'Corporation': [Corporation], 'Rating Date': RatingDate, 'Query': query, 'GPT_description': [result]})

        with open(f'{dataset}_description_part0.csv', 'a') as f:
            df.to_csv(f, header=f.tell() == 0, index=False)


asyncio.run(main())
