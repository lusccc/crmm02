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
