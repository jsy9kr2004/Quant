"""Feature engineering 및 데이터 처리를 위한 전역 변수 및 상수입니다.

이 모듈은 Quant Trading System 전체에서 feature 선택, 데이터 처리, 섹터 분류를 위해
사용되는 전역 상수를 정의합니다. 이 변수들은 주로 machine learning 모델 훈련,
feature engineering, backtesting 컴포넌트에서 사용됩니다.

이 모듈은 여러 유형의 전역 변수를 포함합니다:
    - Feature column 리스트: 사용할 financial metric 정의
    - 섹터 매핑: Industry 분류를 더 넓은 섹터로 매핑
    - 데이터 품질 지표: 희소하거나 문제가 있는 컬럼 식별

주요 변수 카테고리:
    1. **sparse_col_list**: 결측값이나 0이 많아 전처리 시 특별한 처리가 필요하거나
       제거가 필요할 수 있는 컬럼들.

    2. **ratio_col_list**: Fundamental 분석에서 일반적으로 사용되는
       financial ratio metric (P/E, ROE 등).

    3. **meaning_col_list**: 기업 재무 펀더멘털에 대한 의미 있는 정보를
       제공하는 절대값 metric (매출, 자산 등).

    4. **cal_timefeature_col_list**: 추세 분석 및 시간적 패턴을 위해
       기간별로 계산되는 time-series feature.

    5. **cal_ev_col_list**: 밸류에이션 분석을 위한 enterprise value 관련 metric.

    6. **sector_map**: 특정 industry 분류를 더 넓은 GICS 섹터 카테고리로
       매핑하는 dictionary (섹터 기반 분석용).

사용법:
    Feature 선택::

        from config.g_variables import ratio_col_list, meaning_col_list

        # 모델 훈련을 위해 ratio feature만 선택
        ratio_features = df[ratio_col_list]

        # 여러 feature 타입 결합
        all_features = ratio_col_list + meaning_col_list

    섹터 분류::

        from config.g_variables import sector_map

        # Industry를 sector로 매핑
        industry = "Software—Application"
        sector = sector_map.get(industry, "Unknown")
        print(f"{industry} -> {sector}")  # Software—Application -> Information Technology

    데이터 품질 처리::

        from config.g_variables import sparse_col_list

        # 데이터셋에서 희소 컬럼 제거
        df_clean = df.drop(columns=sparse_col_list, errors='ignore')

주의사항:
    - 컬럼명은 Financial Modeling Prep (FMP) API 규칙을 따릅니다
    - 일부 컬럼은 'Ydiff_' (전년 대비) 또는 'Qdiff_' (전분기 대비) 같은
      접두사가 있는 변형이 있습니다
    - 섹터 매핑은 일관성 없는 API 응답을 처리하기 위한 변형을 포함합니다
      (예: "Software - Application"과 "Software—Application" 모두)
    - 파일 끝의 주석 처리된 코드는 과거 예제와 대체 데이터 소스를
      보여줍니다 (참고용으로 보존)

참고:
    - data_collector: API 데이터 수집에 이 변수들을 사용
    - ml_model: 모델 훈련에 feature 리스트를 사용
    - feature_engineering: Feature 선택 및 변환에 사용

TODO:
    - 더 쉬운 유지보수를 위해 이것들을 YAML configuration으로 이동 고려
    - 컬럼명이 현재 FMP API schema와 일치하는지 검증 추가
    - 다른 변수 카테고리를 위한 별도 파일 생성
"""

# ==============================================================================
# SPARSE COLUMNS - 높은 결측값 비율을 가진 컬럼들
# ==============================================================================
# 이 컬럼들은 많은 기업에서 결측값이나 0 값을 자주 포함합니다.
# 전처리 시 특별한 대체(imputation), 제거, 또는 별도 처리가 필요할 수 있습니다.

sparse_col_list = [
    'Ydiff_dcf',                        # Year-over-year change in DCF
    'Ydiff_preferredStock',             # Year-over-year change in preferred stock
    'Qdiff_otherLiabilities',           # Quarter-over-quarter change in other liabilities
    'Ydiff_capitalLeaseObligations',    # Year-over-year change in capital lease obligations
    'Qdiff_dcf',                        # Quarter-over-quarter change in DCF
    'Ydiff_otherLiabilities',           # Year-over-year change in other liabilities
    'Ydiff_deferredRevenueNonCurrent',  # Year-over-year change in deferred revenue
    'Qdiff_otherAssets',                # Quarter-over-quarter change in other assets
    'Qdiff_deferredRevenueNonCurrent',  # Quarter-over-quarter change in deferred revenue
    'Ydiff_sellingAndMarketingExpenses',# Year-over-year change in S&M expenses
    'Ydiff_researchAndDdevelopementToRevenue',  # Year-over-year change in R&D ratio
    'dcf',                              # Discounted Cash Flow valuation
    'Ydiff_otherAssets',                # Year-over-year change in other assets
    'OverMC_dcf',                       # DCF over market cap ratio
    'Qdiff_preferredStock',             # Quarter-over-quarter change in preferred stock
    'Qdiff_capitalLeaseObligations'     # Quarter-over-quarter change in capital leases
]

# ==============================================================================
# RATIO COLUMNS - Financial Ratio 및 백분율 Metric
# ==============================================================================
# 이 컬럼들은 이미 정규화되어 다른 규모의 기업간 비교가 가능한 ratio 기반 metric을 포함합니다.
# Fundamental 분석에서 일반적으로 사용되며 절대값보다 안정적인 경우가 많습니다.

ratio_col_list = [
    # Valuation Ratios
    "interestCoverage",                 # EBIT / Interest Expense
    "dividendYield",                    # Annual Dividend / Stock Price
    "peRatio",                          # Price / Earnings Per Share
    "pbRatio",                          # Price / Book Value Per Share
    "ptbRatio",                         # Price to Tangible Book ratio
    "priceToSalesRatio",                # Price / Sales Per Share
    "pfcfRatio",                        # Price / Free Cash Flow Per Share
    "pocfratio",                        # Price / Operating Cash Flow Per Share
    "enterpriseValueOverEBITDA",        # Enterprise Value / EBITDA

    # Liquidity Ratios
    "currentRatio",                     # Current Assets / Current Liabilities

    # Efficiency Ratios
    "inventoryTurnover",                # Cost of Goods Sold / Average Inventory
    "daysPayablesOutstanding",          # Days to pay suppliers
    "daysOfInventoryOnHand",            # Days of inventory on hand
    "payablesTurnover",                 # Cost of Goods Sold / Average Payables
    "daysSalesOutstanding",             # Days to collect receivables
    "receivablesTurnover",              # Revenue / Average Receivables

    # Profitability Ratios
    "roe",                              # Return on Equity
    "roic",                             # Return on Invested Capital
    "returnOnTangibleAssets",           # Net Income / Tangible Assets
    "earningsYield",                    # Earnings Per Share / Price
    "netIncomeRatio",                   # Net Income / Revenue
    "grossProfitRatio",                 # Gross Profit / Revenue
    "operatingIncomeRatio",             # Operating Income / Revenue
    "incomeBeforeTaxRatio",             # Income Before Tax / Revenue
    "ebitdaratio",                      # EBITDA / Revenue

    # Leverage Ratios
    "debtToAssets",                     # Total Debt / Total Assets
    "debtToEquity",                     # Total Debt / Total Equity
    "netDebtToEBITDA",                  # Net Debt / EBITDA
    "incomeQuality",                    # Cash Flow / Net Income

    # Per Share Metrics
    "eps",                              # Earnings Per Share
    "epsdiluted",                       # Diluted Earnings Per Share
    "revenuePerShare",                  # Revenue / Shares Outstanding
    "netIncomePerShare",                # Net Income / Shares Outstanding
    "freeCashFlowPerShare",             # Free Cash Flow / Shares Outstanding
    "operatingCashFlowPerShare",        # Operating Cash Flow / Shares Outstanding
    "cashPerShare",                     # Cash / Shares Outstanding
    "bookValuePerShare",                # Book Value / Shares Outstanding
    "tangibleBookValuePerShare",        # Tangible Book Value / Shares Outstanding
    "shareholdersEquityPerShare",       # Shareholders Equity / Shares Outstanding
    "interestDebtPerShare",             # Interest-Bearing Debt / Shares Outstanding
    "capexPerShare",                    # Capital Expenditure / Shares Outstanding

    # Capital Efficiency Metrics
    "capexToDepreciation",              # CapEx / Depreciation
    "capexToRevenue",                   # CapEx / Revenue
    "capexToOperatingCashFlow",         # CapEx / Operating Cash Flow
    "stockBasedCompensationToRevenue",  # Stock Comp / Revenue
    "salesGeneralAndAdministrativeToRevenue",  # SG&A / Revenue
    "researchAndDdevelopementToRevenue",# R&D / Revenue
    "intangiblesToTotalAssets",         # Intangibles / Total Assets

    # Valuation Metrics
    "dcf",                              # Discounted Cash Flow
    "evToOperatingCashFlow",            # EV / Operating Cash Flow
    "evToFreeCashFlow",                 # EV / Free Cash Flow
    "evToSales",                        # EV / Sales
    "freeCashFlowYield",                # Free Cash Flow / Market Cap
    "grahamNetNet",                     # Graham Net-Net valuation
    "grahamNumber",                     # Graham Number valuation

    # Shareholder Returns
    "payoutRatio",                      # Dividends / Net Income

    # Growth Metrics (these contain "Growth" in name)
    "dividendsperShareGrowth",          # Dividend growth rate
    "netIncomeGrowth",                  # Net income growth rate
    "epsgrowth",                        # EPS growth rate
    "epsdilutedGrowth",                 # Diluted EPS growth rate
    "revenueGrowth",                    # Revenue growth rate
    "debtGrowth",                       # Debt growth rate
    "operatingCashFlowGrowth",          # Operating CF growth rate
    "ebitgrowth",                       # EBIT growth rate
    "operatingIncomeGrowth",            # Operating income growth rate
    "assetGrowth",                      # Asset growth rate
    "freeCashFlowGrowth",               # Free cash flow growth rate
    "sgaexpensesGrowth",                # SG&A expenses growth rate
    "receivablesGrowth",                # Receivables growth rate
    "grossProfitGrowth",                # Gross profit growth rate
    "weightedAverageSharesGrowth",      # Share count growth
    "weightedAverageSharesDilutedGrowth",  # Diluted share count growth
    "bookValueperShareGrowth",          # Book value per share growth
    "inventoryGrowth",                  # Inventory growth rate
    "rdexpenseGrowth",                  # R&D expense growth rate

    # Multi-Year Growth Metrics
    "tenYDividendperShareGrowthPerShare",   # 10-year dividend CAGR
    "threeYDividendperShareGrowthPerShare", # 3-year dividend CAGR
    "fiveYDividendperShareGrowthPerShare",  # 5-year dividend CAGR
    "threeYOperatingCFGrowthPerShare",      # 3-year operating CF CAGR
    "fiveYOperatingCFGrowthPerShare",       # 5-year operating CF CAGR
    "tenYOperatingCFGrowthPerShare",        # 10-year operating CF CAGR
    "threeYRevenueGrowthPerShare",          # 3-year revenue CAGR
    "fiveYRevenueGrowthPerShare",           # 5-year revenue CAGR
    "tenYRevenueGrowthPerShare",            # 10-year revenue CAGR
    "threeYShareholdersEquityGrowthPerShare",   # 3-year equity CAGR
    "fiveYShareholdersEquityGrowthPerShare",    # 5-year equity CAGR
    "tenYShareholdersEquityGrowthPerShare",     # 10-year equity CAGR
    "threeYNetIncomeGrowthPerShare",        # 3-year net income CAGR
    "fiveYNetIncomeGrowthPerShare",         # 5-year net income CAGR
    "tenYNetIncomeGrowthPerShare",          # 10-year net income CAGR

    # Absolute Value Metrics (also in ratio list for historical reasons)
    "freeCashFlow",                     # Free Cash Flow (absolute)
    "operatingCashFlow",                # Operating Cash Flow (absolute)
    "grossProfit",                      # Gross Profit (absolute)
    "ebitda",                           # EBITDA (absolute)
    "netDebt",                          # Net Debt (absolute)
    "investedCapital",                  # Invested Capital (absolute)
    "stockBasedCompensation",           # Stock-Based Compensation (absolute)
]

# ==============================================================================
# MEANINGFUL COLUMNS - 비즈니스 중요도가 있는 절대값 Metric
# ==============================================================================
# 이 컬럼들은 기업의 재무 상태와 운영에 대한 의미 있는 정보를 제공하는
# 절대값(달러 단위)을 포함합니다. 종합적인 financial 분석을 위해 ratio와 함께 사용됩니다.

meaning_col_list = [
    # Cash Flow Statement Items
    "commonStockRepurchased",           # Share buyback amount
    "netCashProvidedByOperatingActivities",  # Operating cash flow
    "netCashUsedForInvestingActivites", # Investing cash flow
    "netCashUsedProvidedByFinancingActivities",  # Financing cash flow
    "freeCashFlow",                     # Free cash flow (OCF - CapEx)
    "operatingCashFlow",                # Operating cash flow
    "netChangeInCash",                  # Net change in cash position
    "changeInWorkingCapital",           # Change in working capital
    "capitalExpenditure",               # Capital expenditures
    "investmentsInPropertyPlantAndEquipment",  # PP&E investments
    "acquisitionsNet",                  # Net acquisition spending
    "purchasesOfInvestments",           # Investment purchases
    "otherInvestingActivites",          # Other investing activities
    "otherNonCashItems",                # Other non-cash items
    "depreciationAndAmortization",      # D&A expense
    "otherWorkingCapital",              # Other working capital changes
    "effectOfForexChangesOnCash",       # FX impact on cash

    # Balance Sheet - Assets
    "totalAssets",                      # Total assets
    "totalCurrentAssets",               # Total current assets
    "totalNonCurrentAssets",            # Total non-current assets
    "cashAndCashEquivalents",           # Cash and equivalents
    "cashAndShortTermInvestments",      # Cash and short-term investments
    "shortTermInvestments",             # Short-term investments
    "totalInvestments",                 # Total investments
    "longTermInvestments",              # Long-term investments
    "netReceivables",                   # Net receivables
    "accountsReceivables",              # Accounts receivable
    "inventory",                        # Inventory
    "propertyPlantEquipmentNet",        # Net PP&E
    "goodwillAndIntangibleAssets",      # Goodwill and intangibles
    "intangibleAssets",                 # Intangible assets
    "otherAssets",                      # Other assets
    "otherCurrentAssets",               # Other current assets
    "otherNonCurrentAssets",            # Other non-current assets
    "deferredRevenue",                  # Deferred revenue (liability side)
    "deferredRevenueNonCurrent",        # Long-term deferred revenue
    "deferredIncomeTax",                # Deferred tax assets
    "taxAssets",                        # Tax assets

    # Balance Sheet - Liabilities
    "totalLiabilities",                 # Total liabilities
    "totalCurrentLiabilities",          # Total current liabilities
    "totalNonCurrentLiabilities",       # Total non-current liabilities
    "totalLiabilitiesAndStockholdersEquity",  # Total L&E
    "totalLiabilitiesAndTotalEquity",   # Total L&E (alternative name)
    "totalDebt",                        # Total debt
    "shortTermDebt",                    # Short-term debt
    "longTermDebt",                     # Long-term debt
    "accountsPayables",                 # Accounts payable
    "accountPayables",                  # Accounts payable (alternative name)
    "taxPayables",                      # Tax payables
    "otherLiabilities",                 # Other liabilities
    "otherCurrentLiabilities",          # Other current liabilities
    "otherNonCurrentLiabilities",       # Other non-current liabilities
    "capitalLeaseObligations",          # Capital lease obligations
    "deferredTaxLiabilitiesNonCurrent", # Deferred tax liabilities

    # Balance Sheet - Equity
    "totalEquity",                      # Total equity
    "totalStockholdersEquity",          # Total shareholders equity
    "commonStock",                      # Common stock
    "retainedEarnings",                 # Retained earnings
    "accumulatedOtherComprehensiveIncomeLoss",  # AOCI
    "othertotalStockholdersEquity",     # Other equity items
    "minorityInterest",                 # Minority interest

    # Income Statement
    "revenue",                          # Total revenue
    "costOfRevenue",                    # Cost of revenue/COGS
    "grossProfit",                      # Gross profit
    "operatingExpenses",                # Total operating expenses
    "costAndExpenses",                  # Total costs and expenses
    "sellingGeneralAndAdministrativeExpenses",  # SG&A expenses
    "sellingAndMarketingExpenses",      # Sales and marketing
    "generalAndAdministrativeExpenses", # G&A expenses
    "researchAndDevelopmentExpenses",   # R&D expenses
    "operatingIncome",                  # Operating income (EBIT)
    "ebitda",                           # EBITDA
    "interestExpense",                  # Interest expense
    "interestIncome",                   # Interest income
    "totalOtherIncomeExpensesNet",      # Other income/expenses
    "otherExpenses",                    # Other expenses
    "incomeBeforeTax",                  # Pre-tax income
    "incomeTaxExpense",                 # Income tax expense
    "netIncome",                        # Net income

    # Valuation and Per-Share Metrics
    "enterpriseValue",                  # Enterprise value
    "dcf",                              # Discounted cash flow value
    "stockBasedCompensation",           # Stock-based compensation
    "workingCapital",                   # Working capital
    "netCurrentAssetValue",             # NCAV (Graham metric)
    "tangibleAssetValue",               # Tangible asset value
    "investedCapital",                  # Invested capital
    "netDebt",                          # Net debt (debt - cash)

    # Averages (used in ratio calculations)
    "averageInventory",                 # Average inventory
    "averagePayables",                  # Average payables
    "averageReceivables",               # Average receivables

    # Per Share Values (absolute amounts per share)
    "eps",                              # Earnings per share
    "epsdiluted",                       # Diluted EPS
    "revenuePerShare",                  # Revenue per share
    "netIncomePerShare",                # Net income per share
    "freeCashFlowPerShare",             # FCF per share
    "operatingCashFlowPerShare",        # OCF per share
    "cashPerShare",                     # Cash per share
    "bookValuePerShare",                # Book value per share
    "tangibleBookValuePerShare",        # Tangible book value per share
    "shareholdersEquityPerShare",       # Equity per share
    "interestDebtPerShare",             # Interest debt per share
    "capexPerShare",                    # CapEx per share

    # Ratios (also included for convenience)
    "interestCoverage",                 # Interest coverage ratio
    "dividendYield",                    # Dividend yield
    "inventoryTurnover",                # Inventory turnover
    "daysPayablesOutstanding",          # DPO
    "stockBasedCompensationToRevenue",  # SBC / Revenue
    "capexToDepreciation",              # CapEx / D&A
    "currentRatio",                     # Current ratio
    "daysOfInventoryOnHand",            # DOH
    "payablesTurnover",                 # Payables turnover
    "grahamNetNet",                     # Graham net-net
    "capexToRevenue",                   # CapEx / Revenue
    "netDebtToEBITDA",                  # Net debt / EBITDA
    "receivablesTurnover",              # Receivables turnover
    "capexToOperatingCashFlow",         # CapEx / OCF
    "evToOperatingCashFlow",            # EV / OCF
    "evToFreeCashFlow",                 # EV / FCF
    "debtToAssets",                     # Debt / Assets
    "peRatio",                          # P/E ratio
    "enterpriseValueOverEBITDA",        # EV / EBITDA
    "pfcfRatio",                        # P/FCF
    "pocfratio",                        # P/OCF
    "daysSalesOutstanding",             # DSO
    "incomeQuality",                    # Income quality
    "evToSales",                        # EV / Sales
    "grahamNumber",                     # Graham number
    "priceToSalesRatio",                # P/S ratio
    "pbRatio",                          # P/B ratio
    "ptbRatio",                         # P/TB ratio
    "roic",                             # Return on invested capital
    "freeCashFlowYield",                # FCF yield
    "roe",                              # Return on equity
    "returnOnTangibleAssets",           # ROTA
    "earningsYield",                    # Earnings yield
    "debtToEquity",                     # D/E ratio
    "payoutRatio",                      # Payout ratio
    "salesGeneralAndAdministrativeToRevenue",  # SG&A / Revenue
    "intangiblesToTotalAssets",         # Intangibles / Assets
    "ebitdaratio",                      # EBITDA margin
    "dividendsperShareGrowth",          # Dividend growth
    "netIncomeGrowth",                  # Net income growth
    "epsgrowth",                        # EPS growth
    "epsdilutedGrowth",                 # Diluted EPS growth
    "revenueGrowth",                    # Revenue growth
    "grossProfitRatio",                 # Gross margin
    "debtGrowth",                       # Debt growth
    "tenYDividendperShareGrowthPerShare",  # 10Y dividend CAGR
    "netIncomeRatio",                   # Net margin
    "incomeBeforeTaxRatio",             # Pre-tax margin
    "operatingCashFlowGrowth",          # OCF growth
    "ebitgrowth",                       # EBIT growth
    "operatingIncomeGrowth",            # Operating income growth
    "threeYDividendperShareGrowthPerShare",  # 3Y dividend CAGR
    "assetGrowth",                      # Asset growth
    "freeCashFlowGrowth",               # FCF growth
    "sgaexpensesGrowth",                # SG&A growth
    "fiveYDividendperShareGrowthPerShare",  # 5Y dividend CAGR
    "receivablesGrowth",                # Receivables growth
    "fiveYRevenueGrowthPerShare",       # 5Y revenue CAGR
    "threeYOperatingCFGrowthPerShare",  # 3Y OCF CAGR
    "grossProfitGrowth",                # Gross profit growth
    "operatingIncomeRatio",             # Operating margin
    "threeYShareholdersEquityGrowthPerShare",  # 3Y equity CAGR
    "fiveYShareholdersEquityGrowthPerShare",   # 5Y equity CAGR
    "fiveYOperatingCFGrowthPerShare",   # 5Y OCF CAGR
    "threeYRevenueGrowthPerShare",      # 3Y revenue CAGR
    "researchAndDdevelopementToRevenue",# R&D / Revenue
    "threeYNetIncomeGrowthPerShare",    # 3Y net income CAGR
    "tenYOperatingCFGrowthPerShare",    # 10Y OCF CAGR
    "tenYRevenueGrowthPerShare",        # 10Y revenue CAGR
    "tenYShareholdersEquityGrowthPerShare",  # 10Y equity CAGR
    "interestExpense",                  # Interest expense
    "tenYNetIncomeGrowthPerShare",      # 10Y net income CAGR
    "weightedAverageSharesGrowth",      # Share count growth
    "weightedAverageSharesDilutedGrowth",  # Diluted share growth
    "fiveYNetIncomeGrowthPerShare",     # 5Y net income CAGR
    "bookValueperShareGrowth",          # Book value growth
    "inventoryGrowth",                  # Inventory growth
    "rdexpenseGrowth",                  # R&D growth

    # Cash Position Metrics
    "cashAtBeginningOfPeriod",          # Beginning cash
    "cashAtEndOfPeriod",                # Ending cash
]

# ==============================================================================
# TIME FEATURE COLUMNS - Time Series 분석을 위한 Feature
# ==============================================================================
# 이 컬럼들은 이동평균, 추세, 모멘텀 지표, 기타 시간적 패턴 같은
# 시간 기반 feature를 계산하기 위해 선택되었습니다.

cal_timefeature_col_list = [
    "bookValuePerShare",
    "capexPerShare",
    "cashPerShare",
    # "changeInWorkingCapital",         # Commented out - too volatile
    "currentRatio",
    "daysSalesOutstanding",
    "dcf",
    "dividendsperShareGrowth",
    # "ebitdaratio",                    # Commented out - covered by EV metrics
    "enterpriseValue",
    "enterpriseValueOverEBITDA",
    "evToFreeCashFlow",
    "evToOperatingCashFlow",
    "evToSales",
    "freeCashFlowPerShare",
    "grahamNetNet",
    "grahamNumber",
    # "grossProfitRatio",               # Commented out - margin is stable
    "interestDebtPerShare",
    "inventoryTurnover",
    # "netCashProvidedByOperatingActivities",  # Commented out - use per share
    # "netCashUsedForInvestingActivites",      # Commented out - use per share
    # "netChangeInCash",                # Commented out - too volatile
    "netDebtToEBITDA",
    # "netIncomeRatio",                 # Commented out - margin is stable
    # "operatingCashFlowPerShare",      # Commented out - redundant with OCF
    # "operatingIncomeRatio",           # Commented out - margin is stable
    "payablesTurnover",
    "pbRatio",
    "peRatio",
    "pfcfRatio",
    "pocfratio",
    "priceToSalesRatio",
    "ptbRatio",
    "roe",
    "roic",
    "salesGeneralAndAdministrativeToRevenue",
    "shareholdersEquityPerShare"
]

# ==============================================================================
# ENTERPRISE VALUE 계산 컬럼
# ==============================================================================
# 이 컬럼들은 enterprise value 계산 및 EV 기반 ratio에 사용됩니다.
# Enterprise Value = Market Cap + Net Debt
# 일반적인 EV ratio: EV/EBITDA, EV/Sales, EV/OCF, EV/FCF

cal_ev_col_list = [
    "freeCashFlow",     # Used for EV/FCF ratio
    "ebitda"            # Used for EV/EBITDA ratio
]

# Additional metrics available for EV calculations (commented for reference):
# "netdebt"             # Net Debt = Total Debt - Cash (EV = Market Cap + Net Debt)
# "operatingCashflow"   # Used for EV/OCF ratio
# "revenues"            # Used for EV/Sales ratio

# ==============================================================================
# SECTOR MAPPING - Industry에서 GICS 섹터 분류로의 매핑
# ==============================================================================
# 특정 industry 분류를 더 넓은 GICS (Global Industry Classification Standard) 섹터로
# 매핑합니다. 이 매핑은 다른 데이터 소스 및 API 버전의 여러 명명 변형을 처리합니다.
#
# GICS 섹터 (총 11개):
#   1. Energy
#   2. Materials
#   3. Industrials
#   4. Consumer Discretionary
#   5. Consumer Staples
#   6. Healthcare
#   7. Financials
#   8. Information Technology
#   9. Communication Services
#  10. Utilities
#  11. Real Estate
#
# 주의: 일부 industry는 일관성 없는 API 응답을 처리하기 위해 다른 명명 규칙으로
#       여러 번 나타납니다 (예: "Software - Application" vs "Software—Application").

sector_map = {
    # Financials Sector
    "Asset Management": "Financials",
    "Asset Management - Income": "Financials",
    "Asset Management - Global": "Financials",
    "Banks—Regional": "Financials",
    "Banks - Regional": "Financials",
    "Banks—Diversified": "Financials",
    "Banks - Diversified": "Financials",
    "Capital Markets": "Financials",
    "Credit Services": "Financials",
    "Insurance—Property & Casualty": "Financials",
    "Insurance - Property & Casualty": "Financials",
    "Insurance—Life": "Financials",
    "Insurance - Life": "Financials",
    "Insurance—Specialty": "Financials",
    "Insurance - Specialty": "Financials",
    "Insurance—Diversified": "Financials",
    "Insurance - Diversified": "Financials",
    "Insurance—Reinsurance": "Financials",
    "Insurance - Reinsurance": "Financials",
    "Insurance Brokers": "Financials",
    "Mortgage Finance": "Financials",
    "Financial Data & Stock Exchanges": "Financials",
    "Financial - Data & Stock Exchanges": "Financials",
    "Financial - Capital Markets": "Financials",
    "Financial - Credit Services": "Financials",
    "Financial - Conglomerates": "Financials",
    "Financial - Mortgages": "Financials",
    "Investment - Banking & Investment Services": "Financials",
    "Shell Companies": "Financials",

    # Healthcare Sector
    "Biotechnology": "Healthcare",
    "Medical Instruments & Supplies": "Healthcare",
    "Medical Devices": "Healthcare",
    "Diagnostics & Research": "Healthcare",
    "Medical Care Facilities": "Healthcare",
    "Drug Manufacturers—Specialty & Generic": "Healthcare",
    "Drug Manufacturers - Specialty & Generic": "Healthcare",
    "Drug Manufacturers—General": "Healthcare",
    "Drug Manufacturers - General": "Healthcare",
    "Healthcare Plans": "Healthcare",
    "Medical Distribution": "Healthcare",
    "Health Information Services": "Healthcare",
    "Pharmaceutical Retailers": "Healthcare",
    "Healthcare": "Healthcare",
    "Medical - Pharmaceuticals": "Healthcare",
    "Medical - Distribution": "Healthcare",
    "Medical - Healthcare Information Services": "Healthcare",
    "Medical - Healthcare Plans": "Healthcare",
    "Medical - Diagnostics & Research": "Healthcare",
    "Medical - Instruments & Supplies": "Healthcare",
    "Medical - Care Facilities": "Healthcare",
    "Medical - Devices": "Healthcare",

    # Information Technology Sector
    "Software—Application": "Information Technology",
    "Software - Application": "Information Technology",
    "Software—Infrastructure": "Information Technology",
    "Software - Infrastructure": "Information Technology",
    "Semiconductors": "Information Technology",
    "Information Technology Services": "Information Technology",
    "Communication Equipment": "Information Technology",
    "Semiconductor Equipment & Materials": "Information Technology",
    "Electronic Components": "Information Technology",
    "Scientific & Technical Instruments": "Information Technology",
    "Computer Hardware": "Information Technology",
    "Business Equipment & Supplies": "Information Technology",
    "Electronics & Computer Distribution": "Information Technology",
    "Security & Protection Services": "Information Technology",
    "Technology Distributors": "Information Technology",
    "Hardware, Equipment & Parts": "Information Technology",

    # Communication Services Sector
    "Telecom Services": "Communication Services",
    "Telecommunications Services": "Communication Services",
    "Internet Content & Information": "Communication Services",
    "Entertainment": "Communication Services",
    "Publishing": "Communication Services",
    "Electronic Gaming & Multimedia": "Communication Services",
    "Broadcasting": "Communication Services",
    "CATV Systems": "Communication Services",

    # Industrials Sector
    "Specialty Industrial Machinery": "Industrials",
    "Aerospace & Defense": "Industrials",
    "Building Products & Equipment": "Industrials",
    "Specialty Business Services": "Industrials",
    "Engineering & Construction": "Industrials",
    "Rental & Leasing Services": "Industrials",
    "Farm & Heavy Construction Machinery": "Industrials",
    "Integrated Freight & Logistics": "Industrials",
    "Staffing & Employment Services": "Industrials",
    "Railroads": "Industrials",
    "Industrial Distribution": "Industrials",
    "Consulting Services": "Industrials",
    "Electrical Equipment & Parts": "Industrials",
    "Marine Shipping": "Industrials",
    "Metal Fabrication": "Industrials",
    "Trucking": "Industrials",
    "Waste Management": "Industrials",  # Some classify as Materials
    "Pollution & Treatment Controls": "Industrials",
    "Airlines": "Industrials",  # Some classify as Consumer Discretionary
    "Industrial - Distribution": "Industrials",
    "Industrial Materials": "Industrials",
    "Industrial - Machinery": "Industrials",
    "Industrial - Infrastructure Operations": "Industrials",
    "Agricultural - Machinery": "Industrials",
    "Manufacturing - Metal Fabrication": "Industrials",
    "Manufacturing - Tools & Accessories": "Industrials",
    "Construction": "Industrials",
    "Airlines, Airports & Air Services": "Industrials",

    # Consumer Discretionary Sector
    "Apparel Retail": "Consumer Discretionary",
    "Restaurants": "Consumer Discretionary",
    "Specialty Retail": "Consumer Discretionary",
    "Furnishings, Fixtures & Appliances": "Consumer Discretionary",
    "Airlines": "Consumer Discretionary",  # Duplicate - some classify here
    "Residential Construction": "Consumer Discretionary",
    "Auto & Truck Dealerships": "Consumer Discretionary",
    "Footwear & Accessories": "Consumer Discretionary",
    "Leisure": "Consumer Discretionary",
    "Resorts & Casinos": "Consumer Discretionary",
    "Advertising Agencies": "Consumer Discretionary",
    "Education & Training Services": "Consumer Discretionary",
    "Apparel Manufacturing": "Consumer Discretionary",
    "Travel Services": "Consumer Discretionary",
    "Internet Retail": "Consumer Discretionary",
    "Personal Services": "Consumer Discretionary",
    "Discount Stores": "Consumer Discretionary",
    "Auto Manufacturers": "Consumer Discretionary",
    "Tools & Accessories": "Consumer Discretionary",
    "Luxury Goods": "Consumer Discretionary",
    "Department Stores": "Consumer Discretionary",
    "Recreational Vehicles": "Consumer Discretionary",
    "Gambling": "Consumer Discretionary",
    "Home Improvement Retail": "Consumer Discretionary",
    "Lodging": "Consumer Discretionary",
    "Consumer Electronics": "Consumer Discretionary",
    "Auto Parts": "Consumer Discretionary",
    "Gambling, Resorts & Casinos": "Consumer Discretionary",
    "Personal Products & Services": "Consumer Discretionary",
    "Apparel - Footwear & Accessories": "Consumer Discretionary",
    "Auto - Dealerships": "Consumer Discretionary",
    "Auto - Recreational Vehicles": "Consumer Discretionary",
    "Auto - Manufacturers": "Consumer Discretionary",
    "Auto - Parts": "Consumer Discretionary",
    "Apparel - Retail": "Consumer Discretionary",
    "Apparel - Manufacturers": "Consumer Discretionary",
    "Home Improvement": "Consumer Discretionary",
    "Travel Lodging": "Consumer Discretionary",

    # Consumer Staples Sector
    "Packaged Foods": "Consumer Staples",
    "Household & Personal Products": "Consumer Staples",
    "Beverages—Non-Alcoholic": "Consumer Staples",
    "Beverages - Non-Alcoholic": "Consumer Staples",
    "Beverages—Brewers": "Consumer Staples",
    "Beverages - Brewers": "Consumer Staples",
    "Food Distribution": "Consumer Staples",
    "Beverages—Wineries & Distilleries": "Consumer Staples",
    "Beverages - Wineries & Distilleries": "Consumer Staples",
    "Confectioners": "Consumer Staples",
    "Grocery Stores": "Consumer Staples",
    "Farm Products": "Consumer Staples",
    "Tobacco": "Consumer Staples",
    "Agricultural Farm Products": "Consumer Staples",
    "Food Confectioners": "Consumer Staples",
    "Beverages - Alcoholic": "Consumer Staples",

    # Materials Sector
    "Specialty Chemicals": "Materials",
    "Packaging & Containers": "Materials",
    "Steel": "Materials",
    "Other Industrial Metals & Mining": "Materials",
    "Chemicals": "Materials",
    "Agricultural Inputs": "Materials",
    "Paper & Paper Products": "Materials",
    "Building Materials": "Materials",
    "Aluminum": "Materials",
    "Silver": "Materials",
    "Other Precious Metals & Mining": "Materials",
    "Copper": "Materials",
    "Lumber & Wood Production": "Materials",
    "Gold": "Materials",
    "Other Precious Metals": "Materials",
    "Chemicals - Specialty": "Materials",
    "Construction Materials": "Materials",
    "Paper, Lumber & Forest Products": "Materials",

    # Energy Sector
    "Oil & Gas E&P": "Energy",
    "Oil & Gas Exploration & Production": "Energy",
    "Oil & Gas Midstream": "Energy",
    "Oil & Gas Equipment & Services": "Energy",
    "Oil & Gas Integrated": "Energy",
    "Oil & Gas Refining & Marketing": "Energy",
    "Solar": "Energy",
    "Oil & Gas Drilling": "Energy",
    "Thermal Coal": "Energy",
    "Uranium": "Energy",
    "Coking Coal": "Energy",
    "Coal": "Energy",

    # Utilities Sector
    "Utilities—Regulated Electric": "Utilities",
    "Regulated Electric": "Utilities",
    "Utilities—Diversified": "Utilities",
    "Utilities - Diversified": "Utilities",
    "Diversified Utilities": "Utilities",
    "Utilities Diversified": "Utilities",
    "Utilities—Regulated Gas": "Utilities",
    "Utilities - Regulated Gas": "Utilities",
    "Regulated Gas": "Utilities",
    "Utilities—Regulated Water": "Utilities",
    "Regulated Water": "Utilities",
    "Utilities—Independent Power Producers": "Utilities",
    "Utilities - Independent Power Producers": "Utilities",
    "Independent Power Producers": "Utilities",
    "Utilities—Renewable": "Utilities",
    "Utilities - Renewable": "Utilities",
    "Renewable Utilities": "Utilities",

    # Real Estate Sector
    "Real Estate Services": "Real Estate",
    "Real Estate—Diversified": "Real Estate",
    "Real Estate - Diversified": "Real Estate",
    "Real Estate—Development": "Real Estate",
    "Real Estate - Development": "Real Estate",
    "Real Estate - Services": "Real Estate",
    "REIT—Residential": "Real Estate",
    "REIT - Residential": "Real Estate",
    "REIT—Industrial": "Real Estate",
    "REIT - Industrial": "Real Estate",
    "REIT—Diversified": "Real Estate",
    "REIT - Diversified": "Real Estate",
    "REIT—Retail": "Real Estate",
    "REIT - Retail": "Real Estate",
    "REIT - Hotel & Motel": "Real Estate",
    "REIT - Mortgage": "Real Estate",
    "REIT - Specialty": "Real Estate",
    "REIT - Office": "Real Estate",
    "REIT - Healthcare Facilities": "Real Estate",

    # Conglomerates (not a GICS sector, but used in classification)
    "Conglomerates": "Conglomerates",
}

# ==============================================================================
# COMMENTED CODE SECTION - Historical Examples and Alternative Data Sources
# ==============================================================================
# The following commented code shows examples of alternative data collection
# methods and libraries. Preserved for reference and potential future use.

################################################################################################
# (1) Example: Using yfinance and yahoo_fin for financial statements
# import yfinance as yf
# import yahoo_fin.stock_info as si
# from pykrx import stock
# import pymysql
# symbol = 'GOOGL'
# sp500_ticker = si.tickers_sp500()
# print(si.get_balance_sheet(symbol))

# (2) Example: Using pandas_datareader (DataReader)
# import FinanceDataReader as fdr
# symbol = 'GOOGL'
# web = 'yahoo'
# start_date = '2004-08-19'
# end_date = '2020-04-17'
# google_data = data.DataReader(symbol, web, start_date, end_date)
# print(google_data.head(9))
# google_data['Close'].plot()
# df = stock.get_market_fundamental("20220104", "20220206", "005930", freq="m")

# (3) Example: Plotting charts with matplotlib
# import mariadb
# import matplotlib.pyplot as plt
################################################################################################
