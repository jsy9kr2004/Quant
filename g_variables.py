cal_col_list = [
"commonStockRepurchased", "interestCoverage", "dividendYield", "averageInventory", "dividendsPaid", "deferredRevenueNonCurrent",
"inventoryTurnover", "purchasesOfInvestments", "accountsReceivables", "tangibleAssetValue", "daysPayablesOutstanding", "dcf",
"averagePayables", "acquisitionsNet", "accountsPayables", "averageReceivables", "workingCapital", "capexToDepreciation",
"currentRatio", "netCurrentAssetValue", "daysOfInventoryOnHand", "payablesTurnover", "grahamNetNet", "capexToRevenue",
"netDebtToEBITDA",  "receivablesTurnover", "capexToOperatingCashFlow", "evToOperatingCashFlow", "evToFreeCashFlow","debtToAssets",
"tangibleBookValuePerShare","stockBasedCompensation", "capexPerShare", "peRatio", "otherAssets",  "enterpriseValueOverEBITDA",
"enterpriseValue", "bookValuePerShare", "shareholdersEquityPerShare", "pfcfRatio","pocfratio", "daysSalesOutstanding",
"incomeQuality", "interestDebtPerShare", "revenuePerShare", "investmentsInPropertyPlantAndEquipment", "freeCashFlowPerShare",
"evToSales", "netIncomePerShare", "grahamNumber", "operatingCashFlowPerShare", "cashPerShare", "priceToSalesRatio",
"pbRatio", "ptbRatio", "investedCapital", "roic","preferredStock",  "marketCap", "freeCashFlowYield", "roe",
"otherLiabilities", "taxPayables", "returnOnTangibleAssets", "otherCurrentAssets", "earningsYield", "capitalExpenditure",
"debtToEquity", "payoutRatio", "otherNonCashItems", "totalEquity", "costAndExpenses", "totalLiabilities", "totalCurrentLiabilities",
"operatingExpenses", "totalCurrentAssets", "totalNonCurrentAssets", "netCashUsedProvidedByFinancingActivities", "propertyPlantEquipmentNet",
"accountPayables", "salesGeneralAndAdministrativeToRevenue", "otherCurrentLiabilities", "sellingGeneralAndAdministrativeExpenses",
"intangiblesToTotalAssets", "otherInvestingActivites", "totalStockholdersEquity", "netDebt", "totalLiabilitiesAndStockholdersEquity",
"totalAssets", "totalLiabilitiesAndTotalEquity", "totalNonCurrentLiabilities", "capitalLeaseObligations",
"revenue", "ebitdaratio", "ebitda", "dividendsperShareGrowth", "othertotalStockholdersEquity",
"netChangeInCash", "changeInWorkingCapital", "netIncome_x", "depreciationAndAmortization_x", "netReceivables","cashAtBeginningOfPeriod",
"netCashUsedForInvestingActivites", "freeCashFlow", "otherWorkingCapital", "incomeBeforeTax", "cashAtEndOfPeriod",
"sellingAndMarketingExpenses", "netCashProvidedByOperatingActivities", "operatingCashFlow", "netIncomeGrowth",
"otherNonCurrentAssets", "cashAndShortTermInvestments", "accumulatedOtherComprehensiveIncomeLoss", "grossProfit","cashAndCashEquivalents",
"epsgrowth", "commonStock", "totalDebt", "epsdilutedGrowth", "incomeTaxExpense", "retainedEarnings", "revenueGrowth",
"grossProfitRatio", "epsdiluted", "eps", "debtGrowth", "tenYDividendperShareGrowthPerShare", "operatingIncome", "netIncomeRatio",
"totalOtherIncomeExpensesNet", "incomeBeforeTaxRatio", "costOfRevenue", "operatingCashFlowGrowth", "totalInvestments", "ebitgrowth",
"operatingIncomeGrowth", "threeYDividendperShareGrowthPerShare", "assetGrowth", "freeCashFlowGrowth", "sgaexpensesGrowth",
"fiveYDividendperShareGrowthPerShare", "receivablesGrowth", "minorityInterest", "fiveYRevenueGrowthPerShare", "deferredIncomeTax",
"threeYOperatingCFGrowthPerShare", "longTermDebt", "grossProfitGrowth", "operatingIncomeRatio", "otherNonCurrentLiabilities",
"threeYShareholdersEquityGrowthPerShare",  "fiveYShareholdersEquityGrowthPerShare", "fiveYOperatingCFGrowthPerShare", 
"threeYRevenueGrowthPerShare", "researchAndDdevelopementToRevenue", "goodwillAndIntangibleAssets", "threeYNetIncomeGrowthPerShare",
"tenYOperatingCFGrowthPerShare", "tenYRevenueGrowthPerShare", "tenYShareholdersEquityGrowthPerShare", "interestExpense",
"tenYNetIncomeGrowthPerShare", "weightedAverageSharesGrowth", "longTermInvestments", "fiveYNetIncomeGrowthPerShare", "bookValueperShareGrowth",
"inventoryGrowth", "generalAndAdministrativeExpenses", "shortTermDebt", "interestIncome", "rdexpenseGrowth", "effectOfForexChangesOnCash",
"intangibleAssets", "otherExpenses", "deferredRevenue", "shortTermInvestments", "deferredTaxLiabilitiesNonCurrent", "taxAssets",
"researchAndDevelopmentExpenses"
]

use_col_list = ["interestCoverage", "dividendYield", "inventoryTurnover", "daysPayablesOutstanding",
                "stockBasedCompensationToRevenue", "dcf", "capexToDepreciation", "currentRatio",
                "daysOfInventoryOnHand", "payablesTurnover", "grahamNetNet", "capexToRevenue", "netDebtToEBITDA",
                "receivablesTurnover", "capexToOperatingCashFlow", "evToOperatingCashFlow", "evToFreeCashFlow",
                "debtToAssets", "tangibleBookValuePerShare", "stockBasedCompensation", "capexPerShare", "peRatio",
                "enterpriseValueOverEBITDA", "bookValuePerShare", "shareholdersEquityPerShare", "pfcfRatio",
                "pocfratio", "daysSalesOutstanding", "incomeQuality", "interestDebtPerShare", "revenuePerShare",
                "freeCashFlowPerShare", "evToSales", "netIncomePerShare", "grahamNumber", "operatingCashFlowPerShare",
                "cashPerShare", "priceToSalesRatio", "pbRatio", "ptbRatio", "investedCapital", "roic",
                "freeCashFlowYield", "roe", "returnOnTangibleAssets", "earningsYield", "debtToEquity", "payoutRatio",
                "salesGeneralAndAdministrativeToRevenue", "intangiblesToTotalAssets", "netDebt", "ebitdaratio",
                "ebitda", "dividendsperShareGrowth", "freeCashFlow", "operatingCashFlow", "netIncomeGrowth",
                "grossProfit", "epsgrowth", "epsdilutedGrowth", "revenueGrowth", "grossProfitRatio", "epsdiluted",
                "eps", "debtGrowth", "tenYDividendperShareGrowthPerShare", "netIncomeRatio", "incomeBeforeTaxRatio",
                "operatingCashFlowGrowth", "ebitgrowth", "operatingIncomeGrowth",
                "threeYDividendperShareGrowthPerShare", "assetGrowth", "freeCashFlowGrowth", "sgaexpensesGrowth",
                "fiveYDividendperShareGrowthPerShare", "receivablesGrowth", "fiveYRevenueGrowthPerShare",
                "threeYOperatingCFGrowthPerShare", "grossProfitGrowth", "operatingIncomeRatio",
                "threeYShareholdersEquityGrowthPerShare", "fiveYShareholdersEquityGrowthPerShare",
                "fiveYOperatingCFGrowthPerShare", "threeYRevenueGrowthPerShare", "researchAndDdevelopementToRevenue",
                "threeYNetIncomeGrowthPerShare", "tenYOperatingCFGrowthPerShare", "tenYRevenueGrowthPerShare",
                "tenYShareholdersEquityGrowthPerShare", "tenYNetIncomeGrowthPerShare", "weightedAverageSharesGrowth",
                "weightedAverageSharesDilutedGrowth", "fiveYNetIncomeGrowthPerShare", "bookValueperShareGrowth",
                "inventoryGrowth", "rdexpenseGrowth",
                ]


cal_ev_col_list = [
    # "netdebt", # (netdebt + 시총) = EV
    # "operatingCashflow", # ev / operatingCashflow = evToOperatingCashFlow
    "freeCashFlow", # ev / FreeCashflow = evToFreeCashflow
    "ebitda", #  ev / ebitda = enterpriseValueOverEBITDA
    # "revenues" # ev/revenues =  evToSales
]

# use_col_list = [ "bookValuePerShare_normal", "capexPerShare_normal", "capexToOperatingCashFlow_normal",
# "capexToRevenue_normal", "cashPerShare_normal", "currentRatio_normal", "daysOfInventoryOnHand_normal",
# "daysPayablesOutstanding_normal", "daysSalesOutstanding_normal", "dcf_normal", "debtToAssets_normal",
# "debtToEquity_normal", "earningsYield_normal", "ebitda_normal", "ebitdaratio_normal", "ebitgrowth_normal",
# "enterpriseValueOverEBITDA_normal", "eps_normal", "epsdiluted_normal", "epsdilutedGrowth_normal", "epsgrowth_normal",
# "evToFreeCashFlow_normal", "evToOperatingCashFlow_normal", "evToSales_normal", "freeCashFlow_normal",
# "freeCashFlowGrowth_normal", "freeCashFlowPerShare_normal", "freeCashFlowYield_normal", "grossProfit_normal",
# "grossProfitGrowth_normal", "grossProfitRatio_normal", "incomeBeforeTaxRatio_normal", "incomeQuality_normal",
# "intangiblesToTotalAssets_normal", "interestDebtPerShare_normal", "inventoryTurnover_normal",
# "investedCapital_normal", "netDebtToEBITDA_normal", "netIncomeGrowth_normal", "netIncomePerShare_normal",
# "netIncomeRatio_normal", "operatingCashFlow_normal", "operatingCashFlowPerShare_normal",
# "operatingIncomeGrowth_normal", "operatingIncomeRatio_normal", "payoutRatio_normal", "pbRatio_normal",
# "peRatio_normal", "pfcfRatio_normal", "pocfratio_normal", "priceToSalesRatio_normal", "ptbRatio_normal",
# "rdexpenseGrowth_normal", "receivablesGrowth_normal", "researchAndDdevelopementToRevenue_normal",
# "returnOnTangibleAssets_normal", "revenueGrowth_normal", "revenuePerShare_normal", "roe_normal", "roic_normal",
# "salesGeneralAndAdministrativeToRevenue_normal", "shareholdersEquityPerShare_normal",
# "stockBasedCompensationToRevenue_normal","threeYDividendperShareGrowthPerShare_normal",
# "threeYNetIncomeGrowthPerShare_normal", "threeYOperatingCFGrowthPerShare_normal",
# "threeYRevenueGrowthPerShare_normal", "threeYShareholdersEquityGrowthPerShare_normal", "period_price_diff",
# "earning_diff", "symbol" ]

# use_col_list = [
#     "interestCoverage_normal", "dividendYield_normal", "inventoryTurnover_normal", "daysPayablesOutstanding_normal",
#     "stockBasedCompensationToRevenue_normal", "dcf_normal", "capexToDepreciation_normal", "currentRatio_normal",
#     "daysOfInventoryOnHand_normal", "payablesTurnover_normal", "grahamNetNet_normal", "capexToRevenue_normal",
#     "netDebtToEBITDA_normal", "receivablesTurnover_normal", "capexToOperatingCashFlow_normal",
#     "evToOperatingCashFlow_normal", "evToFreeCashFlow_normal", "debtToAssets_normal",
#     "tangibleBookValuePerShare_normal", "stockBasedCompensation_normal", "capexPerShare_normal", "peRatio_normal",
#     "enterpriseValueOverEBITDA_normal", "bookValuePerShare_normal", "shareholdersEquityPerShare_normal",
#     "pfcfRatio_normal", "pocfratio_normal", "daysSalesOutstanding_normal", "incomeQuality_normal",
#     "interestDebtPerShare_normal", "revenuePerShare_normal", "freeCashFlowPerShare_normal", "evToSales_normal",
#     "netIncomePerShare_normal", "grahamNumber_normal", "operatingCashFlowPerShare_normal", "cashPerShare_normal",
#     "priceToSalesRatio_normal", "pbRatio_normal", "ptbRatio_normal", "investedCapital_normal", "roic_normal",
#     "freeCashFlowYield_normal", "roe_normal", "returnOnTangibleAssets_normal", "earningsYield_normal",
#     "debtToEquity_normal", "payoutRatio_normal", "salesGeneralAndAdministrativeToRevenue_normal",
#     "intangiblesToTotalAssets_normal", "netDebt_normal", "ebitdaratio_normal", "ebitda_normal",
#     "dividendsperShareGrowth_normal", "freeCashFlow_normal", "operatingCashFlow_normal", "netIncomeGrowth_normal",
#     "grossProfit_normal", "epsgrowth_normal", "epsdilutedGrowth_normal", "revenueGrowth_normal",
#     "grossProfitRatio_normal", "epsdiluted_normal", "eps_normal", "debtGrowth_normal",
#     "tenYDividendperShareGrowthPerShare_normal", "netIncomeRatio_normal", "incomeBeforeTaxRatio_normal",
#     "operatingCashFlowGrowth_normal", "ebitgrowth_normal", "operatingIncomeGrowth_normal",
#     "threeYDividendperShareGrowthPerShare_normal", "assetGrowth_normal", "freeCashFlowGrowth_normal",
#     "sgaexpensesGrowth_normal", "fiveYDividendperShareGrowthPerShare_normal", "receivablesGrowth_normal",
#     "fiveYRevenueGrowthPerShare_normal", "threeYOperatingCFGrowthPerShare_normal", "grossProfitGrowth_normal",
#     "operatingIncomeRatio_normal", "threeYShareholdersEquityGrowthPerShare_normal",
#     "fiveYShareholdersEquityGrowthPerShare_normal", "fiveYOperatingCFGrowthPerShare_normal",
#     "threeYRevenueGrowthPerShare_normal", "researchAndDdevelopementToRevenue_normal",
#     "threeYNetIncomeGrowthPerShare_normal", "tenYOperatingCFGrowthPerShare_normal",
#     "tenYRevenueGrowthPerShare_normal", "tenYShareholdersEquityGrowthPerShare_normal",
#     "tenYNetIncomeGrowthPerShare_normal", "weightedAverageSharesGrowth_normal",
#     "weightedAverageSharesDilutedGrowth_normal", "fiveYNetIncomeGrowthPerShare_normal",
#     "bookValueperShareGrowth_normal", "inventoryGrowth_normal", "rdexpenseGrowth_normal",
#     "period_price_diff", "earning_diff", "symbol" ]

################################################################################################
# (1) tickers를 이용한 재무재표 예제
# import yfinance as yf
# import yahoo_fin.stock_info as si
# from pykrx import stock
# import pymysql
# symbol = 'GOOGL'
# sp500_ticker = si.tickers_sp500()
# print(si.get_balance_sheet(symbol))

# (2) DataReader 예제
# import FinanceDataReader as fdr
# symbol = 'GOOGL'
# web = 'yahoo'
# start_date = '2004-08-19'
# end_date = '2020-04-17'
# google_data = data.DataReader(symbol, web, start_date, end_date)
# print(google_data.head(9))
# google_data['Close'].plot()
# df = stock.get_market_fundamental("20220104", "20220206", "005930", freq="m")

# (3) chart 그리기
# import mariadb
# import matplotlib.pyplot as plt
################################################################################################
