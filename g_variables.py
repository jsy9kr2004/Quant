sparse_col_list = ['Ydiff_dcf', 'Ydiff_preferredStock', 'Qdiff_otherLiabilities', 'Ydiff_capitalLeaseObligations', 'Qdiff_dcf',
                   'Ydiff_otherLiabilities', 'Ydiff_deferredRevenueNonCurrent', 'Qdiff_otherAssets', 'Qdiff_deferredRevenueNonCurrent', 
                   'Ydiff_sellingAndMarketingExpenses', 'Ydiff_researchAndDdevelopementToRevenue', 
                   'dcf', 'Ydiff_otherAssets', 'OverMC_dcf', 'Qdiff_preferredStock', 'Qdiff_capitalLeaseObligations']

meaning_col_list = [
"commonStockRepurchased", "interestCoverage", "dividendYield", "averageInventory", "dividendsPaid", "deferredRevenueNonCurrent",
"inventoryTurnover", "purchasesOfInvestments", "accountsReceivables", "tangibleAssetValue", "daysPayablesOutstanding", "dcf",
"averagePayables", "acquisitionsNet", "accountsPayables", "averageReceivables", "workingCapital", "capexToDepreciation",
"currentRatio", "netCurrentAssetValue", "daysOfInventoryOnHand", "payablesTurnover", "grahamNetNet", "capexToRevenue",
"netDebtToEBITDA",  "receivablesTurnover", "capexToOperatingCashFlow", "evToOperatingCashFlow", "evToFreeCashFlow","debtToAssets",
"tangibleBookValuePerShare","stockBasedCompensation", "capexPerShare", "peRatio", "otherAssets",  "enterpriseValueOverEBITDA",
"enterpriseValue", "bookValuePerShare", "shareholdersEquityPerShare", "pfcfRatio","pocfratio", "daysSalesOutstanding",
"incomeQuality", "interestDebtPerShare", "revenuePerShare", "investmentsInPropertyPlantAndEquipment", "freeCashFlowPerShare",
"evToSales", "netIncomePerShare", "grahamNumber", "operatingCashFlowPerShare", "cashPerShare", "priceToSalesRatio",
"pbRatio", "ptbRatio", "investedCapital", "roic",  "freeCashFlowYield", "roe",
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
"researchAndDevelopmentExpenses",
]

ratio_col_list = ["interestCoverage", "dividendYield", "inventoryTurnover", "daysPayablesOutstanding",
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


sector_map = {
    "Asset Management": "Financials",
    "Banks—Regional": "Financials",
    "Biotechnology": "Healthcare",
    "Software—Application": "Information Technology",
    "Telecom Services": "Communication Services",
    "Specialty Industrial Machinery": "Industrials",
    "Semiconductors": "Information Technology",
    "Software—Infrastructure": "Information Technology",
    "Utilities—Regulated Electric": "Utilities",
    "Aerospace & Defense": "Industrials",
    "Oil & Gas E&P": "Energy",
    "Specialty Chemicals": "Materials",
    "Auto Parts": "Consumer Discretionary",
    "Information Technology Services": "Information Technology",
    "Apparel Retail": "Consumer Discretionary",
    "Packaged Foods": "Consumer Staples",
    "Communication Equipment": "Information Technology",
    "Banks—Diversified": "Financials",
    "Medical Instruments & Supplies": "Healthcare",
    "Medical Devices": "Healthcare",
    "Oil & Gas Midstream": "Energy",
    "Restaurants": "Consumer Discretionary",
    "Diagnostics & Research": "Healthcare",
    "Specialty Retail": "Consumer Discretionary",
    "Drug Manufacturers—Specialty & Generic": "Healthcare",
    "Oil & Gas Equipment & Services": "Energy",
    "Gold": "Materials",
    "Oil & Gas Integrated": "Energy",
    "Medical Care Facilities": "Healthcare",
    "Insurance—Property & Casualty": "Financials",
    "Building Products & Equipment": "Industrials",
    "Credit Services": "Financials",
    "Entertainment": "Consumer Discretionary",
    "Packaging & Containers": "Materials",
    "Drug Manufacturers—General": "Healthcare",
    "Capital Markets": "Financials",
    "Semiconductor Equipment & Materials": "Information Technology",
    "Internet Content & Information": "Communication Services",
    "Furnishings, Fixtures & Appliances": "Consumer Discretionary",
    "Specialty Business Services": "Industrials",
    "Electronic Components": "Information Technology",
    "Engineering & Construction": "Industrials",
    "Insurance—Life": "Financials",
    "Steel": "Materials",
    "Airlines": "Consumer Discretionary",
    "Rental & Leasing Services": "Industrials",
    "Farm & Heavy Construction Machinery": "Industrials",
    "Oil & Gas Refining & Marketing": "Energy",
    "Utilities—Diversified": "Utilities",
    "Residential Construction": "Consumer Discretionary",
    "Scientific & Technical Instruments": "Information Technology",
    "Integrated Freight & Logistics": "Industrials",
    "Staffing & Employment Services": "Industrials",
    "Household & Personal Products": "Consumer Staples",
    "Computer Hardware": "Information Technology",
    "Healthcare Plans": "Healthcare",
    "Auto & Truck Dealerships": "Consumer Discretionary",
    "Footwear & Accessories": "Consumer Discretionary",
    "Waste Management": "Materials",
    "Leisure": "Consumer Discretionary",
    "Resorts & Casinos": "Consumer Discretionary",
    "Railroads" : "Industrials",
    "Other Industrial Metals & Mining": "Materials",
    "Advertising Agencies" : "Consumer Discretionary",
    "Financial Data & Stock Exchanges" : "Financials",
    "Apparel Manufacturing" : "Consumer Discretionary",
    "Industrial Distribution" : "Industrials",
    "Education & Training Services" : "Consumer Discretionary",
    "Consulting Services" : "Industrials",
    "Beverages—Non-Alcoholic" : "Consumer Staples",
    "Electrical Equipment & Parts" : "Industrials",
    "Travel Services" : "Consumer Discretionary",
    "Internet Retail" : "Consumer Discretionary",
    "Personal Services" : "Consumer Discretionary",
    "Utilities—Regulated Gas" : "Utilities",
    "Insurance—Specialty" : "Financials",
    "Insurance—Diversified" : "Financials",
    "Marine Shipping" : "Industrials",
    "Solar" : "Energy",
    "Discount Stores" : "Consumer Discretionary",
    "Electronic Gaming & Multimedia" : "Consumer Discretionary",
    "Publishing" : "Consumer Discretionary",
    "Chemicals" : "Materials",
    "Auto Manufacturers" : "Consumer Discretionary",
    "Metal Fabrication" : "Industrials",
    "Tools & Accessories" : "Consumer Discretionary",
    "Medical Distribution" : "Healthcare",
    "Trucking" : "Industrials",
    "Luxury Goods" : "Consumer Discretionary",
    "Insurance Brokers" : "Financials",
    "Real Estate Services" : "Real Estate",
    "Department Stores" : "Consumer Discretionary",
    "Conglomerates" : "Conglomerates",
    "Agricultural Inputs" : "Materials",
    "Recreational Vehicles" : "Consumer Discretionary",
    "Utilities—Regulated Water" : "Utilities",
    "Insurance—Reinsurance" : "Financials",
    "Health Information Services" : "Healthcare",
    "Farm Products" : "Consumer Staples",
    "Business Equipment & Supplies" : "Information Technology",
    "Gambling" : "Consumer Discretionary",
    "Tobacco" : "Consumer Staples",
    "Oil & Gas Drilling" : "Energy",
    "Paper & Paper Products" : "Materials",
    "Building Materials" : "Materials",
    "Home Improvement Retail" : "Consumer Discretionary",
    "Lodging" : "Consumer Discretionary",
    "Mortgage Finance" : "Financials",
    "Beverages—Brewers" : "Consumer Staples",
    "Food Distribution" : "Consumer Staples",
    "Beverages—Wineries & Distilleries" : "Consumer Staples",
    "Aluminum" : "Materials",
    "Silver" : "Materials",
    "Electronics & Computer Distribution" : "Information Technology",
    "Confectioners" : "Consumer Staples",
    "Other Precious Metals & Mining" : "Materials",
    "Consumer Electronics": "Consumer Discretionary",
    "Copper": "Materials",
    "Pharmaceutical Retailers": "Healthcare",
    "Security & Protection Services": "Information Technology",
    "Lumber & Wood Production": "Materials",
    "Thermal Coal": "Energy",
    "Real Estate—Diversified": "Real Estate",
    "Pollution & Treatment Controls": "Industrials",
    "Broadcasting": "Consumer Discretionary",
    "Utilities Diversified": "Utilities",
    "Utilities—Independent Power Producers": "Utilities",
    "Utilities—Renewable": "Utilities",
    "REIT—Residential": "Real Estate",
    "Uranium": "Energy",
    "Grocery Stores": "Consumer Staples",
    "REIT—Industrial": "Real Estate",
    "CATV Systems": "Communication Services",
    "Coking Coal": "Energy",
    "REIT—Diversified": "Real Estate",
    "Real Estate—Development": "Real Estate"
}


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
