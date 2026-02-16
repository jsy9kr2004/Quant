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

        from src.infra.g_variables import ratio_col_list, meaning_col_list

        # 모델 훈련을 위해 ratio feature만 선택
        ratio_features = df[ratio_col_list]

        # 여러 feature 타입 결합
        all_features = ratio_col_list + meaning_col_list

    섹터 분류::

        from src.infra.g_variables import sector_map

        # Industry를 sector로 매핑
        industry = "Software—Application"
        sector = sector_map.get(industry, "Unknown")
        print(f"{industry} -> {sector}")  # Software—Application -> Information Technology

    데이터 품질 처리::

        from src.infra.g_variables import sparse_col_list

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
    'Ydiff_dcf',                        # DCF의 전년 대비 변화
    'Ydiff_preferredStock',             # 우선주의 전년 대비 변화
    'Qdiff_otherLiabilities',           # 기타 부채의 전분기 대비 변화
    'Ydiff_capitalLeaseObligations',    # 자본 리스 의무의 전년 대비 변화
    'Qdiff_dcf',                        # DCF의 전분기 대비 변화
    'Ydiff_otherLiabilities',           # 기타 부채의 전년 대비 변화
    'Ydiff_deferredRevenueNonCurrent',  # 이연 수익의 전년 대비 변화
    'Qdiff_otherAssets',                # 기타 자산의 전분기 대비 변화
    'Qdiff_deferredRevenueNonCurrent',  # 이연 수익의 전분기 대비 변화
    'Ydiff_sellingAndMarketingExpenses',# 판매 및 마케팅 비용의 전년 대비 변화
    'Ydiff_researchAndDdevelopementToRevenue',  # R&D 비율의 전년 대비 변화
    'dcf',                              # DCF 밸류에이션
    'Ydiff_otherAssets',                # 기타 자산의 전년 대비 변화
    'OverMC_dcf',                       # DCF / 시가총액 비율
    'Qdiff_preferredStock',             # 우선주의 전분기 대비 변화
    'Qdiff_capitalLeaseObligations'     # 자본 리스의 전분기 대비 변화
]

# ==============================================================================
# RATIO COLUMNS - Financial Ratio 및 백분율 Metric
# ==============================================================================
# 이 컬럼들은 이미 정규화되어 다른 규모의 기업간 비교가 가능한 ratio 기반 metric을 포함합니다.
# Fundamental 분석에서 일반적으로 사용되며 절대값보다 안정적인 경우가 많습니다.

ratio_col_list = [
    # 밸류에이션 비율
    "interestCoverage",                 # EBIT / 이자비용
    "dividendYield",                    # 연간 배당금 / 주가
    "peRatio",                          # 주가 / 주당순이익 (PER)
    "pbRatio",                          # 주가 / 주당순자산 (PBR)
    "ptbRatio",                         # 주가 / 주당유형순자산
    "priceToSalesRatio",                # 주가 / 주당매출액
    "pfcfRatio",                        # 주가 / 주당잉여현금흐름
    "pocfratio",                        # 주가 / 주당영업현금흐름
    "enterpriseValueOverEBITDA",        # EV / EBITDA

    # 유동성 비율
    "currentRatio",                     # 유동자산 / 유동부채

    # 효율성 비율
    "inventoryTurnover",                # 매출원가 / 평균재고
    "daysPayablesOutstanding",          # 매입채무 지급 소요일수
    "daysOfInventoryOnHand",            # 재고 보유일수
    "payablesTurnover",                 # 매출원가 / 평균매입채무
    "daysSalesOutstanding",             # 매출채권 회수 소요일수
    "receivablesTurnover",              # 매출액 / 평균매출채권

    # 수익성 비율
    "roe",                              # ROE (자기자본이익률)
    "roic",                             # ROIC (투하자본수익률)
    "returnOnTangibleAssets",           # 순이익 / 유형자산
    "earningsYield",                    # EPS / 주가
    "netIncomeRatio",                   # 순이익 / 매출액
    "grossProfitRatio",                 # 매출총이익 / 매출액
    "operatingIncomeRatio",             # 영업이익 / 매출액
    "incomeBeforeTaxRatio",             # 세전이익 / 매출액
    "ebitdaratio",                      # EBITDA / 매출액

    # 레버리지 비율
    "debtToAssets",                     # 총부채 / 총자산
    "debtToEquity",                     # 총부채 / 총자본
    "netDebtToEBITDA",                  # 순부채 / EBITDA
    "incomeQuality",                    # 현금흐름 / 순이익

    # 주당 지표
    "eps",                              # 주당순이익 (EPS)
    "epsdiluted",                       # 희석 EPS
    "revenuePerShare",                  # 매출액 / 발행주식수
    "netIncomePerShare",                # 순이익 / 발행주식수
    "freeCashFlowPerShare",             # 잉여현금흐름 / 발행주식수
    "operatingCashFlowPerShare",        # 영업현금흐름 / 발행주식수
    "cashPerShare",                     # 현금 / 발행주식수
    "bookValuePerShare",                # 순자산 / 발행주식수
    "tangibleBookValuePerShare",        # 유형순자산 / 발행주식수
    "shareholdersEquityPerShare",       # 자기자본 / 발행주식수
    "interestDebtPerShare",             # 이자부부채 / 발행주식수
    "capexPerShare",                    # 자본적 지출 / 발행주식수

    # 자본효율성 지표
    "capexToDepreciation",              # 자본적 지출 / 감가상각비
    "capexToRevenue",                   # 자본적 지출 / 매출액
    "capexToOperatingCashFlow",         # 자본적 지출 / 영업현금흐름
    "stockBasedCompensationToRevenue",  # 주식보상비용 / 매출액
    "salesGeneralAndAdministrativeToRevenue",  # 판관비 / 매출액
    "researchAndDdevelopementToRevenue",# R&D / 매출액
    "intangiblesToTotalAssets",         # 무형자산 / 총자산

    # 밸류에이션 지표
    "dcf",                              # DCF (현금흐름할인법)
    "evToOperatingCashFlow",            # EV / 영업현금흐름
    "evToFreeCashFlow",                 # EV / 잉여현금흐름
    "evToSales",                        # EV / 매출액
    "freeCashFlowYield",                # 잉여현금흐름 / 시가총액
    "grahamNetNet",                     # 그레이엄 넷넷 밸류에이션
    "grahamNumber",                     # 그레이엄 넘버 밸류에이션

    # 주주환원 지표
    "payoutRatio",                      # 배당금 / 순이익

    # 성장성 지표 (이름에 "Growth" 포함)
    "dividendsperShareGrowth",          # 배당 성장률
    "netIncomeGrowth",                  # 순이익 성장률
    "epsgrowth",                        # EPS 성장률
    "epsdilutedGrowth",                 # 희석 EPS 성장률
    "revenueGrowth",                    # 매출 성장률
    "debtGrowth",                       # 부채 성장률
    "operatingCashFlowGrowth",          # 영업현금흐름 성장률
    "ebitgrowth",                       # EBIT 성장률
    "operatingIncomeGrowth",            # 영업이익 성장률
    "assetGrowth",                      # 자산 성장률
    "freeCashFlowGrowth",               # 잉여현금흐름 성장률
    "sgaexpensesGrowth",                # 판관비 성장률
    "receivablesGrowth",                # 매출채권 성장률
    "grossProfitGrowth",                # 매출총이익 성장률
    "weightedAverageSharesGrowth",      # 발행주식수 증가율
    "weightedAverageSharesDilutedGrowth",  # 희석주식수 증가율
    "bookValueperShareGrowth",          # 주당순자산 성장률
    "inventoryGrowth",                  # 재고 성장률
    "rdexpenseGrowth",                  # R&D 비용 성장률

    # 다년간 성장 지표
    "tenYDividendperShareGrowthPerShare",   # 10년 배당 CAGR
    "threeYDividendperShareGrowthPerShare", # 3년 배당 CAGR
    "fiveYDividendperShareGrowthPerShare",  # 5년 배당 CAGR
    "threeYOperatingCFGrowthPerShare",      # 3년 영업현금흐름 CAGR
    "fiveYOperatingCFGrowthPerShare",       # 5년 영업현금흐름 CAGR
    "tenYOperatingCFGrowthPerShare",        # 10년 영업현금흐름 CAGR
    "threeYRevenueGrowthPerShare",          # 3년 매출 CAGR
    "fiveYRevenueGrowthPerShare",           # 5년 매출 CAGR
    "tenYRevenueGrowthPerShare",            # 10년 매출 CAGR
    "threeYShareholdersEquityGrowthPerShare",   # 3년 자기자본 CAGR
    "fiveYShareholdersEquityGrowthPerShare",    # 5년 자기자본 CAGR
    "tenYShareholdersEquityGrowthPerShare",     # 10년 자기자본 CAGR
    "threeYNetIncomeGrowthPerShare",        # 3년 순이익 CAGR
    "fiveYNetIncomeGrowthPerShare",         # 5년 순이익 CAGR
    "tenYNetIncomeGrowthPerShare",          # 10년 순이익 CAGR

    # 절대값 지표 (역사적 이유로 ratio 리스트에도 포함)
    "freeCashFlow",                     # 잉여현금흐름 (절대값)
    "operatingCashFlow",                # 영업현금흐름 (절대값)
    "grossProfit",                      # 매출총이익 (절대값)
    "ebitda",                           # EBITDA (절대값)
    "netDebt",                          # 순부채 (절대값)
    "investedCapital",                  # 투하자본 (절대값)
    "stockBasedCompensation",           # 주식보상비용 (절대값)
]

# ==============================================================================
# MEANINGFUL COLUMNS - 비즈니스 중요도가 있는 절대값 Metric
# ==============================================================================
# 이 컬럼들은 기업의 재무 상태와 운영에 대한 의미 있는 정보를 제공하는
# 절대값(달러 단위)을 포함합니다. 종합적인 financial 분석을 위해 ratio와 함께 사용됩니다.

meaning_col_list = [
    # 현금흐름표 항목
    "commonStockRepurchased",           # 자사주 매입 금액
    "netCashProvidedByOperatingActivities",  # 영업활동 현금흐름
    "netCashUsedForInvestingActivites", # 투자활동 현금흐름
    "netCashUsedProvidedByFinancingActivities",  # 재무활동 현금흐름
    "freeCashFlow",                     # 잉여현금흐름 (영업현금흐름 - 자본적 지출)
    "operatingCashFlow",                # 영업현금흐름
    "netChangeInCash",                  # 현금 순변동
    "changeInWorkingCapital",           # 운전자본 변동
    "capitalExpenditure",               # 자본적 지출
    "investmentsInPropertyPlantAndEquipment",  # 유형자산 투자
    "acquisitionsNet",                  # 순인수 지출
    "purchasesOfInvestments",           # 투자자산 매입
    "otherInvestingActivites",          # 기타 투자활동
    "otherNonCashItems",                # 기타 비현금 항목
    "depreciationAndAmortization",      # 감가상각비
    "otherWorkingCapital",              # 기타 운전자본 변동
    "effectOfForexChangesOnCash",       # 환율 변동이 현금에 미치는 영향

    # 대차대조표 - 자산
    "totalAssets",                      # 총자산
    "totalCurrentAssets",               # 총유동자산
    "totalNonCurrentAssets",            # 총비유동자산
    "cashAndCashEquivalents",           # 현금 및 현금성자산
    "cashAndShortTermInvestments",      # 현금 및 단기투자
    "shortTermInvestments",             # 단기투자
    "totalInvestments",                 # 총투자
    "longTermInvestments",              # 장기투자
    "netReceivables",                   # 순매출채권
    "accountsReceivables",              # 매출채권
    "inventory",                        # 재고자산
    "propertyPlantEquipmentNet",        # 순유형자산
    "goodwillAndIntangibleAssets",      # 영업권 및 무형자산
    "intangibleAssets",                 # 무형자산
    "otherAssets",                      # 기타 자산
    "otherCurrentAssets",               # 기타 유동자산
    "otherNonCurrentAssets",            # 기타 비유동자산
    "deferredRevenue",                  # 이연수익 (부채 항목)
    "deferredRevenueNonCurrent",        # 장기 이연수익
    "deferredIncomeTax",                # 이연법인세자산
    "taxAssets",                        # 세금자산

    # 대차대조표 - 부채
    "totalLiabilities",                 # 총부채
    "totalCurrentLiabilities",          # 총유동부채
    "totalNonCurrentLiabilities",       # 총비유동부채
    "totalLiabilitiesAndStockholdersEquity",  # 총부채 및 자본
    "totalLiabilitiesAndTotalEquity",   # 총부채 및 자본 (대체 명칭)
    "totalDebt",                        # 총차입금
    "shortTermDebt",                    # 단기차입금
    "longTermDebt",                     # 장기차입금
    "accountsPayables",                 # 매입채무
    "accountPayables",                  # 매입채무 (대체 명칭)
    "taxPayables",                      # 미지급세금
    "otherLiabilities",                 # 기타 부채
    "otherCurrentLiabilities",          # 기타 유동부채
    "otherNonCurrentLiabilities",       # 기타 비유동부채
    "capitalLeaseObligations",          # 자본 리스 의무
    "deferredTaxLiabilitiesNonCurrent", # 이연법인세부채

    # 대차대조표 - 자본
    "totalEquity",                      # 총자본
    "totalStockholdersEquity",          # 총주주자본
    "commonStock",                      # 보통주
    "retainedEarnings",                 # 이익잉여금
    "accumulatedOtherComprehensiveIncomeLoss",  # 기타포괄손익누계액
    "othertotalStockholdersEquity",     # 기타 자본 항목
    "minorityInterest",                 # 소수주주지분

    # 손익계산서
    "revenue",                          # 총매출액
    "costOfRevenue",                    # 매출원가
    "grossProfit",                      # 매출총이익
    "operatingExpenses",                # 총영업비용
    "costAndExpenses",                  # 총비용
    "sellingGeneralAndAdministrativeExpenses",  # 판관비
    "sellingAndMarketingExpenses",      # 판매 및 마케팅 비용
    "generalAndAdministrativeExpenses", # 일반관리비
    "researchAndDevelopmentExpenses",   # R&D 비용
    "operatingIncome",                  # 영업이익 (EBIT)
    "ebitda",                           # EBITDA
    "interestExpense",                  # 이자비용
    "interestIncome",                   # 이자수익
    "totalOtherIncomeExpensesNet",      # 기타 수익/비용
    "otherExpenses",                    # 기타 비용
    "incomeBeforeTax",                  # 세전이익
    "incomeTaxExpense",                 # 법인세비용
    "netIncome",                        # 순이익

    # 밸류에이션 및 주당 지표
    "enterpriseValue",                  # EV (기업가치)
    "dcf",                              # DCF 가치
    "stockBasedCompensation",           # 주식보상비용
    "workingCapital",                   # 운전자본
    "netCurrentAssetValue",             # NCAV (그레이엄 지표)
    "tangibleAssetValue",               # 유형자산가치
    "investedCapital",                  # 투하자본
    "netDebt",                          # 순부채 (부채 - 현금)

    # 평균값 (비율 계산에 사용)
    "averageInventory",                 # 평균 재고
    "averagePayables",                  # 평균 매입채무
    "averageReceivables",               # 평균 매출채권

    # 주당 가치 (주당 절대 금액)
    "eps",                              # 주당순이익 (EPS)
    "epsdiluted",                       # 희석 EPS
    "revenuePerShare",                  # 주당 매출액
    "netIncomePerShare",                # 주당 순이익
    "freeCashFlowPerShare",             # 주당 잉여현금흐름
    "operatingCashFlowPerShare",        # 주당 영업현금흐름
    "cashPerShare",                     # 주당 현금
    "bookValuePerShare",                # 주당 순자산
    "tangibleBookValuePerShare",        # 주당 유형순자산
    "shareholdersEquityPerShare",       # 주당 자기자본
    "interestDebtPerShare",             # 주당 이자부부채
    "capexPerShare",                    # 주당 자본적 지출

    # 비율 (편의상 포함)
    "interestCoverage",                 # 이자보상배율
    "dividendYield",                    # 배당수익률
    "inventoryTurnover",                # 재고회전율
    "daysPayablesOutstanding",          # 매입채무 지급일수
    "stockBasedCompensationToRevenue",  # 주식보상비용 / 매출액
    "capexToDepreciation",              # 자본적 지출 / 감가상각비
    "currentRatio",                     # 유동비율
    "daysOfInventoryOnHand",            # 재고보유일수
    "payablesTurnover",                 # 매입채무회전율
    "grahamNetNet",                     # 그레이엄 넷넷
    "capexToRevenue",                   # 자본적 지출 / 매출액
    "netDebtToEBITDA",                  # 순부채 / EBITDA
    "receivablesTurnover",              # 매출채권회전율
    "capexToOperatingCashFlow",         # 자본적 지출 / 영업현금흐름
    "evToOperatingCashFlow",            # EV / 영업현금흐름
    "evToFreeCashFlow",                 # EV / 잉여현금흐름
    "debtToAssets",                     # 부채 / 자산
    "peRatio",                          # PER
    "enterpriseValueOverEBITDA",        # EV / EBITDA
    "pfcfRatio",                        # 주가 / 잉여현금흐름
    "pocfratio",                        # 주가 / 영업현금흐름
    "daysSalesOutstanding",             # 매출채권 회수일수
    "incomeQuality",                    # 이익의 질
    "evToSales",                        # EV / 매출액
    "grahamNumber",                     # 그레이엄 넘버
    "priceToSalesRatio",                # PSR
    "pbRatio",                          # PBR
    "ptbRatio",                         # 주가 / 유형순자산
    "roic",                             # ROIC (투하자본수익률)
    "freeCashFlowYield",                # 잉여현금흐름 수익률
    "roe",                              # ROE (자기자본이익률)
    "returnOnTangibleAssets",           # 유형자산수익률
    "earningsYield",                    # 이익수익률
    "debtToEquity",                     # 부채비율
    "payoutRatio",                      # 배당성향
    "salesGeneralAndAdministrativeToRevenue",  # 판관비 / 매출액
    "intangiblesToTotalAssets",         # 무형자산 / 총자산
    "ebitdaratio",                      # EBITDA 마진
    "dividendsperShareGrowth",          # 배당 성장률
    "netIncomeGrowth",                  # 순이익 성장률
    "epsgrowth",                        # EPS 성장률
    "epsdilutedGrowth",                 # 희석 EPS 성장률
    "revenueGrowth",                    # 매출 성장률
    "grossProfitRatio",                 # 매출총이익률
    "debtGrowth",                       # 부채 성장률
    "tenYDividendperShareGrowthPerShare",  # 10년 배당 CAGR
    "netIncomeRatio",                   # 순이익률
    "incomeBeforeTaxRatio",             # 세전이익률
    "operatingCashFlowGrowth",          # 영업현금흐름 성장률
    "ebitgrowth",                       # EBIT 성장률
    "operatingIncomeGrowth",            # 영업이익 성장률
    "threeYDividendperShareGrowthPerShare",  # 3년 배당 CAGR
    "assetGrowth",                      # 자산 성장률
    "freeCashFlowGrowth",               # 잉여현금흐름 성장률
    "sgaexpensesGrowth",                # 판관비 성장률
    "fiveYDividendperShareGrowthPerShare",  # 5년 배당 CAGR
    "receivablesGrowth",                # 매출채권 성장률
    "fiveYRevenueGrowthPerShare",       # 5년 매출 CAGR
    "threeYOperatingCFGrowthPerShare",  # 3년 영업현금흐름 CAGR
    "grossProfitGrowth",                # 매출총이익 성장률
    "operatingIncomeRatio",             # 영업이익률
    "threeYShareholdersEquityGrowthPerShare",  # 3년 자기자본 CAGR
    "fiveYShareholdersEquityGrowthPerShare",   # 5년 자기자본 CAGR
    "fiveYOperatingCFGrowthPerShare",   # 5년 영업현금흐름 CAGR
    "threeYRevenueGrowthPerShare",      # 3년 매출 CAGR
    "researchAndDdevelopementToRevenue",# R&D / 매출액
    "threeYNetIncomeGrowthPerShare",    # 3년 순이익 CAGR
    "tenYOperatingCFGrowthPerShare",    # 10년 영업현금흐름 CAGR
    "tenYRevenueGrowthPerShare",        # 10년 매출 CAGR
    "tenYShareholdersEquityGrowthPerShare",  # 10년 자기자본 CAGR
    "interestExpense",                  # 이자비용
    "tenYNetIncomeGrowthPerShare",      # 10년 순이익 CAGR
    "weightedAverageSharesGrowth",      # 발행주식수 증가율
    "weightedAverageSharesDilutedGrowth",  # 희석주식수 증가율
    "fiveYNetIncomeGrowthPerShare",     # 5년 순이익 CAGR
    "bookValueperShareGrowth",          # 주당순자산 성장률
    "inventoryGrowth",                  # 재고 성장률
    "rdexpenseGrowth",                  # R&D 비용 성장률

    # 현금 포지션 지표
    "cashAtBeginningOfPeriod",          # 기초 현금
    "cashAtEndOfPeriod",                # 기말 현금
]

# ==============================================================================
# TIME FEATURE COLUMNS - Time Series 분석을 위한 Feature
# ==============================================================================
# 이 컬럼들은 이동평균, 추세, 모멘텀 지표, 기타 시간적 패턴 같은
# 시간 기반 feature를 계산하기 위해 선택되었습니다.
#
# IMPORTANT: Price momentum features added (2025-12-03)
# Philosophy: Combine Value (fundamentals) + Momentum (price trends)
# - 80% Value: Fundamental metrics (PBR, ROE, etc.) - existing
# - 20% Momentum: Price/volume trends - NEW
# Empirical research shows Value + Momentum outperforms either alone

cal_timefeature_col_list = [
    # ===== Price Momentum Features (NEW) =====
    # These extract price-based momentum (3m/6m returns, trends, etc.)
    "price",                            # Stock price (tsfresh will extract: trends, moving averages, volatility)
    "volume",                           # Trading volume (tsfresh will extract: volume trends, surges)
    "volume_mul_price",                 # Trading value (combines price × volume momentum)

    # ===== Fundamental Features (EXISTING) =====
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
