# 데이터 파이프라인 상세 분석

> **작성일**: 2025-12-17 (최종 업데이트: 2026-01-17)
> **이전 문서**: [01_architecture.md](./01_architecture.md)
> **다음 문서**: [03_ml_strategy.md](./03_ml_strategy.md)

---

## 목차

1. [파이프라인 개요](#1-파이프라인-개요)
2. [Stage 1: 데이터 수집 (FMP API)](#2-stage-1-데이터-수집)
3. [Stage 2: Parquet 변환](#3-stage-2-parquet-변환)
4. [Stage 3: Feature Engineering](#4-stage-3-feature-engineering)
5. [데이터 품질 검증](#5-데이터-품질-검증)
6. [미래 유출 방지](#6-미래-유출-방지)
7. [문제점 및 개선안](#7-문제점-및-개선안)

---

## 1. 파이프라인 개요

### 전체 흐름

```
[1. 수집]       FMP API (30년 히스토리)
                   ↓ Ray 병렬 처리
                {ROOT_PATH}/fmp_raw/*.parquet

[2. 변환]       VIEW 테이블 구축 
                   ↓ parquet_converter.py
                {ROOT_PATH}/processed/views/*.parquet

[3. Feature]    tsfresh + 재무 비율 계산
                   ↓ make_mldata.py
                   ↓ Extreme mover 필터링
                   ↓ Winsorization
                {ROOT_PATH}/processed/ml_data/per_year/*.parquet

[4. 학습]       DataProcessor 전처리
                   ↓ NaN/Infinite 제거
                   ↓ Sparse row/col 제거
                   ↓ Feature selection
                ML 모델 학습

[5. 백테스트]   Walk-Forward Analysis
                   ↓ Filing Date 기준 cutoff
                예측 → Top-K 선택 → 수익률 계산
```

---

## 2. Stage 1: 데이터 수집

### FMP API 구조

```python
# src/data_collector/fmp.py
class FMP:
    def collect(self):
        # 1. 종목 리스트 로드 (stocks.csv)
        stocks = self._load_stock_list()
        
        # 2. API 엔드포인트 리스트 로드
        api_list = [
            'income-statement',      # 손익계산서
            'balance-sheet-statement', # 재무상태표
            'cash-flow-statement',   # 현금흐름표
            'ratios',                # 재무 비율
            'key-metrics',           # 핵심 지표
            'financial-growth',      # 성장 지표
            'historical-price-full'  # 가격 데이터
        ]
        
        # 3. Ray 병렬 처리로 수집
        ray.init()
        results = ray.get([
            fetch_worker.remote(symbol, api_url) 
            for symbol in stocks
            for api_url in api_list
        ])
        
        # 4. Parquet 저장
        save_to_parquet(results, f'{ROOT_PATH}/fmp_raw/')
```

**강점**:
- Ray 병렬 처리로 속도 향상 (10배+)
- Parquet 포맷으로 효율적 저장
- 30년 히스토리 (1996~2025)
- **강건한 오류 처리 시스템**:
  - Rate Limit 자동 감지 및 재시도 (`fmp_fetch_worker.py:156-159`)
  - 네트워크 오류 처리 (ValueError, HTTPError 핸들링)
  - 무한 재시도 루프로 일시적 장애 극복 (`fmp_fetch_worker.py:134-210`)
  - Worker 수 제한 (max 8개)으로 API 과부하 방지
  - **Quarantine 시스템** (`fmp.py:332-446`):
    - 실패한 다운로드를 `_quarantine/` 폴더로 격리
    - `_retry_list.csv` 생성으로 재시도 대상 관리
    - API 오류 메시지 자동 감지 (regex 필터링)

**약점**:
- 메모리 사용량 높음 (모든 워커가 동시 메모리 로드)
- 재시도 전략이 무한 루프 (최대 재시도 횟수 제한 없음)

### 수집 데이터 크기

```
FMP API 수집량 (약 3000개 종목 × 30년):
- income-statement: ~300MB
- balance-sheet-statement: ~400MB
- cash-flow-statement: ~250MB
- ratios: ~200MB
- key-metrics: ~150MB
- financial-growth: ~100MB
- historical-price-full: ~1GB
─────────────────────────────────
Total: ~2.5GB (Parquet 압축)
```

---

## 3. Stage 2: Parquet 변환

### VIEW 테이블 구축

```python
# src/storage/parquet_converter.py
class Parquet:
    def rebuild_table_view(self):
        """
        Raw 데이터 → VIEW 테이블 변환
        """
        # 1. 각 API 카테고리별로 통합
        self._build_view('price')              # 가격 데이터
        self._build_view('financial_statement') # 재무제표 (IS+BS+CF 통합)
        self._build_view('ratios')             # 재무 비율
        self._build_view('key_metrics')        # 핵심 지표
        self._build_view('financial_growth')   # 성장 지표
        
    def _build_view(self, view_name):
        # 2. 모든 종목 데이터 로드
        dfs = []
        for symbol in stocks:
            df = load_parquet(f'fmp_raw/{view_name}/{symbol}.parquet')
            df['symbol'] = symbol
            dfs.append(df)
        
        # 3. 통합 및 정렬
        view_df = pd.concat(dfs, ignore_index=True)
        view_df = view_df.sort_values(['symbol', 'date'])
        
        # 4. 저장
        save_parquet(view_df, f'processed/views/{view_name}.parquet')
```

**VIEW 테이블 구조**:
```
price.parquet: (symbol, date, open, high, low, close, volume, ...)
financial_statement.parquet: (symbol, date, revenue, netIncome, totalAssets, ...)
ratios.parquet: (symbol, date, pe, pb, roe, roa, ...)
```

**강점**:
- 종목별 분산 → 통합 VIEW로 분석 용이
- Parquet 포맷으로 빠른 쿼리
- 날짜 기준 정렬로 시계열 분석 준비

**약점**:
- 메모리 사용량 높음 (모든 종목 동시 로드)
- 증분 업데이트 미지원 (전체 재구축)

---

## 4. Stage 3: Feature Engineering

### tsfresh 기반 시계열 Feature

```python
# src/training/make_mldata.py
class AIDataMaker:
    def make_ml_data(self, start_year, end_year):
        # 1. VIEW 테이블 로드
        price_df = load_view('price')
        fs_df = load_view('financial_statement')
        
        # 2. 날짜 인덱스 생성 (3개월 간격)
        date_list = self.generate_date_list(start_year, end_year, freq='3M')
        
        # 3. 각 날짜마다 Feature 생성
        for date in date_list:
            # a. 과거 20일 가격 데이터 로드
            price_window = price_df[
                (price_df['date'] < date) &
                (price_df['date'] >= date - timedelta(days=20))
            ]
            
            # b. tsfresh Feature 추출
            ts_features = extract_features(
                price_window, 
                column_id='symbol',
                column_sort='date',
                default_fc_parameters=ComprehensiveFCParameters()
            )
            # 생성 예: close__mean, close__std, close__skewness, ...
            
            # c. 재무 비율 계산
            fs_latest = fs_df[fs_df['date'] <= date].groupby('symbol').last()
            financial_ratios = self._calculate_ratios(fs_latest)
            
            # d. 타겟 계산 (다음 분기 수익률)
            future_price = price_df[
                (price_df['date'] >= date) &
                (price_df['date'] < date + timedelta(days=90))
            ]
            target = self._calculate_target(future_price)
            
            # e. Merge
            ml_df = pd.merge(ts_features, financial_ratios, on='symbol')
            ml_df = pd.merge(ml_df, target, on='symbol')
            
            # f. 저장
            save_parquet(ml_df, f'ml_data/per_year/rnorm_ml_{year}_Q{quarter}.parquet')
```

**생성 Feature 예시**:
```
가격 시계열 (20일 × 7개 컬럼: open, high, low, close, volume, ...):
- close__mean: 평균 가격
- close__std: 변동성
- close__skewness: 왜도
- close__kurtosis: 첨도
- close__maximum: 최대값
- close__minimum: 최소값
- close__variance: 분산
- volume__mean: 평균 거래량
- ... (총 수백 개)

재무 비율:
- pe_ratio: PER
- pb_ratio: PBR
- roe: ROE
- roa: ROA
- debt_to_equity: 부채비율
- current_ratio: 유동비율
- ... (총 수십 개)

타겟:
- price_dev: 다음 분기 가격 변동
- price_dev_subavg: 전체 평균 대비 상대 수익률
- sec_price_dev_subavg: 섹터 평균 대비 상대 수익률
```

### Extreme Mover 필터링

```python
def _filter_extreme_movers(self, df):
    """
    뉴스/이벤트 기반 급등락 제거
    철학: 극단적 수익률은 펀더멘털 아닌 뉴스/심리 기반
    """
    if self.config['FEATURES']['FILTER_EXTREME_MOVERS'] != 'Y':
        return df
    
    method = self.config['FEATURES']['EXTREME_FILTER_METHOD']
    threshold = self.config['FEATURES']['EXTREME_FILTER_THRESHOLD']
    
    if method == 'robust_zscore':
        # 중앙값/MAD 사용 (이상치에 강건)
        median = df['price_dev'].median()
        mad = median_abs_deviation(df['price_dev'])
        z_score = (df['price_dev'] - median) / mad
        mask = np.abs(z_score) <= threshold
        
    elif method == 'zscore':
        # 평균/표준편차 사용
        mean = df['price_dev'].mean()
        std = df['price_dev'].std()
        z_score = (df['price_dev'] - mean) / std
        mask = np.abs(z_score) <= threshold
        
    elif method == 'percentile':
        # 백분위수 하드 컷
        lower = df['price_dev'].quantile(threshold)
        upper = df['price_dev'].quantile(1 - threshold)
        mask = (df['price_dev'] >= lower) & (df['price_dev'] <= upper)
    
    df_filtered = df[mask]
    
    print(f"Extreme mover filtered: {len(df)} → {len(df_filtered)}")
    # 예: Extreme mover filtered: 10000 → 9500 (5% 제거)
    
    return df_filtered
```

**효과**:
- 노이즈 제거 (뉴스 기반 급등락)
- 펀더멘털 패턴 학습 집중

**주의**:
- Threshold가 너무 작으면 좋은 데이터도 제거
- Threshold가 너무 크면 노이즈 남음
- 민감도 분석 필요

### Winsorization

```python
def _apply_winsorization(self, df):
    """
    이상치 클리핑 (제거 대신 상한/하한으로 대체)
    """
    if self.config['FEATURES']['USE_WINSORIZATION'] != 'Y':
        return df
    
    from scipy.stats.mstats import winsorize
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        df[col] = winsorize(df[col], limits=[0.01, 0.01])
        # 하위 1%, 상위 1%를 1st, 99th percentile 값으로 대체
    
    return df
```

**효과**:
- 이상치 영향 감소
- 데이터 손실 없음 (제거 대신 클리핑)

---

## 5. 데이터 품질 검증

### NaN 분석

```python
# DataProcessor
def _export_nan_analysis(X, y, logger):
    """
    NaN 분석 리포트 생성
    """
    nan_ratio = X.isnull().sum() / len(X)
    
    print("\n=== NaN Analysis ===")
    print(f"Total columns: {len(X.columns)}")
    print(f"Columns with NaN: {(nan_ratio > 0).sum()}")
    print(f"Columns with >80% NaN: {(nan_ratio > 0.8).sum()}")
    
    # 상위 20개 NaN 컬럼
    top_20_nan = nan_ratio.sort_values(ascending=False).head(20)
    print("\nTop 20 columns with highest NaN ratio:")
    for col, ratio in top_20_nan.items():
        print(f"{col}: {ratio:.2%}")
    
    # CSV 저장
    nan_ratio.to_csv('nan_analysis.csv')
```

**예시 출력**:
```
=== NaN Analysis ===
Total columns: 532
Columns with NaN: 128
Columns with >80% NaN: 23

Top 20 columns with highest NaN ratio:
close__agg_autocorrelation__f_agg_"var"__maxlag_40: 95.2%
close__partial_autocorrelation__lag_10: 92.1%
close__ar_coefficient__coeff_10__k_10: 89.3%
...
```

### Infinite 값 검증

```python
def _check_infinite_values(X, logger):
    """
    Infinite 값 검사
    """
    inf_mask = np.isinf(X.select_dtypes(include=[np.number]))
    inf_count = inf_mask.sum().sum()
    
    if inf_count > 0:
        print(f"⚠️ WARNING: {inf_count} infinite values detected!")
        
        # 컬럼별 Infinite 개수
        inf_by_col = inf_mask.sum()
        inf_by_col = inf_by_col[inf_by_col > 0].sort_values(ascending=False)
        
        print("\nColumns with infinite values:")
        for col, count in inf_by_col.items():
            print(f"{col}: {count}")
    else:
        print("✅ No infinite values detected")
```

---

## 6. 미래 유출 방지

### Filing Date 기준 Cutoff

**문제**:
```
Q1 2024 재무제표:
- 분기 종료일 (report_date): 2024-03-31
- 공시일 (filingDate): 2024-05-15 (45일 후)

만약 report_date 기준으로 사용하면:
→ 2024-04-01에 Q1 데이터 사용 가능 (미래 유출! ❌)

Filing Date 기준:
→ 2024-05-16 이후에만 Q1 데이터 사용 가능 (정상 ✅)
```

**구현**:
```python
# make_mldata.py
indices = np.searchsorted(date_index, fs_metrics['filingDate'], side='right')
fs_metrics['rebalance_date'] = date_index[indices]
```

**검증**:
```python
# 공시 지연 분석
filling_delay_days = (
    pd.to_datetime(current_quarter_data['filingDate']) -
    pd.to_datetime(current_quarter_data['report_date'])
).dt.days

print(f"Filing delay (days): {filling_delay_days.describe()}")
# count    10000
# mean       45.2
# std        15.3
# min        30
# 25%        42
# 50%        45
# 75%        50
# max        90
```

### Walk-Forward 백테스트

```python
# ml_backtest.py
def _get_available_data_until(self, cutoff_date):
    """
    cutoff_date까지 사용 가능한 데이터만 로드
    """
    # Filing Date <= cutoff_date인 데이터만 로드
    df = load_all_ml_data()
    df_available = df[df['filingDate'] <= cutoff_date]
    
    return df_available
```

---

## 7. 문제점 및 개선안

### 문제 1: 메모리 사용량

**현재**:
- VIEW 테이블 구축 시 모든 종목 동시 로드
- 메모리 사용량: ~10GB

**개선안**:
```python
# 청크 단위 처리
def rebuild_table_view_chunked(self, chunk_size=100):
    for i in range(0, len(stocks), chunk_size):
        chunk_stocks = stocks[i:i+chunk_size]
        df_chunk = self._build_view_chunk(chunk_stocks)
        save_parquet(df_chunk, f'views/price_chunk_{i}.parquet')
    
    # 최종 통합
    merge_parquet_files('views/price_chunk_*.parquet', 'views/price.parquet')
```

### 문제 2: Feature 수 과다

**현재 상황**:
- tsfresh: 수백 개 Feature (시계열 통계량)
- 재무 비율: 수십 개
- 총: 500~1000개
- **문제점**:
  - 과적합 위험 (샘플 수 대비 feature 수 과다)
  - 학습 시간 증가
  - 노이즈 feature가 유의미한 신호 희석
  - Curse of Dimensionality

---

## Feature Selection 방법론 종합

### 1. Filter Methods (통계 기반 - 가장 빠름)

**1.1 Variance Threshold**
```python
from sklearn.feature_selection import VarianceThreshold

# 분산이 거의 없는 feature 제거 (상수에 가까운 feature)
selector = VarianceThreshold(threshold=0.01)
X_reduced = selector.fit_transform(X)

print(f"Features removed: {X.shape[1] - X_reduced.shape[1]}")
```

**장점**: 매우 빠름, 계산 간단
**단점**: 타겟과의 관계 무시
**추천**: 전처리 단계에서 1차 필터링용

---

**1.2 Correlation-Based Selection**
```python
# 타겟과의 상관관계 기반
correlations = X.corrwith(y).abs()
top_50_features = correlations.nlargest(50).index

X_reduced = X[top_50_features]
```

**장점**: 직관적, 빠름
**단점**:
- 선형 관계만 포착
- Feature 간 상호작용 무시
- 중복 feature 제거 안 됨

---

**1.3 Mutual Information (비선형 관계 포착)**
```python
from sklearn.feature_selection import mutual_info_regression

# 비선형 관계도 포착 가능
mi_scores = mutual_info_regression(X, y)
mi_scores = pd.Series(mi_scores, index=X.columns)

top_50_features = mi_scores.nlargest(50).index
X_reduced = X[top_50_features]
```

**장점**: 비선형 관계 포착, 엔트로피 기반
**단점**: 계산 비용 높음, Feature 간 중복 미고려
**추천**: Correlation과 함께 사용하여 보완

---

### 2. Wrapper Methods (모델 기반 반복 - 정확하지만 느림)

**2.1 Recursive Feature Elimination (RFE)**
```python
from sklearn.feature_selection import RFE
from xgboost import XGBRegressor

# 재귀적으로 feature 제거
estimator = XGBRegressor(max_depth=6)
selector = RFE(estimator, n_features_to_select=50, step=10)
selector.fit(X_train, y_train)

X_reduced = X_train[:, selector.support_]
feature_ranking = pd.Series(selector.ranking_, index=X.columns)
```

**장점**:
- 모델 성능 직접 최적화
- Feature 간 상호작용 고려
**단점**:
- 매우 느림 (많은 재학습 필요)
- 과적합 위험
**추천**: 최종 fine-tuning 단계에서 사용

---

**2.2 Sequential Feature Selection**
```python
from sklearn.feature_selection import SequentialFeatureSelector

# Forward Selection (50개까지 추가)
sfs = SequentialFeatureSelector(
    estimator=XGBRegressor(),
    n_features_to_select=50,
    direction='forward',
    cv=5
)
sfs.fit(X_train, y_train)
```

**장점**: CV 기반으로 일반화 성능 최적화
**단점**: RFE보다 더 느림
**추천**: 소규모 feature set (< 100개)에서만 사용

---

### 3. Embedded Methods (모델 내장 - **추천 ⭐**)

**3.1 Tree-Based Feature Importance (XGBoost/LightGBM)**
```python
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# XGBoost Feature Importance
xgb = XGBRegressor(max_depth=8, n_estimators=100)
xgb.fit(X_train, y_train)

importances = pd.Series(xgb.feature_importances_, index=X.columns)
top_50_features = importances.nlargest(50).index

X_reduced = X_train[top_50_features]

# 중요도 플롯 저장
import matplotlib.pyplot as plt
importances.nlargest(30).plot(kind='barh', figsize=(10, 8))
plt.title('Top 30 Feature Importances')
plt.tight_layout()
plt.savefig('feature_importance.png')
```

**장점**:
- 현재 사용 중인 XGBoost/LGBM과 일관성
- 빠르고 정확
- 비선형 관계 포착
- Feature 간 상호작용 자동 고려
**단점**:
- 트리 깊이에 민감
- Ensemble해야 안정적
**추천**: **1차 추천 방법** ⭐

---

**3.2 L1 Regularization (Lasso)**
```python
from sklearn.linear_model import LassoCV

# Cross-validation으로 최적 alpha 자동 탐색
lasso = LassoCV(cv=5, alphas=np.logspace(-4, 1, 50))
lasso.fit(X_train, y_train)

# 0이 아닌 계수를 가진 feature 선택
selected_features = X.columns[lasso.coef_ != 0]
X_reduced = X_train[selected_features]

print(f"Features selected: {len(selected_features)}")
print(f"Optimal alpha: {lasso.alpha_}")
```

**장점**:
- 자동으로 feature selection (계수 → 0)
- 다중공선성 처리
- Sparse solution
**단점**:
- 선형 관계 가정
- 상관관계 높은 feature 중 하나만 선택
**추천**: Tree-based와 교차 검증용

---

### 4. Advanced Methods (통계적 엄밀성 - **추천 ⭐**)

**4.1 Boruta Algorithm (통계적 유의성 검증)**
```python
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor

# Shadow feature와 비교하여 통계적으로 유의미한 feature만 선택
rf = RandomForestRegressor(n_jobs=-1, max_depth=7)
boruta = BorutaPy(
    estimator=rf,
    n_estimators='auto',
    max_iter=100,
    random_state=42
)
boruta.fit(X_train.values, y_train.values)

# 확정 feature + 잠정 feature
selected_features = X.columns[boruta.support_]  # 확정
tentative_features = X.columns[boruta.support_weak_]  # 잠정

X_reduced = X_train[selected_features]

print(f"Confirmed: {len(selected_features)}")
print(f"Tentative: {len(tentative_features)}")
print(f"Rejected: {(~boruta.support_).sum()}")
```

**장점**:
- 통계적으로 유의미한 feature만 선택
- False positive 최소화
- 모든 유의미한 feature 보존 (사용자 지정 개수 X)
**단점**:
- 느림 (100+ iterations)
- Feature 개수를 사전 지정할 수 없음
**추천**: **과학적 정당성이 중요한 경우 추천** ⭐

---

**4.2 SHAP Feature Selection**
```python
import shap

# XGBoost 학습
xgb = XGBRegressor()
xgb.fit(X_train, y_train)

# SHAP values 계산
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_train)

# 평균 절대 SHAP value로 중요도 측정
shap_importance = np.abs(shap_values).mean(axis=0)
shap_df = pd.DataFrame({
    'feature': X.columns,
    'shap_importance': shap_importance
}).sort_values('shap_importance', ascending=False)

# Top 50 선택
top_50_features = shap_df.head(50)['feature'].tolist()
X_reduced = X_train[top_50_features]

# SHAP Summary Plot
shap.summary_plot(shap_values, X_train, show=False)
plt.tight_layout()
plt.savefig('shap_summary.png')
```

**장점**:
- 해석 가능성 최고 (Shapley value 기반)
- Feature 간 상호작용 고려
- 글로벌/로컬 설명 가능
**단점**:
- 계산 비용 높음
- 대규모 데이터에서 느림
**추천**: 최종 모델 해석 및 검증용

---

### 5. Ensemble Feature Selection (다수결 투표)

```python
# 여러 방법론의 결과를 종합
def ensemble_feature_selection(X, y, top_k=50):
    results = {}

    # 1. XGBoost Importance
    xgb = XGBRegressor()
    xgb.fit(X, y)
    results['xgb'] = pd.Series(xgb.feature_importances_, index=X.columns)

    # 2. LightGBM Importance
    lgbm = LGBMRegressor()
    lgbm.fit(X, y)
    results['lgbm'] = pd.Series(lgbm.feature_importances_, index=X.columns)

    # 3. Mutual Information
    mi = mutual_info_regression(X, y)
    results['mi'] = pd.Series(mi, index=X.columns)

    # 4. Lasso
    lasso = LassoCV(cv=5)
    lasso.fit(X, y)
    results['lasso'] = pd.Series(np.abs(lasso.coef_), index=X.columns)

    # 각 방법론에서 Top-K 투표
    votes = pd.Series(0, index=X.columns)
    for method, scores in results.items():
        top_features = scores.nlargest(top_k).index
        votes[top_features] += 1

    # 2개 이상 방법론에서 선택된 feature
    selected_features = votes[votes >= 2].sort_values(ascending=False).index

    return selected_features, votes

selected_features, votes = ensemble_feature_selection(X_train, y_train, top_k=50)
X_reduced = X_train[selected_features]

print(f"Features selected by ensemble: {len(selected_features)}")
print(f"\nVote distribution:")
print(votes.value_counts().sort_index(ascending=False))
# 4 votes: 25 features
# 3 votes: 30 features
# 2 votes: 45 features
```

**장점**:
- 여러 관점에서 검증
- 노이즈 feature 제거
- Robust
**단점**:
- 계산 비용 높음
- 복잡함
**추천**: **Production 환경에서 추천** ⭐

---

## 추천 워크플로우 (3-Stage Pipeline)

```python
# ===== Stage 1: Fast Filter (1분) =====
# 명백한 노이즈 제거
from sklearn.feature_selection import VarianceThreshold

# 1.1 분산 거의 없는 feature 제거
selector = VarianceThreshold(threshold=0.01)
X_stage1 = selector.fit_transform(X)

# 1.2 타겟과 상관관계 거의 없는 feature 제거
correlations = pd.DataFrame(X_stage1).corrwith(y).abs()
X_stage1 = X_stage1[:, correlations > 0.01]

print(f"Stage 1: {X.shape[1]} → {X_stage1.shape[1]} features")
# Stage 1: 532 → 320 features

# ===== Stage 2: Tree-Based Selection (10분) =====
# XGBoost + LightGBM 앙상블

xgb = XGBRegressor(max_depth=8, n_estimators=100)
xgb.fit(X_stage1, y)
xgb_imp = pd.Series(xgb.feature_importances_, index=range(X_stage1.shape[1]))

lgbm = LGBMRegressor(max_depth=8, n_estimators=100)
lgbm.fit(X_stage1, y)
lgbm_imp = pd.Series(lgbm.feature_importances_, index=range(X_stage1.shape[1]))

# 두 모델 중요도 평균
avg_importance = (xgb_imp + lgbm_imp) / 2
top_100_idx = avg_importance.nlargest(100).index
X_stage2 = X_stage1[:, top_100_idx]

print(f"Stage 2: {X_stage1.shape[1]} → {X_stage2.shape[1]} features")
# Stage 2: 320 → 100 features

# ===== Stage 3: Boruta Validation (30분) =====
# 통계적으로 유의미한 feature만 최종 선택

from boruta import BorutaPy

rf = RandomForestRegressor(n_jobs=-1, max_depth=7)
boruta = BorutaPy(rf, n_estimators='auto', max_iter=100)
boruta.fit(X_stage2, y)

X_final = X_stage2[:, boruta.support_]

print(f"Stage 3: {X_stage2.shape[1]} → {X_final.shape[1]} features")
# Stage 3: 100 → 45 features

print(f"\n=== Final Result ===")
print(f"Total reduction: {X.shape[1]} → {X_final.shape[1]} ({X_final.shape[1]/X.shape[1]*100:.1f}%)")
# Total reduction: 532 → 45 (8.5%)
```

---

## 최종 추천안 ⭐

### 방법 A: 빠르고 실용적 (Production 추천)
```python
# XGBoost + LightGBM 앙상블 (10분)
# → 현재 모델과 일관성 유지
# → 빠르고 효과적
# → 즉시 적용 가능
```

**Target**: 532 → 50~80 features
**소요 시간**: ~10분
**장점**:
- 현재 파이프라인과 완벽한 일관성
- 즉시 적용 가능
- 빠른 실험 iteration

### 방법 B: 과학적으로 엄밀 (학술/검증 추천)
```python
# 3-Stage Pipeline (Variance → Tree → Boruta)
# → 통계적 정당성 확보
# → False positive 최소화
# → 논문/보고서에 사용 가능
```

**Target**: 532 → 30~60 features (데이터 의존적)
**소요 시간**: ~40분
**장점**:
- 통계적으로 검증된 feature만 선택
- 과적합 최소화
- 해석 가능성 높음

### 방법 C: 하이브리드 (최고 성능 추구)
```python
# Ensemble Voting (4가지 방법론)
# → 여러 관점에서 검증
# → 가장 robust
# → 성능 최우선
```

**Target**: 532 → 60~100 features
**소요 시간**: ~20분
**장점**:
- 다양한 관점에서 검증
- Noise 최소화
- 안정적

---

## 구현 우선순위

1. **Week 1**: 방법 A (XGBoost+LightGBM) 구현 및 백테스트
2. **Week 2**: 방법 B (3-Stage) 구현 및 성능 비교
3. **Week 3**: SHAP 기반 해석성 분석
4. **Week 4**: 최종 방법 선정 및 Production 배포

---

## 성능 검증 체크리스트

Feature selection 후 반드시 확인:
- [ ] 학습 데이터 RMSE 유지 또는 소폭 상승 (과적합 감소)
- [ ] 검증 데이터 RMSE **감소** (일반화 성능 개선)
- [ ] 백테스트 Sharpe Ratio 개선
- [ ] 학습 시간 단축 (50% 이상)
- [ ] 선택된 feature의 경제적 해석 가능성

### 문제 3: 증분 업데이트 미지원

**현재**:
- 데이터 업데이트 시 전체 재구축 필요
- 시간 소요: ~2시간

**개선안**:
```python
# 증분 업데이트
def update_incremental(self, last_update_date):
    # 1. last_update_date 이후 데이터만 수집
    new_data = collect_fmp_data(start_date=last_update_date)
    
    # 2. 기존 VIEW 테이블 로드
    existing_view = load_parquet('views/price.parquet')
    
    # 3. 새 데이터 추가
    updated_view = pd.concat([existing_view, new_data])
    updated_view = updated_view.drop_duplicates(subset=['symbol', 'date'])
    updated_view = updated_view.sort_values(['symbol', 'date'])
    
    # 4. 저장
    save_parquet(updated_view, 'views/price.parquet')
```

---

## 결론

### 강점
1. 엄격한 미래 유출 방지 (Filing Date 기준)
2. 체계적인 데이터 흐름 (FMP → Parquet → VIEW → ML Data)
3. Extreme Mover 필터링으로 노이즈 제거

### 약점
1. 메모리 사용량 높음
2. Feature 수 과다 (500~1000개)
3. 증분 업데이트 미지원

### 개선 우선순위
1. Feature 선택 (Top-50)
2. 청크 단위 처리
3. 증분 업데이트 지원

---

**다음 문서**: [03_ml_strategy.md](./03_ml_strategy.md)
