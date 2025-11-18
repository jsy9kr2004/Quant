# 개선 로드맵 (Improvement Roadmap)

## 📋 문서 헤더

**목적**: Quant 트레이딩 시스템의 현재 한계점을 분석하고, 체계적인 개선 계획을 제시하여 장기적으로 더욱 견고하고 수익성 높은 시스템으로 발전시키기 위한 전략 문서

**작성일**: 2025-11-17

**대상**:
- Quant 트레이딩 시스템 개발팀
- 백테스팅 및 라이브 트레이딩 운영자
- 시스템 아키텍처 담당자
- 데이터 엔지니어

**문서 버전**: 1.0

---

## 🎯 이 문서는 누구를 위한 것인가?

본 문서는 Quant 트레이딩 시스템의 **현재 한계점을 명확히 이해**하고, **단계별 개선 방안을 실행**하고자 하는 팀을 위해 작성되었습니다.

### 주요 대상과 용도

| 역할 | 사용 목적 |
|------|---------|
| **개발 리드** | 전체 로드맵 검토 및 팀 리소스 배분 결정 |
| **ML 엔지니어** | 모델 개선, 특성 공학, 하이퍼파라미터 최적화 작업 |
| **데이터 엔지니어** | 데이터 품질 개선, 대체 데이터 소스 통합 |
| **시스템 운영팀** | Risk Management, 실시간 모니터링 구축 |
| **투자자/이해관계자** | 시스템 로드맵 이해 및 개선 우선순위 파악 |

---

## 📑 목차

1. [현재 한계점](#현재-한계점)
   - [데이터 품질](#1-데이터-품질)
   - [Feature Engineering](#2-feature-engineering)
   - [모델 성능](#3-모델-성능)
   - [백테스팅 한계](#4-백테스팅-한계)
   - [실시간 트레이딩 지원 부족](#5-실시간-트레이딩-지원-부족)

2. [추가 필요 기능](#추가-필요-기능)
   - [Risk Management](#1-risk-management)
   - [포트폴리오 최적화](#2-포트폴리오-최적화)
   - [Time Series CV 강화](#3-time-series-cv-강화)
   - [Explainability](#4-explainability설명-가능성)
   - [대체 데이터 통합](#5-대체-데이터-통합)
   - [Regime Detection](#6-regime-detection시장-국면-감지)

3. [우선순위 제안](#우선순위-제안)
   - [단기 계획 (1-2주)](#단기-계획-1-2주)
   - [중기 계획 (1-2개월)](#중기-계획-1-2개월)
   - [장기 계획 (3-6개월)](#장기-계획-3-6개월)

4. [관련 문서 및 리소스](#관련-문서-및-리소스)

---

## 현재 한계점

현재 Quant 트레이딩 시스템은 End-to-End 자동화, 모듈화 설계, 앙상블 전략 등 여러 강점을 갖추고 있습니다.
그러나 실제 운영 과정에서 다음과 같은 한계점들이 식별되었으며, 이를 단계적으로 개선할 필요가 있습니다.

### 1. 데이터 품질

#### 문제점

- **결측치**: FMP API의 데이터 결측치가 많음 (특히 소형주, 신규상장 종목)
- **정확성 오류**: 일부 재무지표가 부정확함 (API 제공사의 데이터 오류)
- **상장폐지 종목**: 상장폐지 후 데이터가 불완전하여 학습에 영향

#### 개선 방향

```python
# 다중 데이터 소스 통합 전략
from data_collector.multi_source import MultiSourceCollector

# 1단계: Primary 소스 (FMP) + Secondary 소스 (Yahoo Finance, Alpha Vantage)
collector = MultiSourceCollector(
    primary_source='fmp',
    fallback_sources=['yahoo', 'alpha_vantage'],
    validation_rules={
        'missing_threshold': 0.05,  # 5% 이상 결측은 제외
        'outlier_detection': 'iqr'   # IQR 기반 이상치 탐지
    }
)

# 2단계: 데이터 검증 및 보정
from validation.data_validator import DataValidator

validator = DataValidator()
validated_data = validator.validate_and_impute(
    raw_data,
    method='interpolation',  # 선형 보간
    outlier_method='winsorize'  # 극값 조정 (상위/하위 5% 절삭)
)

# 3단계: 이상치 자동 탐지 및 알림
from monitoring.anomaly_detector import AnomalyDetector

anomaly_detector = AnomalyDetector(method='isolation_forest')
anomalies = anomaly_detector.detect(validated_data)

if anomalies.found:
    logger.warning(f"Data anomalies detected: {anomalies.details}")
    # 관리자 알림 발송
```

**기대 효과**:
- 데이터 품질 향상으로 모델 성능 3-5% 개선
- 결측치 처리로 학습 샘플 증가
- 신뢰할 수 있는 예측 생성

---

### 2. Feature Engineering

#### 문제점

- **고정된 Features**: 현재 139개 재무비율 + 36개 시계열 지표로 고정되어 있음
- **Feature Selection 미흡**: 중요도가 낮은 특성이 다수 포함되어 있음
- **Sector/Industry 미활용**: 섹터별로 다른 중요 지표를 동일하게 적용 중

#### 개선 방향

```python
# 1단계: Feature Selection을 통한 차원 축소
from feature_engineering.feature_selector import FeatureSelector

# 중요도 기반 선택 (threshold 이상의 feature만 유지)
selector = FeatureSelector(
    method='importance',  # 또는 'permutation', 'shap'
    threshold=0.01,
    estimator='xgboost'
)

X_selected = selector.fit_transform(X_train, y_train)
print(f"Reduced features: {X_train.shape[1]} → {X_selected.shape[1]}")
# 예상 결과: 175개 → 80-100개로 축소

# 2단계: 자동화된 Feature Engineering (Featuretools)
import featuretools as ft

# 엔티티와 관계 정의
es = ft.EntitySet(id="stock_data")
es.add_dataframe(dataframe_name="stocks", dataframe=stock_df, index="stock_id")
es.add_dataframe(dataframe_name="financials", dataframe=financial_df, index="financial_id")

# 자동으로 새로운 특성 생성
feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_dataframe_name="stocks",
    max_depth=2,
    trans_primitives=['sum', 'mean', 'std', 'max', 'min'],
    agg_primitives=['sum', 'mean', 'std']
)

# 3단계: Sector별 특화 Features
from feature_engineering.sector_features import SectorFeatureEngine

sector_engine = SectorFeatureEngine(sector_mapping=SECTOR_MAPPING)

for sector in SECTORS:
    sector_features = sector_engine.get_sector_features(
        sector=sector,
        include_global=True,
        custom_features={
            'tech': ['roe_volatility', 'rd_to_revenue'],
            'finance': ['equity_ratio', 'interest_coverage'],
            'retail': ['inventory_turnover', 'gross_margin_trend']
        }
    )
    X_sector_enhanced = pd.concat([X, sector_features], axis=1)
```

**기대 효과**:
- 특성 수 50% 감소로 학습 속도 2배 향상
- 중요 특성 집중으로 모델 해석성 향상
- 섹터별 최적화로 예측 정확도 2-3% 개선

---

### 3. 모델 성능

#### 문제점

- **분류 정확도**: 현재 55-60% (충분하지 않음)
- **회귀 오차**: RMSE 개선 여지가 있음
- **변동성 시기 약세**: 시장 변동성이 높을 때 성능이 현저히 저하됨

#### 개선 방향

```python
# 1단계: Stacking Ensemble로 성능 향상
from models.ensemble import StackingEnsemble
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Base learners: 다양한 알고리즘 조합
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=10)),
    ('xgb', XGBClassifier(n_estimators=100, learning_rate=0.1)),
    ('lgb', LGBMClassifier(n_estimators=100, learning_rate=0.1))
]

# Meta learner: 기본 모델의 예측을 결합
ensemble = StackingEnsemble(
    base_models=base_models,
    meta_learner='ridge',  # Ridge Regression
    cv=5,
    verbose=1
)

# 학습 및 예측
ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)

print(f"Ensemble Performance:")
print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"  ROC-AUC: {roc_auc_score(y_test, y_pred_proba[:, 1]):.4f}")

# 2단계: Optuna를 이용한 하이퍼파라미터 전역 최적화
import optuna
from optuna.samplers import TPESampler

def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0)
    }

    model = LGBMClassifier(**params, n_estimators=100)
    cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    return cv_score.mean()

# 최적화 실행 (200 trials)
sampler = TPESampler(seed=42)
study = optuna.create_study(
    direction='maximize',
    sampler=sampler
)
study.optimize(objective, n_trials=200, n_jobs=-1)

best_params = study.best_params
print(f"Best Parameters: {best_params}")
print(f"Best ROC-AUC: {study.best_value:.4f}")

# 3단계: Deep Learning 실험 (충분한 데이터 시)
import tensorflow as tf
from tensorflow import keras

# LSTM 모델 (시계열 예측)
model = keras.Sequential([
    keras.layers.LSTM(64, activation='relu', input_shape=(seq_length, n_features)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation='sigmoid')  # 이진 분류
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['auc']
)

history = model.fit(
    X_train_seq, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_auc', patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    ]
)
```

**기대 효과**:
- Stacking Ensemble로 정확도 3-5% 향상
- 하이퍼파라미터 최적화로 추가 2-3% 개선
- Deep Learning으로 시계열 패턴 포착으로 5-7% 향상 가능

---

### 4. 백테스팅 한계

#### 문제점

- **거래 비용 미반영**: 수수료, 슬리피지 등 실제 거래 비용이 고려되지 않음
- **유동성 제약 미고려**: 대량 매수/매도 시 시장에 미치는 영향 무시
- **Market Regime 미반영**: 불장/약장 등 시장 상황 변화에 따른 성능 편차 미반영

#### 개선 방향

```python
# 1단계: 거래 비용 모델링
class RealisticBacktester:
    def __init__(self,
                 commission_rate=0.001,  # 0.1% 수수료
                 slippage_rate=0.002,    # 0.2% 슬리피지
                 bid_ask_spread=0.0005): # 0.05% 호가 차
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.bid_ask_spread = bid_ask_spread

    def calculate_entry_price(self, market_price):
        """진입 가격 계산"""
        # Slippage 추가
        actual_price = market_price * (1 + self.slippage_rate)
        # Bid-Ask Spread 추가 (매수 시)
        actual_price = actual_price * (1 + self.bid_ask_spread)
        return actual_price

    def calculate_exit_price(self, market_price):
        """청산 가격 계산"""
        # Slippage 추가
        actual_price = market_price * (1 - self.slippage_rate)
        # Bid-Ask Spread 추가 (매도 시)
        actual_price = actual_price * (1 - self.bid_ask_spread)
        return actual_price

    def calculate_transaction_costs(self, entry_price, exit_price, volume):
        """거래 비용 계산"""
        entry_cost = entry_price * volume * self.commission_rate
        exit_cost = exit_price * volume * self.commission_rate
        return entry_cost + exit_cost

    def backtest(self, signals, prices, volumes):
        """비용이 반영된 백테스트"""
        gross_return = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]

        # 거래 비용 계산
        entry_actual = self.calculate_entry_price(prices.iloc[0])
        exit_actual = self.calculate_exit_price(prices.iloc[-1])
        transaction_costs = self.calculate_transaction_costs(
            entry_actual, exit_actual, volumes.mean()
        )

        # 순 수익률
        net_return = gross_return - (transaction_costs / prices.iloc[0])

        return {
            'gross_return': gross_return,
            'transaction_costs': transaction_costs,
            'net_return': net_return
        }

backtester = RealisticBacktester(
    commission_rate=0.001,
    slippage_rate=0.002,
    bid_ask_spread=0.0005
)

# 2단계: 유동성 필터 및 Volume 기반 포지션 크기 조정
class LiquidityFilter:
    def __init__(self, min_avg_volume=1_000_000, min_volume_percentile=50):
        """
        유동성 필터 설정
        - min_avg_volume: 최소 평균 거래량 (달러)
        - min_volume_percentile: 최소 거래량 백분위수
        """
        self.min_avg_volume = min_avg_volume
        self.min_volume_percentile = min_volume_percentile

    def filter_liquid_stocks(self, stocks_df):
        """유동성 조건을 만족하는 종목만 선택"""
        # 평균 거래대금 계산
        stocks_df['avg_dollar_volume'] = (
            stocks_df['volume'] * stocks_df['close']
        ).rolling(20).mean()

        # 최소 유동성 기준 초과 종목만 선택
        liquid_stocks = stocks_df[
            stocks_df['avg_dollar_volume'] >= self.min_avg_volume
        ]

        return liquid_stocks

    def adjust_position_size(self, position_value, avg_volume, max_position_pct=0.01):
        """일일 거래량의 1% 이상 거래하지 않도록 포지션 크기 조정"""
        max_position = avg_volume * max_position_pct
        return min(position_value, max_position)

liquidity_filter = LiquidityFilter(
    min_avg_volume=500_000,
    min_volume_percentile=50
)

# 3단계: Market Regime 감지 기반 동적 전략 조정
from market_analysis.regime_detector import RegimeDetector

regime_detector = RegimeDetector(
    method='hmm',  # Hidden Markov Model
    n_regimes=3,   # Bull, Neutral, Bear
    lookback=252   # 1년 데이터로 학습
)

regimes = regime_detector.fit_predict(price_history)

# Regime별 전략 파라미터 조정
regime_params = {
    'bull': {'aggressive_factor': 1.5, 'stop_loss': -0.15, 'take_profit': 0.30},
    'neutral': {'aggressive_factor': 1.0, 'stop_loss': -0.10, 'take_profit': 0.15},
    'bear': {'aggressive_factor': 0.5, 'stop_loss': -0.05, 'take_profit': 0.08}
}

current_regime = regime_detector.get_current_regime()
params = regime_params[current_regime]
```

**기대 효과**:
- 거래 비용 반영으로 백테스트 결과가 3-8% 하향 조정 (더 현실적)
- 유동성 필터로 슬리피지 증가 방지
- Regime 감지로 시장 상황에 맞는 전략 조정으로 Sharpe Ratio 10-20% 향상

---

### 5. 실시간 트레이딩 지원 부족

#### 문제점

- **시간 단위 제한**: 현재 일별(일중) 가격만 지원, 분/초 단위 미지원
- **실시간 데이터 수집 없음**: 정적 데이터만 사용, 실시간 수집 불가
- **자동 주문 기능 없음**: 예측 후 수동으로 매매해야 함

#### 개선 방향

```python
# 1단계: WebSocket 기반 실시간 데이터 수집
import asyncio
import websocket
from datetime import datetime

class RealtimeDataCollector:
    def __init__(self, broker_api_key, symbols):
        self.api_key = broker_api_key
        self.symbols = symbols
        self.data_buffer = {}

    async def connect_websocket(self):
        """WebSocket 연결"""
        ws = websocket.WebSocketApp(
            "wss://data.alpaca.markets/stream",
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        ws.on_open = self.on_open

        ws.run_forever()

    def on_message(self, ws, msg):
        """데이터 수신"""
        import json
        data = json.loads(msg)

        if 'T' in data:  # Trade data
            symbol = data['S']
            price = data['p']
            volume = data['s']
            timestamp = data['t']

            # 버퍼에 저장
            if symbol not in self.data_buffer:
                self.data_buffer[symbol] = []

            self.data_buffer[symbol].append({
                'timestamp': timestamp,
                'price': price,
                'volume': volume
            })

            # 버퍼가 크면 DB에 저장
            if len(self.data_buffer[symbol]) >= 100:
                self.save_to_db(symbol, self.data_buffer[symbol])
                self.data_buffer[symbol] = []

# 2단계: 자동 매매 시스템 (Interactive Brokers / Alpaca)
from ib_insync import IB, Stock, Order

class AutoTradingSystem:
    def __init__(self, broker='interactive_brokers'):
        self.broker = broker
        if broker == 'interactive_brokers':
            self.ib = IB()
            self.ib.connect('127.0.0.1', 7497, clientId=1)
        elif broker == 'alpaca':
            from alpaca_trade_api import REST
            self.api = REST(key_id, secret_key, base_url)

    def place_order(self, symbol, quantity, price, order_type='market'):
        """주문 생성"""
        if self.broker == 'interactive_brokers':
            contract = Stock(symbol, 'SMART', 'USD')

            order = Order()
            order.action = 'BUY' if quantity > 0 else 'SELL'
            order.totalQuantity = abs(quantity)
            order.orderType = order_type
            if order_type == 'limit':
                order.lmtPrice = price

            trade = self.ib.placeOrder(contract, order)
            return trade

        elif self.broker == 'alpaca':
            return self.api.submit_order(
                symbol=symbol,
                qty=quantity,
                side='buy' if quantity > 0 else 'sell',
                type=order_type,
                time_in_force='day',
                limit_price=price if order_type == 'limit' else None
            )

    def get_portfolio_value(self):
        """포트폴리오 가치 조회"""
        if self.broker == 'interactive_brokers':
            account_values = self.ib.accountSummary()
            portfolio_value = next(
                v for v in account_values
                if v.tag == 'TotalCashValue'
            ).value
        elif self.broker == 'alpaca':
            account = self.api.get_account()
            portfolio_value = account.portfolio_value

        return float(portfolio_value)

    def get_positions(self):
        """현재 포지션 조회"""
        if self.broker == 'interactive_brokers':
            return self.ib.positions()
        elif self.broker == 'alpaca':
            return self.api.list_positions()

# 3단계: 실시간 모니터링 시스템
from monitoring.performance_monitor import PerformanceMonitor
import logging

class RealtimeMonitor:
    def __init__(self, trading_system, alert_config):
        self.trading_system = trading_system
        self.alert_config = alert_config
        self.monitor = PerformanceMonitor()
        self.logger = logging.getLogger(__name__)

    async def monitor_portfolio(self):
        """포트폴리오 실시간 모니터링"""
        while True:
            # 포트폴리오 정보 수집
            portfolio_value = self.trading_system.get_portfolio_value()
            positions = self.trading_system.get_positions()

            # 성능 추적
            self.monitor.track_portfolio_value(portfolio_value)

            # Drawdown 확인
            current_drawdown = self.monitor.calculate_drawdown()
            if current_drawdown > self.alert_config['max_drawdown']:
                self._send_alert(
                    level='WARNING',
                    message=f"Drawdown exceeded: {current_drawdown:.2%}"
                )

            # 포지션별 손실 모니터링
            for position in positions:
                unrealized_pnl = position.unrealized_pnl
                pnl_pct = unrealized_pnl / (position.market_value * position.avg_fill_price)

                if pnl_pct < self.alert_config['position_stop_loss']:
                    self._send_alert(
                        level='CRITICAL',
                        message=f"{position.symbol}: PnL {pnl_pct:.2%}"
                    )
                    # 자동 손절
                    self.trading_system.place_order(
                        position.symbol,
                        -position.quantity,
                        order_type='market'
                    )

            await asyncio.sleep(60)  # 1분마다 확인

    def _send_alert(self, level, message):
        """알림 발송"""
        self.logger.log(level, message)
        # Email, SMS, Slack 등으로 알림 발송
```

**기대 효과**:
- 실시간 데이터로 고빈도 거래 가능 (분 단위 이상)
- 자동 매매로 실행 지연 없음 (수동 거래 제거)
- 실시간 모니터링으로 위험 상황에 즉시 대응

---

## 추가 필요 기능

위의 한계점 개선 외에도, 다음과 같은 추가 기능들이 시스템을 더욱 견고하고 수익성 있게 만들 것으로 기대됩니다.

### 1. Risk Management

#### 현재 상태
- 단순 Top-K 선정으로 동일 가중치 적용
- 포지션 크기 제한 없음
- 손절/익절 메커니즘 부재

#### 개선 방향

```python
from portfolio_optimization.risk_manager import RiskManager

class EnhancedRiskManager:
    def __init__(self,
                 max_position_size=0.05,      # 종목당 최대 5%
                 max_sector_exposure=0.30,     # 섹터당 최대 30%
                 stop_loss=0.10,               # 10% 손절
                 take_profit=0.20,             # 20% 익절
                 max_portfolio_beta=1.2):      # 포트폴리오 베타 상한
        self.max_position_size = max_position_size
        self.max_sector_exposure = max_sector_exposure
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_portfolio_beta = max_portfolio_beta

    def allocate_weights(self, predictions, prices, sectors, market_caps):
        """위험 조정 가중치 배분"""
        # 1단계: 예측 신뢰도 기반 가중치
        weights = predictions.values.copy()
        weights = weights / weights.sum()  # 정규화

        # 2단계: 시가총액 기반 조정 (소형주 감소)
        market_cap_factor = np.sqrt(market_caps / market_caps.mean())
        weights = weights * market_cap_factor
        weights = weights / weights.sum()

        # 3단계: 포지션 크기 제한
        max_weight = self.max_position_size
        weights = np.minimum(weights, max_weight)
        weights = weights / weights.sum()

        # 4단계: 섹터 노출 제한
        for sector in sectors.unique():
            sector_mask = (sectors == sector)
            sector_weight = weights[sector_mask].sum()
            if sector_weight > self.max_sector_exposure:
                # 섹터 내에서 동일 가중치로 조정
                sector_weights = weights[sector_mask] * (
                    self.max_sector_exposure / sector_weight
                )
                weights[sector_mask] = sector_weights

        weights = weights / weights.sum()

        return weights

    def set_stop_loss_and_take_profit(self, entry_price, symbol, quantity):
        """손절/익절 주문 자동 설정"""
        stop_loss_price = entry_price * (1 - self.stop_loss)
        take_profit_price = entry_price * (1 + self.take_profit)

        return {
            'symbol': symbol,
            'quantity': quantity,
            'entry_price': entry_price,
            'stop_loss_price': stop_loss_price,
            'take_profit_price': take_profit_price,
            'max_loss': entry_price * quantity * self.stop_loss,
            'potential_profit': entry_price * quantity * self.take_profit
        }

risk_mgr = EnhancedRiskManager(
    max_position_size=0.05,
    max_sector_exposure=0.30,
    stop_loss=0.10,
    take_profit=0.20
)

# 사용 예시
portfolio_weights = risk_mgr.allocate_weights(
    predictions=model_predictions,
    prices=current_prices,
    sectors=stock_sectors,
    market_caps=stock_market_caps
)

for symbol, weight in portfolio_weights.items():
    entry_price = current_prices[symbol]
    quantity = int(portfolio_value * weight / entry_price)

    exit_orders = risk_mgr.set_stop_loss_and_take_profit(
        entry_price, symbol, quantity
    )
```

**기대 효과**:
- 포지션 제한으로 최악의 손실 통제
- 손절/익절로 감정적 거래 방지
- 섹터 분산으로 시스템 리스크 감소

---

### 2. 포트폴리오 최적화

#### 고도화된 최적화 알고리즘

```python
from portfolio_optimization.optimizer import PortfolioOptimizer
import numpy as np
from scipy.optimize import minimize

class AdvancedPortfolioOptimizer:
    """고급 포트폴리오 최적화"""

    def __init__(self, method='mean_variance'):
        self.method = method

    def mean_variance_optimization(self, expected_returns, cov_matrix, risk_aversion=2.5):
        """
        Markowitz Mean-Variance Optimization

        목적함수: maximize (w^T * μ) - (λ/2) * (w^T * Σ * w)
        """
        n_assets = len(expected_returns)

        def objective(w):
            portfolio_return = w @ expected_returns
            portfolio_variance = w @ cov_matrix @ w
            return -(portfolio_return - risk_aversion * 0.5 * portfolio_variance)

        constraints = ({'type': 'eq', 'fun': lambda w: w.sum() - 1})  # w.sum() = 1
        bounds = tuple((0, 1) for _ in range(n_assets))  # 0 ≤ w_i ≤ 1

        x0 = np.array([1/n_assets] * n_assets)
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

        return result.x

    def black_litterman_optimization(self,
                                     market_cap_weights,
                                     expected_returns,
                                     cov_matrix,
                                     views,
                                     confidence_levels):
        """
        Black-Litterman Model
        주관적 견해(views)를 시장 암시 예상수익에 반영
        """
        # 시장 암시 수익률 계산
        lambda_param = 2.5  # Risk aversion
        implied_returns = lambda_param * cov_matrix @ market_cap_weights

        # 뷰를 처리하는 행렬 구성
        P = np.array([views])  # 뷰 행렬
        Q = np.array([expected_returns])  # 뷰의 기대수익
        Omega = np.diag(confidence_levels)  # 신뢰도 공분산 행렬

        # Black-Litterman 공식
        tau = 1 / len(market_cap_weights)
        BL_cov = cov_matrix
        P_transpose = P.T

        M = np.linalg.inv(
            np.linalg.inv(tau * BL_cov) + P_transpose @ np.linalg.inv(Omega) @ P
        )

        bl_returns = (
            np.linalg.inv(tau * BL_cov) @ implied_returns +
            P_transpose @ np.linalg.inv(Omega) @ Q
        ) @ M

        # 최적화
        return self.mean_variance_optimization(bl_returns, BL_cov)

    def risk_parity_optimization(self, cov_matrix):
        """
        Risk Parity (위험 균등 배분)
        각 자산이 포트폴리오 전체 위험에 같은 기여도
        """
        n_assets = cov_matrix.shape[0]

        def risk_contribution(w):
            """위험 기여도 계산"""
            portfolio_vol = np.sqrt(w @ cov_matrix @ w)
            marginal_contrib = cov_matrix @ w / portfolio_vol
            return w * marginal_contrib

        def objective(w):
            rc = risk_contribution(w)
            target_rc = 1 / n_assets
            return np.sum((rc - target_rc) ** 2)

        constraints = ({'type': 'eq', 'fun': lambda w: w.sum() - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))

        x0 = np.array([1/n_assets] * n_assets)
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

        return result.x

# 사용 예시
optimizer = AdvancedPortfolioOptimizer(method='mean_variance')

# 1. Mean-Variance 최적화
mv_weights = optimizer.mean_variance_optimization(
    expected_returns=model_predictions,
    cov_matrix=price_covariance,
    risk_aversion=2.5
)

# 2. Black-Litterman 최적화
bl_weights = optimizer.black_litterman_optimization(
    market_cap_weights=market_caps / market_caps.sum(),
    expected_returns=base_returns,
    cov_matrix=price_covariance,
    views=model_predictions,  # ML 모델의 예측을 뷰로 사용
    confidence_levels=prediction_confidence
)

# 3. Risk Parity 최적화
rp_weights = optimizer.risk_parity_optimization(
    cov_matrix=price_covariance
)
```

**기대 효과**:
- 위험-수익 트레이드오프 최적화로 Sharpe Ratio 향상
- 주관적 견해 반영으로 모델 우위 극대화
- 위험 균등 배분으로 드ロー다운 감소

---

### 3. Time Series CV 강화

#### 현재 상태
- 단순 Train/Test 분할
- 데이터 누출(leakage) 가능성

#### 개선 방향

```python
from validation.time_series_cv import TimeSeriesCV
from validation.walk_forward import WalkForwardAnalysis

class EnhancedTimeSeriesCV:
    """시계열 데이터를 위한 고도화된 검증"""

    def __init__(self, n_splits=5, gap=3, test_size=None):
        """
        n_splits: 분할 수
        gap: 학습과 테스트 사이의 gap (개월)
        test_size: 테스트 세트 크기 (None이면 자동)
        """
        self.n_splits = n_splits
        self.gap = gap
        self.test_size = test_size

    def split(self, X, y=None):
        """Forward-Looking Cross-Validation Split"""
        n_samples = len(X)
        test_size = self.test_size or n_samples // (self.n_splits + 1)

        for fold in range(self.n_splits):
            # 학습 기간
            train_end = fold * (test_size + self.gap)
            # Gap 기간 (데이터 누출 방지)
            gap_start = train_end
            gap_end = train_end + self.gap
            # 테스트 기간
            test_start = gap_end
            test_end = test_start + test_size

            if test_end > n_samples:
                break

            train_idx = np.arange(train_end)
            test_idx = np.arange(test_start, test_end)

            yield train_idx, test_idx

# Walk-Forward Analysis (점진적 재학습)
class WalkForwardBacktester:
    def __init__(self,
                 train_period=12,  # 12개월 학습
                 test_period=3,    # 3개월 테스트
                 step=1):           # 1개월씩 이동
        self.train_period = train_period
        self.test_period = test_period
        self.step = step

    def run(self, X, y, model_class, **model_params):
        """Walk-Forward 백테스트"""
        results = []

        n_periods = len(X) // self.step
        for i in range(n_periods - self.train_period - self.test_period):
            # 학습 기간
            train_start = i * self.step
            train_end = train_start + self.train_period * 21  # 21 거래일/월

            # 테스트 기간
            test_start = train_end
            test_end = test_start + self.test_period * 21

            if test_end > len(X):
                break

            # 모델 학습
            X_train = X[train_start:train_end]
            y_train = y[train_start:train_end]

            model = model_class(**model_params)
            model.fit(X_train, y_train)

            # 테스트
            X_test = X[test_start:test_end]
            y_test = y[test_start:test_end]

            y_pred = model.predict(X_test)

            # 성능 계산
            accuracy = (y_pred == y_test).mean()

            results.append({
                'period': i,
                'train_start': train_start,
                'test_start': test_start,
                'test_end': test_end,
                'accuracy': accuracy
            })

        return pd.DataFrame(results)

# 사용 예시
cv = EnhancedTimeSeriesCV(n_splits=5, gap=3)

for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = XGBClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    fold_score = accuracy_score(y_test, y_pred)

    print(f"Fold {fold}: {fold_score:.4f}")

# Walk-Forward Analysis
wfa = WalkForwardBacktester(
    train_period=12,
    test_period=3,
    step=1
)

wfa_results = wfa.run(X.values, y.values, XGBClassifier, n_estimators=100)
print(f"Average Accuracy: {wfa_results['accuracy'].mean():.4f}")
print(f"Accuracy Std: {wfa_results['accuracy'].std():.4f}")
```

**기대 효과**:
- 시간 순서 보존으로 현실적 성능 평가
- Gap 적용으로 데이터 누출 방지
- Walk-Forward로 모델 안정성 검증

---

### 4. Explainability (설명 가능성)

#### SHAP 도입

```python
import shap
import matplotlib.pyplot as plt

class ModelExplainer:
    """모델 예측 설명 가능성 분석"""

    def __init__(self, model, X_train):
        self.model = model
        self.X_train = X_train

        # 모델 타입에 따른 Explainer 선택
        if hasattr(model, 'predict') and hasattr(model, 'feature_importances_'):
            self.explainer = shap.TreeExplainer(model)
        else:
            # Kernel SHAP (더 느리지만 모든 모델에 적용 가능)
            self.explainer = shap.KernelExplainer(
                model.predict,
                shap.sample(X_train, 100)  # 배경 샘플
            )

    def explain_prediction(self, X_sample, stock_symbol=None):
        """개별 예측 설명"""
        shap_values = self.explainer.shap_values(X_sample)

        # Waterfall plot: 개별 예측 분해
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=self.explainer.expected_value,
                data=X_sample[0],
                feature_names=X_sample.columns
            ),
            show=False
        )
        plt.title(f"Prediction Breakdown: {stock_symbol}")
        plt.tight_layout()

        return shap_values

    def feature_importance(self, X_test):
        """Feature 중요도 분석"""
        shap_values = self.explainer.shap_values(X_test)

        # Summary plot
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            X_test,
            plot_type='bar',
            show=False
        )
        plt.title("Feature Importance (Mean |SHAP|)")
        plt.tight_layout()

        return shap_values

    def dependence_plot(self, X_test, feature_name):
        """Feature와 예측의 관계"""
        shap_values = self.explainer.shap_values(X_test)

        feature_idx = X_test.columns.get_loc(feature_name)

        shap.dependence_plot(
            feature_idx,
            shap_values,
            X_test,
            feature_names=X_test.columns
        )
        plt.title(f"Dependence Plot: {feature_name}")
        plt.tight_layout()

    def get_decision_rules(self, predictions, threshold=0.5):
        """Decision Rules 추출"""
        rules = []

        high_confidence = predictions[predictions > threshold]
        for idx in high_confidence.index:
            X_sample = X_test.loc[[idx]]
            shap_values = self.explainer.shap_values(X_sample)

            # 상위 5개 특성
            top_features = np.argsort(np.abs(shap_values[0]))[-5:]

            rule = {
                'stock_symbol': idx,
                'prediction': predictions[idx],
                'key_factors': [
                    {
                        'feature': X_test.columns[f_idx],
                        'value': X_sample.iloc[0, f_idx],
                        'shap_value': shap_values[0][f_idx]
                    }
                    for f_idx in top_features
                ]
            }
            rules.append(rule)

        return rules

# 사용 예시
explainer = ModelExplainer(best_model, X_train)

# 1. 특정 종목의 예측 설명
sample_stock = X_test.iloc[[0]]
sample_symbol = "AAPL"
shap_values = explainer.explain_prediction(sample_stock, sample_symbol)

print(f"\n{sample_symbol} 예측 설명:")
print("상위 영향 요인 (SHAP 값):")
top_features = np.argsort(np.abs(shap_values[0]))[-5:]
for f_idx in top_features:
    feature = X_test.columns[f_idx]
    value = sample_stock.iloc[0, f_idx]
    shap_val = shap_values[0][f_idx]
    print(f"  {feature}: {value:.4f} (SHAP: {shap_val:+.4f})")

# 2. 전체 Feature 중요도
explainer.feature_importance(X_test)

# 3. 특정 Feature의 영향도
explainer.dependence_plot(X_test, "pe_ratio")

# 4. Decision Rules 추출
decision_rules = explainer.get_decision_rules(y_pred, threshold=0.6)
for rule in decision_rules[:3]:
    print(f"\n{rule['stock_symbol']}: {rule['prediction']:.2%}")
    for factor in rule['key_factors']:
        print(f"  {factor['feature']}: {factor['shap_value']:+.4f}")
```

**기대 효과**:
- 투자자 신뢰도 향상 ("왜 이 종목을 추천했나?")
- 규제 대응 (설명 가능한 AI)
- 모델 디버깅 (이상 예측 원인 파악)

---

### 5. 대체 데이터 통합

#### 뉴스 감성 분석, 소셜 미디어, 거래량 이상 탐지 등

```python
from alternative_data.news_sentiment import NewsSentiment
from alternative_data.social_sentiment import SocialSentiment
from alternative_data.volume_analysis import VolumeAnalysis

class AlternativeDataIntegration:
    """대체 데이터 통합"""

    def __init__(self, news_api_key, twitter_api_key):
        self.news_sentiment = NewsSentiment(api_key=news_api_key)
        self.social_sentiment = SocialSentiment(api_key=twitter_api_key)
        self.volume_analysis = VolumeAnalysis()

    def get_news_sentiment(self, symbols, start_date, end_date):
        """뉴스 감성 분석 (FinBERT)"""
        news_data = self.news_sentiment.get_sentiment(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            language='ko'  # 한국어
        )

        return news_data

    def get_social_sentiment(self, symbols, lookback_days=30):
        """소셜 미디어 감성 분석"""
        # Twitter, Reddit (WallStreetBets 포함) 감성
        social_data = self.social_sentiment.get_sentiment(
            symbols=symbols,
            sources=['twitter', 'reddit'],
            lookback_days=lookback_days
        )

        return social_data

    def detect_unusual_volume(self, price_volume_data, zscore_threshold=2.0):
        """거래량 이상 탐지"""
        unusual_volumes = self.volume_analysis.detect_anomalies(
            price_volume_data,
            method='zscore',
            threshold=zscore_threshold,
            lookback_period=30
        )

        return unusual_volumes

    def get_option_signals(self, symbols, market_type='stock'):
        """옵션 데이터 기반 신호"""
        # Put/Call Ratio, IV Skew 등
        option_signals = {}

        for symbol in symbols:
            put_call_ratio = self._calculate_put_call_ratio(symbol)
            iv_skew = self._calculate_iv_skew(symbol)

            option_signals[symbol] = {
                'put_call_ratio': put_call_ratio,
                'iv_skew': iv_skew,
                'signal': self._interpret_option_signal(put_call_ratio, iv_skew)
            }

        return option_signals

    def integrate_features(self, X_base, symbols, dates):
        """기본 특성에 대체 데이터 통합"""
        X_enhanced = X_base.copy()

        # 뉴스 감성
        news_sentiment = self.get_news_sentiment(symbols, dates[0], dates[-1])
        X_enhanced['news_sentiment'] = news_sentiment
        X_enhanced['news_sentiment_ma7'] = news_sentiment.rolling(7).mean()

        # 소셜 미디어 감성
        social_sentiment = self.get_social_sentiment(symbols)
        X_enhanced['social_sentiment'] = social_sentiment

        # 거래량 이상
        volume_anomalies = self.detect_unusual_volume(price_volume_data)
        X_enhanced['volume_anomaly'] = volume_anomalies

        # 옵션 신호
        option_signals_dict = self.get_option_signals(symbols)
        X_enhanced['put_call_ratio'] = [
            option_signals_dict[sym]['put_call_ratio'] for sym in symbols
        ]
        X_enhanced['iv_skew'] = [
            option_signals_dict[sym]['iv_skew'] for sym in symbols
        ]

        return X_enhanced

# 사용 예시
alt_data = AlternativeDataIntegration(
    news_api_key='your_news_api_key',
    twitter_api_key='your_twitter_api_key'
)

# 기본 특성에 대체 데이터 통합
X_enhanced = alt_data.integrate_features(
    X_base=X_train,
    symbols=symbols,
    dates=dates
)

# 증강된 데이터로 모델 학습
model_enhanced = XGBClassifier(n_estimators=150)
model_enhanced.fit(X_enhanced, y_train)

# Feature 중요도 확인
feature_importance = pd.Series(
    model_enhanced.feature_importances_,
    index=X_enhanced.columns
).sort_values(ascending=False)

print("Top 10 Features:")
print(feature_importance.head(10))

# 뉴스 감성 기반 거래 신호
news_sentiment = alt_data.get_news_sentiment(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    start_date='2024-01-01',
    end_date='2024-11-17'
)

print("\nNews Sentiment by Symbol:")
print(news_sentiment.groupby('symbol')['sentiment'].mean())
```

**기대 효과**:
- 뉴스/소셜 감성으로 예측력 향상 (5-10%)
- 거래량 이상 탐지로 고수익 기회 포착
- 옵션 데이터로 시장 참여자의 의도 파악

---

### 6. Regime Detection (시장 국면 감지)

#### 불장/약장 자동 감지 및 동적 전략 조정

```python
from market_analysis.regime_detector import RegimeDetector
from hmmlearn import hmm
import numpy as np

class MarketRegimeDetector:
    """시장 국면 감지 및 동적 전략 조정"""

    def __init__(self, n_regimes=3, method='hmm'):
        """
        n_regimes: 레짐 개수 (Bull, Neutral, Bear)
        method: 'hmm' (Hidden Markov Model) 또는 'gmm' (Gaussian Mixture Model)
        """
        self.n_regimes = n_regimes
        self.method = method

        if method == 'hmm':
            self.model = hmm.GaussianHMM(
                n_components=n_regimes,
                covariance_type='diag',
                n_iter=1000,
                random_state=42
            )
        else:
            from sklearn.mixture import GaussianMixture
            self.model = GaussianMixture(
                n_components=n_regimes,
                random_state=42
            )

    def fit(self, price_history, features=['returns', 'volatility']):
        """가격 데이터로 레짐 모델 학습"""
        # Feature 계산
        returns = np.log(price_history.pct_change() + 1)
        volatility = returns.rolling(20).std()

        X = np.column_stack([returns, volatility]).dropna()

        self.model.fit(X)

        # 레짐 레이블 지정
        self.regime_labels = ['Bull', 'Neutral', 'Bear']

        return self

    def predict(self, price_history):
        """현재 레짐 예측"""
        returns = np.log(price_history.pct_change() + 1)
        volatility = returns.rolling(20).std()

        X = np.column_stack([returns, volatility]).dropna()
        regimes = self.model.predict(X)

        return regimes

    def get_current_regime(self, price_history):
        """현재 레짐 조회"""
        regimes = self.predict(price_history)
        current_regime_idx = regimes[-1]
        current_regime = self.regime_labels[current_regime_idx]

        return current_regime

    def get_regime_transition_probabilities(self):
        """레짐 전이 확률 (HMM)"""
        if self.method == 'hmm':
            transition_matrix = self.model.transmat_

            transition_df = pd.DataFrame(
                transition_matrix,
                index=self.regime_labels,
                columns=self.regime_labels
            )

            return transition_df
        else:
            return None

# 레짐별 전략 파라미터
REGIME_STRATEGIES = {
    'bull': {
        'model': 'aggressive_ensemble',
        'position_multiplier': 1.5,
        'stop_loss': -0.20,
        'take_profit': 0.40,
        'max_holding_period': 120,  # days
        'allocation_strategy': 'concentration',  # 집중 투자
    },
    'neutral': {
        'model': 'balanced_ensemble',
        'position_multiplier': 1.0,
        'stop_loss': -0.15,
        'take_profit': 0.25,
        'max_holding_period': 60,
        'allocation_strategy': 'balanced',
    },
    'bear': {
        'model': 'defensive_ensemble',
        'position_multiplier': 0.5,
        'stop_loss': -0.08,
        'take_profit': 0.12,
        'max_holding_period': 30,
        'allocation_strategy': 'diversified',  # 분산 투자
    }
}

class RegimeAdaptiveTrader:
    """레짐 기반 적응형 트레이더"""

    def __init__(self, detector):
        self.detector = detector

    def get_trading_parameters(self, current_regime):
        """현재 레짐에 맞는 거래 파라미터"""
        return REGIME_STRATEGIES.get(
            current_regime.lower(),
            REGIME_STRATEGIES['neutral']
        )

    def select_model(self, current_regime):
        """레짐에 맞는 모델 선택"""
        params = self.get_trading_parameters(current_regime)
        model_name = params['model']

        if model_name == 'aggressive_ensemble':
            return self._load_aggressive_model()
        elif model_name == 'balanced_ensemble':
            return self._load_balanced_model()
        else:
            return self._load_defensive_model()

    def allocate_capital(self,
                        predictions,
                        prices,
                        current_regime,
                        portfolio_value):
        """레짐에 맞는 자본 배분"""
        params = self.get_trading_parameters(current_regime)
        multiplier = params['position_multiplier']

        # 기본 가중치
        base_weights = predictions / predictions.sum()

        # 레짐 승수 적용
        adjusted_weights = base_weights * multiplier

        # 정규화
        adjusted_weights = adjusted_weights / adjusted_weights.sum()

        # 포지션 크기
        positions = (portfolio_value * adjusted_weights / prices).astype(int)

        return positions

# 사용 예시
# 1. 레짐 탐지
detector = MarketRegimeDetector(n_regimes=3, method='hmm')
detector.fit(historical_prices)

# 2. 현재 레짐 확인
current_regime = detector.get_current_regime(historical_prices)
print(f"Current Market Regime: {current_regime}")

# 3. 레짐 전이 확률 확인
transition_probs = detector.get_regime_transition_probabilities()
print("\nRegime Transition Probabilities:")
print(transition_probs)

# 4. 레짐별 전략 적용
trader = RegimeAdaptiveTrader(detector)
strategy_params = trader.get_trading_parameters(current_regime)
print(f"\nTrading Parameters for {current_regime}:")
print(strategy_params)

# 5. 레짐별 모델 선택 및 예측
model = trader.select_model(current_regime)
predictions = model.predict(X_test)

# 6. 레짐별 자본 배분
positions = trader.allocate_capital(
    predictions=predictions,
    prices=current_prices,
    current_regime=current_regime,
    portfolio_value=1_000_000
)
```

**기대 효과**:
- 시장 상황에 맞는 유동적 전략 조정
- 약장에서 방어적 포지셍으로 손실 제한
- 불장에서 공격적 투자로 수익 극대화
- Sharpe Ratio 15-30% 향상

---

## 우선순위 제안

체계적인 개선을 위해 다음과 같이 우선순위를 제시합니다.
각 항목은 구현 난이도, 예상 효과, 소요 시간을 고려하여 배정되었습니다.

### 단기 계획 (1-2주)

#### 목표
기존 시스템의 성능을 빠르게 개선하고, 백테스트의 현실성을 높이기

| 우선순위 | 작업 | 예상 효과 | 난이도 | 소요 시간 |
|---------|------|----------|--------|---------|
| 1 | **Feature Selection** | 학습 속도 2배, 성능 2-3% | 낮음 | 3-4일 |
| 2 | **거래 비용 모델링** | 백테스트 현실화 | 낮음 | 2-3일 |
| 3 | **Stacking Ensemble** | 정확도 3-5% | 중간 | 3-4일 |

**실행 계획**:
```bash
# 1. Feature Selection 적용
python scripts/feature_selection.py --method importance --threshold 0.01

# 2. 거래 비용 반영
# backtest.py 수정 (위의 코드 참고)

# 3. Stacking Ensemble 실험
python scripts/train_stacking_ensemble.py --base_models xgb,lgb,rf --meta ridge
```

---

### 중기 계획 (1-2개월)

#### 목표
모델 성능을 고도화하고, Risk Management 시스템 구축

| 우선순위 | 작업 | 예상 효과 | 난이도 | 소요 시간 |
|---------|------|----------|--------|---------|
| 1 | **Optuna 하이퍼파라미터 최적화** | 성능 2-3%, 안정성 향상 | 중간 | 2주 |
| 2 | **Time Series CV 강화** | 오버피팅 방지, 신뢰도 향상 | 중간 | 1주 |
| 3 | **Risk Management 모듈** | 위험 제한, 안정성 확보 | 중간 | 1주 |
| 4 | **Data Quality 개선** | 모델 성능 3-5% | 중간 | 1.5주 |

**실행 계획**:
```bash
# 1. Optuna 최적화 (200 trials)
python scripts/optimize_hyperparameters.py --n_trials 200 --n_jobs -1

# 2. Time Series CV 적용
python training/train_with_ts_cv.py --n_splits 5 --gap 3

# 3. Risk Management 모듈 구축
python scripts/setup_risk_management.py

# 4. 데이터 검증 강화
python scripts/validate_and_impute_data.py --method interpolation
```

---

### 장기 계획 (3-6개월)

#### 목표
시스템의 고급 기능 구축 및 실시간 트레이딩 지원

| 우선순위 | 작업 | 예상 효과 | 난이도 | 소요 시간 |
|---------|------|----------|--------|---------|
| 1 | **Regime Detection** | Sharpe Ratio 15-30% 향상 | 높음 | 3-4주 |
| 2 | **대체 데이터 통합** | 예측력 5-10% 향상 | 높음 | 4-5주 |
| 3 | **Explainability (SHAP)** | 신뢰도 향상, 규제 대응 | 중간 | 2주 |
| 4 | **실시간 트레이딩** | 자동 매매, 고빈도 거래 | 높음 | 4-6주 |
| 5 | **Deep Learning 실험** | 성능 5-7% 향상 (조건부) | 높음 | 3-4주 |

**실행 계획**:
```bash
# 1. Regime Detection 구축
python scripts/setup_regime_detection.py --n_regimes 3 --method hmm

# 2. 대체 데이터 통합
python scripts/integrate_alternative_data.py --sources news,social,options

# 3. SHAP 기반 설명 가능성
python scripts/setup_explainability.py --method shap

# 4. 실시간 데이터 수집
python scripts/setup_realtime_collector.py --broker alpaca

# 5. LSTM 모델 실험
python scripts/train_lstm_model.py --lookback 60 --epochs 50
```

---

## 관련 문서 및 리소스

### 주요 문서 위치

| 문서 | 위치 | 목적 |
|------|------|------|
| **원본 WORKFLOW_GUIDE** | `/home/user/Quant/Quant-refactoring/docs/archive/WORKFLOW_GUIDE_ORIGINAL.md` | 전체 시스템 설명 |
| **시스템 아키텍처** | `/home/user/Quant/Quant-refactoring/docs/SYSTEM_ARCHITECTURE.md` | 컴포넌트 상세 설명 |
| **설치 및 실행 가이드** | `/home/user/Quant/Quant-refactoring/README.md` | 초기 설정 방법 |

### 핵심 모듈

| 모듈 | 경로 | 설명 |
|------|------|------|
| Feature Selection | `feature_engineering/feature_selector.py` | 중요 특성 선택 |
| Ensemble | `models/ensemble.py` | 앙상블 모델 |
| Backtester | `backtest.py` | 전략 백테스팅 |
| Risk Manager | `portfolio_optimization/risk_manager.py` | 위험 관리 |
| 데이터 검증 | `validation/data_validator.py` | 데이터 품질 보증 |

### 외부 리소스

**Python 라이브러리**:
- `Optuna`: 하이퍼파라미터 최적화
- `SHAP`: 모델 설명 가능성
- `Featuretools`: 자동 특성 공학
- `hmmlearn`: Hidden Markov Model
- `Alpaca API`: 자동 매매
- `Interactive Brokers API`: 실시간 거래

**학습 자료**:
- Markowitz 포트폴리오 이론
- Black-Litterman 모델
- Hidden Markov Model (시장 레짐 감지)
- SHAP 값 해석

---

## 📊 예상 성과

### 현재 시스템 성과 (Baseline)
- 분류 정확도: 55-60%
- Sharpe Ratio: 0.8-1.0
- 최대 드로우다운: 15-20%

### 개선 후 예상 성과 (12개월 후)

| 항목 | 현재 | 단기 후 (2주) | 중기 후 (2개월) | 장기 후 (6개월) | 개선율 |
|------|------|----------|----------|----------|--------|
| 분류 정확도 | 57% | 59% | 62% | 67% | +10-15% |
| Sharpe Ratio | 0.95 | 1.15 | 1.35 | 1.65 | +40-75% |
| 최대 드로우다운 | 18% | 16% | 12% | 8% | -55% |
| 연간 수익률 | 12% | 15% | 18% | 25%+ | +100%+ |

---

## 🎯 체크리스트

### 단기 (1-2주)
- [ ] Feature Selection 스크립트 작성 및 테스트
- [ ] 거래 비용 모델 backtest.py에 반영
- [ ] Stacking Ensemble 모델 구현 및 성능 평가
- [ ] 프로젝트 문서 업데이트

### 중기 (1-2개월)
- [ ] Optuna 기반 하이퍼파라미터 튜닝 완료
- [ ] Time Series CV 적용 및 검증
- [ ] Risk Management 모듈 구현 및 포트폴리오에 통합
- [ ] 데이터 품질 개선 및 검증 로직 강화

### 장기 (3-6개월)
- [ ] Regime Detection 모델 구축 및 전략 적용
- [ ] 대체 데이터 (뉴스, 소셜) 통합
- [ ] SHAP 기반 Explainability 구현
- [ ] 실시간 데이터 수집 및 자동 매매 시스템
- [ ] LSTM/Transformer 모델 실험

---

## 📞 지원 및 문의

개선 작업 중 문제가 발생하거나 추가 지원이 필요한 경우:

1. **GitHub Issues**: 버그 또는 기능 요청 등록
2. **문서 검토**: 관련 문서 다시 확인
3. **팀 협력**: 다른 팀원과 상의

---

**문서 버전**: 1.0
**마지막 업데이트**: 2025-11-17
**작성자**: Claude AI + Development Team
**검토자**: (담당자 이름)

---

**라이센스**: MIT License
**저작권**: Quant Trading Development Team
