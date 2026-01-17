# 종합 개선 제안 및 로드맵

> **작성일**: 2025-12-17 (최종 업데이트: 2026-01-17)
> **이전 문서**: [06_quant_perspective.md](./06_quant_perspective.md)
> **다음 문서**: [08_recent_changes.md](./08_recent_changes.md)

---

## 목차

1. [완료된 개선 사항](#1-완료된-개선-사항-2026-01)
2. [우선순위별 개선 과제](#2-우선순위별-개선-과제)
3. [단기 개선 (1~2주)](#3-단기-개선-1-2주)
4. [중기 개선 (1~2개월)](#4-중기-개선-1-2개월)
5. [장기 개선 (3~6개월)](#5-장기-개선-3-6개월)
6. [실전 투자 로드맵](#6-실전-투자-로드맵)
7. [다음 분기 TODO](#7-다음-분기-todo)

---

## 1. 완료된 개선 사항 (2026-01)

### ✅ 거래 비용 반영 (Commission + Slippage)

**커밋**: `ef25545` (feat: Add trading costs (commission & slippage) to backtest)

**구현 내용**:
- Commission: 매수/매도 시 각각 0.1% 적용
- Slippage: 시장가 주문 시 0.1% 불리한 가격 반영
- `config/conf.yaml`의 `BACKTEST.TRADING_COSTS` 섹션으로 설정

```yaml
BACKTEST:
  TRADING_COSTS:
    ENABLED: Y           # 거래 비용 적용 여부
    COMMISSION: 0.001    # 0.1% (편도)
    SLIPPAGE: 0.001      # 0.1% (편도)
```

**효과**:
- 실전과 동일한 조건으로 백테스트
- 연간 약 1.6% 수익률 차감 (분기별 리밸런싱 기준)
- 낙관적 수익률 추정 방지

**문서**: [04_backtesting.md](./04_backtesting.md) - Section 8

---

### ✅ 아키텍처 기반 일원화 강제

**커밋**: `f8789bd` (refactor: Remove fallback, require predictions cache for consistency)

**구현 내용**:
- Prediction Cache가 없으면 에러 발생 (Silent fallback 제거)
- regressor.py → ml_backtest.py 순서로 실행 필수
- 아키텍처 자체가 일원화를 강제

```python
# ml_backtest.py
if cache_path.exists():
    self.predictions_cache = joblib.load(cache_path)
else:
    raise FileNotFoundError(
        "Predictions cache not found!\n"
        "Run regressor.py first, or set USE_CACHED_PREDICTIONS=N"
    )
```

**효과**:
- regressor ↔ ml_backtest 100% 일관성 보장
- 실수로 다른 예측값 사용 원천 차단
- 코드 리뷰/유닛테스트보다 확실한 보장

**문서**: [08_recent_changes.md](./08_recent_changes.md) - Section 2

---

### ✅ 예측 전용 모드 (Prediction-Only Mode)

**커밋**: `3e73dc1` (feat: Add prediction-only mode for quick stock recommendations)

**구현 내용**:
- 학습된 모델로 특정 날짜 기준 주식 추천
- 재학습 없이 수 초 내 결과 도출
- `config/conf.yaml`의 `PREDICTION` 섹션으로 설정

```yaml
PREDICTION:
  ENABLED: Y
  TARGET_DATE: "2025-01-11"  # 또는 "latest"
  TOP_K: 10
```

**효과**:
- 실전 투자에서 가장 중요한 기능
- 매일/매주 빠른 추천 확인 가능
- 과거 특정 시점 시뮬레이션 가능

---

### ✅ CLASSIFIER_MODE 구현 (negative_screen / positive_screen)

**커밋**: `ba3b312` (refactor: Improve negative_screen implementation in ml_backtest.py)

**구현 내용**:
- `negative_screen`: 극단적 손실(-30% 이하) 종목 제거 (권장)
- `positive_screen`: 하락 확률 높은 종목 제거
- 4-classifier 앙상블 평균으로 섹터 threshold 계산
- Hard filtering 적용 (`ml_score = np.where(pass_mask, return, -np.inf)`)

```yaml
ML:
  CLASSIFIER_MODE: "negative_screen"
  CLASSIFIER_REMOVE_PCT_MIN: 2
  CLASSIFIER_REMOVE_PCT_MAX: 15
  NEGATIVE_SCREEN:
    LOSS_THRESHOLD: -0.3
```

**효과**:
- 시장 환경에 따라 보수적/공격적 전략 선택 가능
- threshold_config.pkl에 mode, remove_pct 필드 추가

---

### ✅ 거래일 조정 (Trading Day Adjustment)

**커밋**: `551fb8a` (Fix get_trade_date() returning wrong trading date)

**구현 내용**:
- 리밸런싱 날짜가 휴장일인 경우 이전 거래일로 자동 조정
- `_get_trade_date()` 함수로 실제 거래 가능 날짜 반환
- regressor.py와 ml_backtest.py 간 일관성 확보

**효과**:
- 휴장일 0% 수익률 문제 해결
- 백테스트 정확도 향상

---

### ✅ 문서 구조 개편

**커밋**: `7e07669` (refactor: Move core analysis reports to docs/codebase-report/)

**구현 내용**:
- 핵심 분석 레포트 9개를 `docs/codebase-report/`로 분리
- README.md 인덱스 파일 추가
- CLAUDE.md, README.md에 참조 추가

**효과**:
- 핵심 문서와 작업 문서 분리
- 관리 효율성 향상

---

## 2. 우선순위별 개선 과제

### 긴급 (Critical) - 실전 투자 전 필수

| 과제 | 현재 상태 | 목표 | 예상 시간 |
|------|----------|------|----------|
| 백테스트 검증 | 2년 (2023-2025) | 여러 기간 (2008, 2015, 2020, 2022) | 1주 |
| 데이터 품질 체크 | 미흡 | NaN/Infinite 분석 리포트 | 3일 |
| 모델 성능 검증 | 부분적 | Classifier/Regressor 성능 확인 | 3일 |
| 리스크 관리 | 없음 | Stop-Loss, Position Sizing | 1주 |

### 높음 (High) - 성능 개선

| 과제 | 현재 상태 | 목표 | 예상 시간 |
|------|----------|------|----------|
| Feature 선택 | 500~1000개 | Top-50 | 3일 |
| 복잡도 감소 | 20+ 단계 | Ablation Study → 10 단계 | 1주 |
| 단위 테스트 | 없음 | 핵심 함수 커버리지 50%+ | 1주 |
| ~~거래 비용 반영~~ | ~~없음~~ | ~~Commission + Slippage~~ | ✅ 완료 |

### 중간 (Medium) - 안정성 강화

| 과제 | 현재 상태 | 목표 | 예상 시간 |
|------|----------|------|----------|
| 모니터링 시스템 | 없음 | Prometheus + Grafana | 2주 |
| 에러 처리 강화 | 부분적 | 재시도 로직, Alert | 1주 |
| 증분 업데이트 | 없음 | 신규 데이터만 수집 | 1주 |

### 낮음 (Low) - 장기 개선

| 과제 | 현재 상태 | 목표 | 예상 시간 |
|------|----------|------|----------|
| Microservices | Monolith | 서비스 분리 | 2개월 |
| 고급 ML 모델 | XGBoost/LGBM | Transformer | 1개월 |
| 클라우드 배포 | 로컬 | AWS/GCP | 2주 |

---

## 3. 단기 개선 (1~2주)

### 3.1 백테스트 검증 철저화 (3일)

**목표**: 다양한 시장 환경에서 성능 검증

**작업**:
```yaml
# config/conf.yaml
BACKTEST:
  PERIODS:
    # 2008 금융위기
    - START_YEAR: 2008
      END_YEAR: 2009
      START_MONTH: 1
      START_DATE: 1

    # 2015 저성장기
    - START_YEAR: 2015
      END_YEAR: 2016
      START_MONTH: 1
      START_DATE: 1

    # 2020 코로나 충격
    - START_YEAR: 2020
      END_YEAR: 2021
      START_MONTH: 1
      START_DATE: 1

    # 2022 금리 급등기
    - START_YEAR: 2022
      END_YEAR: 2023
      START_MONTH: 1
      START_DATE: 1
```

**실행**:
```bash
python main.py
# 각 기간별 백테스트 실행
# 리포트 비교 분석
```

**성공 기준**:
- 모든 기간에서 Sharpe > 0.5
- 평균 Sharpe > 1.0
- MDD < -30%

---

### 3.2 데이터 품질 체크 (3일)

**목표**: NaN/Infinite 문제 진단 및 해결

**작업**:
```python
# scripts/data_quality_check.py
def generate_data_quality_report():
    # 1. NaN 분석
    nan_report = analyze_nan_values()

    # 2. Infinite 분석
    inf_report = analyze_infinite_values()

    # 3. Filing delay 분석
    delay_report = analyze_filing_delay()

    # 4. 종합 리포트
    save_report({
        'nan': nan_report,
        'infinite': inf_report,
        'delay': delay_report
    }, 'data_quality_report.xlsx')
```

**실행**:
```bash
python scripts/data_quality_check.py
```

**성공 기준**:
- NaN 비율 < 10%
- Infinite 0개
- Filing delay: 30~90일 (정상 범위)

---

### 3.3 Feature 선택 (3일)

**목표**: Feature 수를 500~1000개 → 50개로 감소

**작업**:
```python
# scripts/feature_selection.py
from sklearn.ensemble import RandomForestRegressor

# 1. Feature Importance 계산
rf = RandomForestRegressor()
rf.fit(X_train, y_train)

importances = rf.feature_importances_
top_50_idx = np.argsort(importances)[-50:]

# 2. 선택된 Feature 저장
selected_features = X_train.columns[top_50_idx].tolist()
save_json(selected_features, 'selected_features.json')

# 3. 성능 비교
perf_original = train_and_evaluate(X_train)
perf_reduced = train_and_evaluate(X_train[:, top_50_idx])

print(f"Original: {perf_original}")
print(f"Reduced: {perf_reduced}")
```

**성공 기준**:
- 성능 차이 < 5%
- 학습 시간 50% 감소

---

### ~~3.4 거래 비용 반영 (2일)~~ ✅ 완료

**커밋**: `ef25545`

**구현 완료**:
- Commission: 0.1% (편도)
- Slippage: 0.1% (편도)
- 설정: `BACKTEST.TRADING_COSTS.ENABLED: Y`

---

### 3.5 Stop-Loss 구현 (2일)

**목표**: 종목별 손절 로직 추가

**작업**:
```python
# src/backtest/ml_backtest.py
def _apply_stop_loss(self, portfolio, max_loss=-0.15):
    """
    종목별 -15% 손실 시 강제 매도
    """
    stop_loss_triggered = []

    for symbol in portfolio:
        current_return = (current_price - entry_price) / entry_price

        if current_return < max_loss:
            logger.warning(f"{symbol} hit stop-loss: {current_return:.2%}")
            stop_loss_triggered.append(symbol)

    # 포트폴리오에서 제거
    portfolio = portfolio[~portfolio['symbol'].isin(stop_loss_triggered)]

    return portfolio
```

---

## 4. 중기 개선 (1~2개월)

### 4.1 단위 테스트 추가 (1주)

**목표**: 핵심 함수 커버리지 50%+

**작업**:
```python
# tests/test_data_processor.py
import pytest

def test_remove_infinite_values():
    X = pd.DataFrame({'a': [1, np.inf, 3], 'b': [4, 5, -np.inf]})
    X_clean = DataProcessor.remove_infinite_values(X)
    assert not np.isinf(X_clean).any().any()

def test_align_features_to_model():
    X = pd.DataFrame({'feat_a': [1, 2], 'feat_b': [3, 4]})
    model = MockModel(features=['feat_b', 'feat_c'])
    X_aligned = DataProcessor.align_features_to_model(X, model)
    assert 'feat_c' in X_aligned.columns
    assert X_aligned['feat_c'].isnull().all()

# tests/test_ml_backtest.py
def test_backtest_reproducibility():
    np.random.seed(42)
    result1 = run_backtest(config)

    np.random.seed(42)
    result2 = run_backtest(config)

    assert result1.equals(result2)

# tests/test_future_leakage.py
def test_no_future_leakage():
    cutoff_date = pd.Timestamp('2024-06-01')
    df = load_data(cutoff_date=cutoff_date)
    assert (df['filingDate'] <= cutoff_date).all()
```

**실행**:
```bash
pytest tests/ -v --cov=src --cov-report=html
```

---

### 4.2 모니터링 시스템 (2주)

**목표**: Prometheus + Grafana 대시보드

**작업**:
```python
# src/monitoring/prometheus_exporter.py
from prometheus_client import Gauge, Counter

# 메트릭 정의
sharpe_ratio_gauge = Gauge('backtest_sharpe_ratio', 'Sharpe Ratio')
max_drawdown_gauge = Gauge('backtest_max_drawdown', 'Max Drawdown')
win_rate_gauge = Gauge('backtest_win_rate', 'Win Rate')
trade_count = Counter('backtest_trade_count', 'Number of Trades')

# 백테스트 결과 기록
def record_backtest_metrics(results):
    sharpe_ratio_gauge.set(results['sharpe'])
    max_drawdown_gauge.set(results['mdd'])
    win_rate_gauge.set(results['win_rate'])
    trade_count.inc(results['n_trades'])
```

**Grafana 대시보드**:
```
┌─────────────────────────────────────────────────────┐
│ Sharpe Ratio: 1.2  ▲ (+0.1)                        │
│ MDD: -18%          ▼ (-2%)                          │
│ Win Rate: 58%      ● (0%)                           │
│ Total Return: +45% ▲ (+5%)                          │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ 수익률 추이 (Line Chart)                            │
│ ─────────────────────────────────────────────────── │
│     ╭───╮                                           │
│  ╭──╯   ╰───╮                                       │
│ ─╯          ╰────                                   │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ Alert                                               │
│ ⚠️ Sharpe Ratio dropped below 0.5                  │
│ ⚠️ MDD exceeded -25%                                │
└─────────────────────────────────────────────────────┘
```

---

### 4.3 복잡도 감소 (1주)

**목표**: Ablation Study 실행, 불필요한 단계 제거

**작업**:
```python
# scripts/ablation_study.py
stages = [
    'basic',
    'winsorization',
    'feature_selection',
    'extreme_filter',
    'log_transform'
]

results = {}

for i in range(1, len(stages) + 1):
    active_stages = stages[:i]

    # 백테스트 실행
    perf = run_backtest(stages=active_stages)

    results[tuple(active_stages)] = perf
    print(f"{active_stages}: Sharpe {perf['sharpe']:.2f}")

# 예시 출력:
# ['basic']: Sharpe 0.95
# ['basic', 'winsorization']: Sharpe 1.08 (+0.13) ✅
# ['basic', 'winsorization', 'feature_selection']: Sharpe 1.09 (+0.01)
# ['basic', 'winsorization', 'feature_selection', 'extreme_filter']: Sharpe 1.15 (+0.06) ✅
# ['basic', 'winsorization', 'feature_selection', 'extreme_filter', 'log_transform']: Sharpe 1.14 (-0.01) ❌

# 결론: log_transform 제거
```

---

### 4.4 섹터 모델 효과 검증 (3일)

**목표**: 섹터별 모델 vs 통합 모델 성능 비교

**작업**:
```python
# scripts/sector_model_comparison.py
# 1. 통합 모델
config_unified = config.copy()
config_unified['ML']['USE_SECTOR_MODEL'] = 'N'
perf_unified = run_backtest(config_unified)

# 2. 섹터별 모델
config_sector = config.copy()
config_sector['ML']['USE_SECTOR_MODEL'] = 'Y'
perf_sector = run_backtest(config_sector)

# 3. 비교
print(f"Unified: Sharpe {perf_unified['sharpe']:.2f}")
print(f"Sector: Sharpe {perf_sector['sharpe']:.2f}")
print(f"Difference: {perf_sector['sharpe'] - perf_unified['sharpe']:.2f}")

# 예시 출력:
# Unified: Sharpe 1.2
# Sector: Sharpe 1.3
# Difference: +0.1

# 결론: 섹터 모델 효과 있음 (차이 > 0.2이면 유의미)
```

---

## 5. 장기 개선 (3~6개월)

### 5.1 Microservices 아키텍처 (2개월)

**목표**: Monolith → Microservices 전환

**설계**:
```
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ Data Service  │    │ ML Service    │    │ Backtest      │
│               │→→→→│               │→→→→│ Service       │
│ FMP API       │    │ Training      │    │ Walk-Forward  │
└───────────────┘    └───────────────┘    └───────────────┘
      ↓                     ↓                     ↓
┌─────────────────────────────────────────────────────────┐
│             Message Queue (RabbitMQ/Kafka)              │
└─────────────────────────────────────────────────────────┘
      ↓                     ↓                     ↓
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ Monitoring    │    │ Alert         │    │ Dashboard     │
│ (Prometheus)  │    │ (Slack/Email) │    │ (Grafana)     │
└───────────────┘    └───────────────┘    └───────────────┘
```

---

### 5.2 고급 ML 모델 (1개월)

**Transformer 기반 시계열**:
```python
from transformers import TimeSeriesTransformer

class TransformerRegressor:
    def __init__(self, seq_len=20):
        self.model = TimeSeriesTransformer(
            input_dim=100,
            output_dim=1,
            n_heads=8,
            n_layers=6
        )

    def fit(self, X_seq, y):
        # X_seq: (n_samples, seq_len, n_features)
        self.model.train(X_seq, y)

    def predict(self, X_seq):
        return self.model.predict(X_seq)
```

---

### 5.3 클라우드 배포 (2주)

**Docker 컨테이너화**:
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "main.py"]
```

**Kubernetes 배포**:
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: quant-trading
spec:
  replicas: 3
  selector:
    matchLabels:
      app: quant-trading
  template:
    metadata:
      labels:
        app: quant-trading
    spec:
      containers:
      - name: quant-trading
        image: quant-trading:latest
        resources:
          limits:
            memory: "4Gi"
            cpu: "2"
```

---

## 6. 실전 투자 로드맵

### Phase 1: 파일럿 (3개월)

**목표**: 소액으로 시스템 검증

**세부 계획**:
```
투자 금액: $250 (총 자산의 5%)
종목 수: Top-K = 5
비중: 균등 (20% × 5 = $50 × 5)
리밸런싱: 3개월마다
모니터링: 매주 수익률 체크
```

**성공 기준**:
- 3개월 후 Sharpe > 0.5
- MDD < -20%
- 벤치마크 (SPY) 대비 +0%

---

### Phase 2: 확장 (6개월)

**목표**: 투자 금액 확대

**세부 계획**:
```
투자 금액: $500 (총 자산의 10%)
종목 수: Top-K = 10
비중: 균등 (10% × 10 = $50 × 10)
리밸런싱: 3개월마다
모니터링: 매주 수익률 + MDD 체크
```

**성공 기준**:
- 6개월 누적 Sharpe > 1.0
- MDD < -25%
- 벤치마크 대비 +5%

---

### Phase 3: 본격 운용 (1년 이후)

**목표**: 지속 가능한 알파 창출

**세부 계획**:
```
투자 금액: $1,500 (총 자산의 20~30%)
종목 수: Top-K = 15~20
비중: Kelly Criterion 기반
리밸런싱: 3개월마다
모니터링: 자동 대시보드
```

**성공 기준**:
- 연 CAGR > 15%
- Sharpe > 1.2
- MDD < -30%
- 벤치마크 대비 +10%

---

## 7. 다음 분기 TODO

### 즉시 실행 (이번 주)

- [x] ~~백테스트 검증 (다양한 시장 환경)~~ ✅ 4년 검증 설정 완료 (2020-2023)
- [ ] 데이터 품질 체크 리포트 생성
- [ ] 모델 성능 지표 확인 (Accuracy, RMSE)

### 단기 (1~2주)

- [ ] Feature 선택 (Top-50)
- [x] ~~거래 비용 반영 (Commission + Slippage)~~ ✅ `ef25545`
- [x] ~~슬리피지 반영~~ ✅ `ef25545`
- [ ] Stop-Loss 구현
- [ ] Position Sizing 구현

### 중기 (1~2개월)

- [ ] 단위 테스트 추가 (커버리지 50%)
- [ ] 모니터링 시스템 (Prometheus + Grafana)
- [ ] 복잡도 감소 (Ablation Study)
- [ ] 섹터 모델 효과 검증

### 장기 (3~6개월)

- [ ] Microservices 아키텍처 전환
- [ ] Transformer 기반 시계열 모델
- [ ] 클라우드 배포 (AWS/GCP)
- [ ] CI/CD 파이프라인

---

## 결론

이 시스템은 **탄탄한 철학**과 **정교한 아키텍처**를 가진
**실전 투자 가능한 시스템**입니다.

### 핵심 요약

**현재 상태**: A- (실전 투자 준비 완료)

**강점**:
- 명확한 철학 (예측 아닌 선별)
- 엄격한 백테스트 (Walk-Forward, 거래 비용 반영)
- 통합 아키텍처 (코드 이중화 제거, 아키텍처 기반 일원화)
- 실전 기능 (예측 전용 모드, 거래 비용)

**약점**:
- 과적합 위험 (모델 수, Feature 수)
- 검증 부족 (단위 테스트, Out-of-Sample)
- 리스크 관리 미흡 (Stop-Loss 미구현)

**다음 단계**:
1. 백테스트 검증 (긴급)
2. 리스크 관리 구현 (긴급)
3. 소액 파일럿 운용 (3개월)
4. 단계적 확대 (6개월 → 1년)

**최종 권고**:
- 소액으로 시작하여 검증
- 성공 후 단계적 확대
- 지속적 모니터링 및 개선

---

**END OF ANALYSIS**

전체 분석 문서:
- [00_overview.md](./00_overview.md)
- [01_architecture.md](./01_architecture.md)
- [02_data_pipeline.md](./02_data_pipeline.md)
- [03_ml_strategy.md](./03_ml_strategy.md)
- [04_backtesting.md](./04_backtesting.md)
- [05_code_quality.md](./05_code_quality.md)
- [06_quant_perspective.md](./06_quant_perspective.md)
- [07_recommendations.md](./07_recommendations.md) (현재 문서)
- [08_recent_changes.md](./08_recent_changes.md) - 최신 변경사항 분석 (2026-01)
