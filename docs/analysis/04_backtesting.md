# 백테스팅 시스템 분석

> **작성일**: 2025-12-17 (최종 업데이트: 2026-01-10)
> **이전 문서**: [03_ml_strategy.md](./03_ml_strategy.md)
> **다음 문서**: [05_code_quality.md](./05_code_quality.md)

---

## 핵심 요약

### 백테스팅 평가: A- (엄격하고 신뢰성 높음)

**강점**:
- Walk-Forward Analysis (미래 유출 방지)
- Filing Date 기준 엄격한 cutoff
- 벤치마크 비교 (AAPL, NVDA, QQQ, SPY)
- Excel 통합 리포트

**약점**:
- Out-of-Sample 기간 짧음 (2년)
- 거래 비용 미반영
- 슬리피지 미반영

---

## 1. Walk-Forward Analysis

### 전통적 백테스트 (X)

```python
# ❌ 나쁜 예: 전체 데이터로 학습 → 전체 기간 테스트
model.fit(X_all, y_all)  # 1996~2025 전체
predictions = model.predict(X_all)
backtest(predictions)  # 미래 유출!
```

**문제**:
- 미래 정보 유출
- 과적합
- 실전과 괴리

### Walk-Forward (O)

```python
# ✅ 좋은 예: 시점별로 그 당시 알 수 있었던 정보만 사용
for date in rebalance_dates:
    # 1. 미래 유출 방지 데이터 로드
    X_available = load_data(cutoff_date=date)  # Filing Date <= date
    
    # 2. 모델 학습 (필요 시)
    if should_retrain(date):
        model.fit(X_available, y_available)
    
    # 3. 예측
    predictions = model.predict(X_available)
    
    # 4. Top-K 선택
    top_k = select_top_k(predictions)
    
    # 5. 실제 수익률 계산
    returns = calculate_returns(top_k, date)
    
    results.append(returns)
```

**효과**:
- 미래 유출 방지 ✅
- 실전과 동일한 환경
- 신뢰성 높은 백테스트

---

## 2. Retrain 전략

### 4가지 옵션

| 전략 | 설명 | 장점 | 단점 |
|------|------|------|------|
| **every** | 매 리밸런싱마다 재학습 | 최신 데이터 반영 | 계산 비용 높음 |
| **quarterly** | 분기마다 재학습 | 균형적 | 모델 drift 가능 |
| **yearly** | 연마다 재학습 | 계산 효율적 | 오래된 모델 |
| **once** | 최초 1회만 학습 | 계산 최소 | 시장 변화 미반영 |

**권장**: quarterly (기본값)

### Window 전략

#### Expanding Window (권장)

```python
# 시작~현재 (점점 증가)
train_data = data[start_date:current_date]
# 예: 2020 (5년) → 2021 (6년) → 2022 (7년)
```

**장점**:
- 데이터 축적 → 성능 향상
- 시장 사이클 포함

**단점**:
- 오래된 데이터 → Regime shift 대응 어려움

#### Rolling Window

```python
# 고정 크기 윈도우
train_data = data[current_date-window_size:current_date]
# 예: 2020~2022 (3년) → 2021~2023 (3년)
```

**장점**:
- 최근 데이터만 → Regime shift 대응

**단점**:
- 데이터 부족 (초기)
- 사이클 미포함

---

## 3. 벤치마크 비교

### 지원 벤치마크

```yaml
BENCHMARK:
  ENABLED: Y
  SYMBOLS:
    - AAPL  # 대형주 대표
    - NVDA  # 성장주 대표
    - QQQ   # Nasdaq-100 (기술주)
    - SPY   # S&P 500 (시장 전체)
```

### 비교 메트릭

```python
metrics = {
    'Total Return': (final_value - initial_value) / initial_value,
    'CAGR': (final_value / initial_value) ** (1/years) - 1,
    'Sharpe Ratio': mean(returns) / std(returns) * sqrt(252),
    'Max Drawdown': max(peak - trough) / peak,
    'Win Rate': sum(returns > 0) / len(returns),
    'Volatility': std(returns) * sqrt(252)
}
```

### 해석

```
예시 결과:
                 Return  MDD    Sharpe  Win Rate
ML Model         +45%    -18%   1.2     58%
AAPL             +35%    -22%   0.9     52%
QQQ              +40%    -25%   1.0     55%
SPY              +30%    -20%   0.8     50%

해석:
- ML Model이 모든 지표에서 우수 ✅
- 특히 Sharpe Ratio 1.2 (매우 좋음)
- MDD -18% (리스크 관리 우수)
```

---

## 4. 리포트 생성

### Excel 통합 리포트

```
ml_backtest_report_{timestamp}.xlsx:

Sheet 1: Summary (요약)
┌──────────┬─────────┬─────────┬────────┬──────┐
│ Date     │ Return  │ MDD     │ Sharpe │ Win% │
├──────────┼─────────┼─────────┼────────┼──────┤
│ 2020-01  │ +5.2%   │ -2.1%   │ 1.5    │ 60%  │
│ 2020-04  │ +3.8%   │ -3.2%   │ 1.2    │ 55%  │
│ ...      │ ...     │ ...     │ ...    │ ...  │
└──────────┴─────────┴─────────┴────────┴──────┘

Sheet 2: Detailed (상세)
┌──────────┬────────┬─────────┬────────┬────────┐
│ Date     │ Symbol │ Entry   │ Exit   │ Return │
├──────────┼────────┼─────────┼────────┼────────┤
│ 2020-01  │ AAPL   │ 150.00  │ 158.00 │ +5.3%  │
│ 2020-01  │ MSFT   │ 200.00  │ 208.00 │ +4.0%  │
│ ...      │ ...    │ ...     │ ...    │ ...    │
└──────────┴────────┴─────────┴────────┴────────┘

Sheet 3: Benchmark (벤치마크)
┌──────────┬──────────┬──────┬──────┬──────┐
│ Date     │ ML Model │ AAPL │ QQQ  │ SPY  │
├──────────┼──────────┼──────┼──────┼──────┤
│ 2020-01  │ +5.2%    │ +3.5%│ +4.0%│ +2.8%│
│ 2020-04  │ +3.8%    │ +2.1%│ +3.2%│ +1.9%│
│ ...      │ ...      │ ...  │ ...  │ ...  │
└──────────┴──────────┴──────┴──────┴──────┘
```

---

## 5. 거래일 조정 (Trading Day Adjustment) ✅ 구현 완료

> **상태**: 2026-01-10 완료
> **구현 위치**: `src/backtest/ml_backtest.py` - `_get_trade_date()` 메서드

### 문제

```
리밸런싱 날짜: 2025-01-01 (New Year's Day)
→ 휴장일 → 거래 불가 → 0% 수익률 ❌
```

### 해결

```python
def _get_trade_date(target_date, price_table):
    """
    target_date 이전 10일 내 가장 최근 거래일 반환
    """
    for i in range(10):
        check_date = target_date - timedelta(days=i)
        if check_date in price_table['date'].values:
            return check_date
    
    return None  # 10일 내 거래일 없으면 None

# 사용
for target_date in rebalance_dates:
    actual_trade_date = _get_trade_date(target_date, price_table)
    
    if actual_trade_date is None:
        continue  # 스킵
    
    adjusted_dates.append(actual_trade_date)
```

**효과**:
- 2025-01-01 → 2024-12-31 (이전 거래일)
- 실제 거래 가능한 날짜만 사용
- 0% 수익률 문제 해결

---

## 6. CLASSIFIER_MODE와 Hard Filtering ✅ 구현 완료

> **상태**: 2026-01-10 완료
> **상세 문서**: [03_ml_strategy.md](./03_ml_strategy.md) Section 2

### 개요

백테스트 시 분류기(Classifier)가 두 가지 모드로 작동합니다:

| 모드 | 목적 | 제거 대상 |
|------|------|-----------|
| `negative_screen` | 극단적 손실 종목 제거 | BAD 확률 상위 2~15% |
| `positive_screen` | 저성장 종목 제거 | GOOD 확률 하위 2~15% |

### Hard Filtering 구현

```python
# ml_backtest.py
if mode == 'negative_screen':
    # BAD 확률이 threshold 미만인 종목만 통과
    pass_mask = y_pred_proba < threshold
else:
    # GOOD 확률이 threshold 이상인 종목만 통과
    pass_mask = y_pred_proba >= threshold

# Hard filtering: 미통과 종목은 -inf로 설정
ml_score = np.where(pass_mask, y_pred_return, -np.inf)
```

### 효과

- 필터 미통과 종목이 Top-K 선정에서 완전히 배제됨
- regressor.py와 ml_backtest.py 간 일관성 확보
- 위험 종목 제외로 포트폴리오 안정성 향상

---

## 7. 문제점 및 개선안

### 문제 1: 거래 비용 미반영

**현재**:
```python
# 수익률 계산 시 거래 비용 미반영
return = (exit_price - entry_price) / entry_price
```

**개선안**:
```python
# 거래 비용 반영 (0.1% × 2 = 0.2%)
commission = 0.001  # 0.1%
return = (exit_price - entry_price) / entry_price - 2 * commission
```

**효과**:
- 실전과 동일한 조건
- 수익률 약간 감소 (연 -1~2%)

### 문제 2: 슬리피지 미반영

**현재**:
```python
# 종가 기준 매수/매도
entry_price = close_price
```

**개선안**:
```python
# 슬리피지 반영 (0.1%)
slippage = 0.001
entry_price = close_price * (1 + slippage)  # 매수 시 +0.1%
exit_price = close_price * (1 - slippage)   # 매도 시 -0.1%
```

### 문제 3: Out-of-Sample 기간 짧음

**현재**:
```yaml
TEST_START_YEAR: 2023
TEST_END_YEAR: 2025  # 2년
```

**우려**:
- 시장 사이클 1회도 안 됨
- 다양한 환경 검증 부족

**개선안**:
```yaml
# 여러 기간 백테스트
PERIODS:
  - START_YEAR: 2008  # 금융위기
    END_YEAR: 2009
  
  - START_YEAR: 2015  # 저성장기
    END_YEAR: 2016
  
  - START_YEAR: 2020  # 코로나 충격
    END_YEAR: 2021
  
  - START_YEAR: 2022  # 금리 급등기
    END_YEAR: 2023
```

---

## 8. 시장 레짐 분석

### 레짐 정의

```python
def classify_market_regime(spy_return):
    if spy_return > 0.10:
        return 'Bull'  # 상승장
    elif spy_return < -0.10:
        return 'Bear'  # 하락장
    else:
        return 'Sideways'  # 횡보장
```

### 레짐별 성능

```python
regime_performance = {}

for year in backtest_years:
    spy_return = get_spy_return(year)
    regime = classify_market_regime(spy_return)
    
    ml_sharpe = backtest_results[year]['sharpe']
    regime_performance.setdefault(regime, []).append(ml_sharpe)

for regime, sharpes in regime_performance.items():
    print(f"{regime}: Sharpe {np.mean(sharpes):.2f}")

# 예시 출력:
# Bull: Sharpe 1.5 (좋음)
# Bear: Sharpe 0.8 (보통)
# Sideways: Sharpe 0.5 (나쁨) ← 문제 발견!
```

**효과**:
- 어떤 시장 환경에서 잘 작동하는지 파악
- 약점 발견 → 개선 방향 설정

---

## 결론

### 백테스팅 평가: A-

**강점**:
- Walk-Forward Analysis (A+)
- Filing Date 기준 (A+)
- 벤치마크 비교 (A)
- 거래일 조정 (A) ✅
- CLASSIFIER_MODE + Hard Filtering (A) ✅

**약점**:
- 거래 비용 미반영 (C)
- 슬리피지 미반영 (C)
- Out-of-Sample 짧음 (B)

### 개선 우선순위
1. 거래 비용 + 슬리피지 반영
2. Out-of-Sample 확대
3. 시장 레짐 분석

---

**다음 문서**: [05_code_quality.md](./05_code_quality.md)
