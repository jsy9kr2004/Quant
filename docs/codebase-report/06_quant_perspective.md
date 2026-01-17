# 퀀트 투자 관점 평가

> **작성일**: 2025-12-17
> **이전 문서**: [05_code_quality.md](./05_code_quality.md)
> **다음 문서**: [07_recommendations.md](./07_recommendations.md)

---

## 핵심 요약

### 퀀트 투자 관점 평가: B+ (실전 투자 가능, 단 주의 필요)

**강점**:
- 명확한 투자 철학
- 엄격한 백테스트
- 리스크 인식

**약점**:
- 리스크 관리 미흡
- 프로덕션 준비 부족
- 검증 기간 짧음

---

## 1. 일반적인 퀀트 전략 분류

### Factor-Based vs ML-Based

| 유형 | 설명 | 예시 | 해석 가능성 | 안정성 |
|------|------|------|-------------|--------|
| **Factor-Based** | 알려진 팩터 사용 | Value, Momentum, Quality | 높음 | 높음 |
| **ML-Based** | 데이터 기반 학습 | 이 시스템 | 낮음 | 중간 |

**이 시스템**: ML-Based, 단 펀더멘털 팩터 중심

---

## 2. 투자 철학 평가

### "예측이 아닌 선별"

**학술적 근거**:
- Fama-French 3-Factor Model: Value, Size, Market
- Carhart 4-Factor: + Momentum
- Fama-French 5-Factor: + Profitability, Investment

**이 시스템의 접근**:
```
Classifier: Profitability, Quality 필터링 (Factor-Based 근거)
Regressor: 상대 가치 회귀 (Mean Reversion 가정)
```

**평가**: 학술적 근거 있음 (A)

### 펀더멘털 중심

**일반 퀀트 vs 이 시스템**:
```
일반 퀀트: 가격 데이터 중심 (Technical)
이 시스템: 재무제표 + 가격 (Fundamental + Technical)
```

**장점**:
- 장기적 안정성 (단기 노이즈 적음)
- 투자자 이해 용이

**단점**:
- 분기별 업데이트 (느린 대응)
- 뉴스/이벤트 미반영

**평가**: 장기 투자에 적합 (A)

---

## 3. 시장 효율성 관점

### EMH (Efficient Market Hypothesis)

**약형 효율성 (Weak Form)**:
- 과거 가격만으로 미래 예측 불가
- 이 시스템: 가격 + 재무제표 → 통과 가능 ✅

**준강형 효율성 (Semi-Strong Form)**:
- 공개 정보로 초과 수익 불가
- 이 시스템: 공개 정보 (재무제표) 사용 → 위배 가능성 ⚠️

**해석**:
- 시장이 완전히 효율적이면 이 시스템도 작동 안 함
- 실제로는 비효율성 존재 (Anomaly)
- Value Anomaly, Quality Anomaly 등

**평가**: 실전 테스트 필요

---

## 4. 리스크 관리 평가

### 현재 리스크 관리: C

**구현된 것**:
- Top-K 선택 (분산 투자)
- 3개월 리밸런싱 (과도한 거래 방지)

**미구현**:
- Stop-Loss (손절)
- Position Sizing (비중 조절)
- 섹터 집중도 제한
- VaR/CVaR

### 개선안

#### Stop-Loss

```python
def apply_stop_loss(portfolio, max_loss=-0.15):
    """
    종목별 -15% 손실 시 강제 매도
    """
    for symbol in portfolio:
        current_return = (current_price - entry_price) / entry_price
        
        if current_return < max_loss:
            logger.warning(f"{symbol} hit stop-loss: {current_return:.2%}")
            sell(symbol)
```

#### Position Sizing

```python
def calculate_position_size(symbol, portfolio_value, risk_per_trade=0.02):
    """
    Kelly Criterion 기반 비중 조절
    """
    expected_return = model.predict(symbol)
    volatility = calculate_volatility(symbol)
    
    # Kelly Criterion
    kelly = expected_return / (volatility ** 2)
    
    # 보수적: Kelly의 50%만 사용
    position_size = min(kelly * 0.5, risk_per_trade)
    
    return portfolio_value * position_size
```

#### 섹터 집중도

```python
def check_sector_concentration(portfolio, max_sector_weight=0.3):
    """
    한 섹터 최대 30% 제한
    """
    sector_weights = portfolio.groupby('sector')['weight'].sum()
    
    if (sector_weights > max_sector_weight).any():
        logger.warning(f"Sector concentration exceeded: {sector_weights}")
        # 리밸런싱
```

---

## 5. 일반적인 퀀트 전략과 비교

### A. Value Investing (가치 투자)

**전통적 Value**:
```
Factors: PER, PBR, PSR
Strategy: 저평가 종목 매수
```

**이 시스템**:
```
Factors: PER, PBR, ... + tsfresh features
Strategy: 2-Stage (안정성 필터 + 가치 회귀)
```

**차이점**:
- 전통: Factor만 사용
- 이 시스템: Factor + ML

### B. Momentum Investing (모멘텀 투자)

**전통적 Momentum**:
```
Strategy: 과거 12개월 수익률 상위 매수
```

**이 시스템**:
```
Strategy: Extreme Mover 필터링 (모멘텀 배제!)
```

**철학**:
- 전통: "추세는 계속된다"
- 이 시스템: "극단은 회귀한다" (Mean Reversion)

### C. Quality Investing (품질 투자)

**전통적 Quality**:
```
Factors: ROE, ROA, 부채비율
Strategy: 고품질 기업 매수
```

**이 시스템**:
```
Stage 1 Classifier: 품질 필터링
Stage 2 Regressor: 가치 회귀
```

**평가**: Quality + Value 하이브리드 (A)

---

## 6. 실전 투자 시 주의사항

### 1. 유동성 (Liquidity)

**문제**:
```
소형주 (Small Cap):
- 유동성 낮음
- 슬리피지 큼 (실제 거래가 ≠ 백테스트가)
```

**해결**:
```yaml
FEATURES:
  MIN_VOLUME_PERCENTILE: 10  # 하위 10% 제거
```

**추가 개선**:
```python
# 시가총액 필터
min_market_cap = 1_000_000_000  # $1B
filtered = stocks[stocks['marketCap'] > min_market_cap]
```

### 2. 시장 영향 (Market Impact)

**문제**:
```
대량 매수 시:
- 가격 상승 (불리한 체결)
- 슬리피지 증가
```

**해결**:
```python
# 포트폴리오 크기 제한
max_portfolio_size = total_assets * 0.3  # 총 자산의 30%만
```

### 3. 블랙 스완 (Black Swan)

**문제**:
```
2008 금융위기, 2020 코로나 등:
- 모든 종목 동시 하락
- 분산 투자 무용
```

**해결**:
```python
# MDD 모니터링
if current_mdd < -0.30:  # -30% 이상 손실
    send_alert("Critical MDD! Consider reducing exposure")
```

### 4. Regime Shift (시장 환경 변화)

**문제**:
```
과거 패턴이 미래에도 유효하지 않을 수 있음
예: 저금리 → 고금리 전환
```

**해결**:
```python
# 정기적 재학습 (quarterly)
# 성능 모니터링
if sharpe_ratio_last_6m < 0.5:
    logger.warning("Performance degradation detected")
    # 재학습 또는 전략 변경
```

---

## 7. 헤지펀드/자산운용사 관점

### 도입 가능성: YES, but...

**필요 개선사항**:
1. 단위 테스트 (현재 없음)
2. 프로덕션 모니터링 (Prometheus + Grafana)
3. 리스크 관리 (Stop-Loss, Position Sizing)
4. 규제 준수 (FINRA, SEC)
5. CI/CD 파이프라인
6. 클라우드 배포 (AWS/GCP)
7. 감사 로그 (Audit Trail)

### 예상 수익률

**보수적 추정**:
```
벤치마크 (SPY): +10% CAGR
이 시스템: +15~20% CAGR (초과 수익 +5~10%)
MDD: -20~30%
Sharpe Ratio: 1.0~1.5
```

**근거**:
- 학술 연구: Value + Quality → +3~5% 초과
- ML 효과: +2~5% 추가 (과적합 위험 고려)

---

## 8. 개인 투자자 관점

### 실전 투자 가능? YES

**장점**:
- 명확한 철학
- 자동화 가능
- 장기 투자 적합

**단점**:
- 기술 장벽 (Python, ML)
- 초기 설정 복잡
- API 비용 (FMP: ~$30/month)

### 권장 사항

**소액 투자자 ($1,000~10,000)**:
```
Top-K: 5개
리밸런싱: 분기 (3개월)
비중: 균등 (20% × 5)
```

**중액 투자자 ($10,000~100,000)**:
```
Top-K: 10~15개
리밸런싱: 분기
비중: Kelly Criterion 기반
```

**대액 투자자 ($100,000+)**:
```
Top-K: 20~30개
리밸런싱: 분기
비중: 최적화 (Markowitz)
섹터 제한: 최대 30%
```

---

## 결론

### 퀀트 투자 관점 평가: B+

**강점**:
- 학술적 근거 (A)
- 투자 철학 명확 (A)
- 백테스트 엄격 (A-)

**약점**:
- 리스크 관리 미흡 (C)
- 프로덕션 준비 부족 (C+)
- 검증 기간 짧음 (B)

### 실전 투자 권장

**권장**:
- 소액 파일럿 (총 자산 5%)
- 3~6개월 검증
- 단계적 확대

**비권장**:
- 전 재산 투입
- 레버리지 사용
- 백테스트만 믿고 대규모 투자

---

**다음 문서**: [07_recommendations.md](./07_recommendations.md)
