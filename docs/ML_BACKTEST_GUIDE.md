# ML Walk-Forward 백테스트 가이드

## 📋 개요

이 문서는 **미래 유출(Future Leakage)을 방지**하면서 ML 모델의 실제 투자 성과를 측정하는 Walk-Forward 백테스트 시스템을 설명합니다.

---

## ⚠️ 미래 유출(Future Leakage)이란?

### **잘못된 백테스트 예시 (미래 유출 발생)**

```
❌ 잘못된 방식:
2020~2024 전체 데이터로 모델 학습
         ↓
2020~2024 기간에 백테스트
         ↓
수익률 200%! 😃

→ 문제: 2020년에 매매할 때 2021~2024년 데이터를 이미 봤음!
→ 현실: 실제로는 절대 불가능한 시나리오
```

### **올바른 백테스트 (Walk-Forward)**

```
✅ 올바른 방식:

[2023-03-13 매매]
  → 2020~2022 데이터로 학습 ───→ 예측 → 매수

[2023-06-13 매매]
  → 2020~2023Q1 데이터로 학습 ───→ 예측 → 매수

[2023-09-13 매매]
  → 2020~2023Q2 데이터로 학습 ───→ 예측 → 매수

→ 각 시점에서 "그때 사용 가능했던 데이터만" 사용!
→ 현실적인 수익률 측정 가능
```

---

## 🔧 핵심 기능

### **1. Filing Date 엄수**

```python
# 재무제표 공시일을 고려
if fillingDate > 매매날짜:
    → 사용 불가! ✗
else:
    → 사용 가능! ✓
```

**예시:**
- 2024 Q1 실적 발표: 2024-05-15
- 2024-04-01에 매매한다면?
  - Q1 실적 아직 모름 → 사용 불가
- 2024-06-01에 매매한다면?
  - Q1 실적 이미 공시됨 → 사용 가능

### **2. Walk-Forward Analysis**

각 리밸런싱 시점마다:
1. 그 시점까지의 데이터만 로드
2. 모델 재학습 (또는 기존 모델 사용)
3. 예측 수행
4. 실제 수익률 계산

### **3. 학습 윈도우 전략**

#### **Expanding Window (권장)**
```
2023-03: [2020────────2022] 학습
2023-06: [2020──────────2023Q1] 학습
2023-09: [2020────────────2023Q2] 학습

→ 데이터가 점점 증가
→ 더 많은 데이터 = 더 나은 학습
```

#### **Rolling Window**
```
2023-03: [2020─2021─2022] 학습 (3년)
2023-06: [2020─2021─2023Q1] 학습 (3년)
2023-09: [2021─2022─2023Q2] 학습 (3년)

→ 최근 N년만 사용
→ 과거 데이터 영향 제거
→ 시장 체제 변화에 민감
```

---

## ⚙️ **설정 파일 구성**

### **위치**
```
/home/user/Quant/Quant-refactoring/config/conf.yaml
```

### **여러 구간 백테스트 (권장)**

```yaml
BACKTEST:
  # 여러 시기를 별도로 백테스트
  PERIODS:
    # 구간 1: 금융위기 (2008-2009)
    - START_YEAR: 2008
      END_YEAR: 2009
      START_MONTH: 3
      START_DATE: 13

    # 구간 2: 최근 (2024-2025)
    - START_YEAR: 2024
      END_YEAR: 2025
      START_MONTH: 3
      START_DATE: 13

  REBALANCE_PERIOD: 3
  TOP_K_NUM: 20
```

**장점:**
- 서로 다른 시장 환경(금융위기 vs 호황기)에서 성과 비교
- 각 구간의 수익률을 별도로 측정
- 모델의 견고성 검증

### **단일 구간 백테스트**

```yaml
BACKTEST:
  # PERIODS 섹션 제거하고 단일 구간 설정
  START_YEAR: 2023
  END_YEAR: 2024
  START_MONTH: 3
  START_DATE: 13

  REBALANCE_PERIOD: 3
  TOP_K_NUM: 20
```

---

## 🚀 사용법

### **기본 실행**

```bash
cd /home/user/Quant/Quant-refactoring

# config/conf.yaml에 설정된 구간으로 실행
python scripts/run_ml_backtest.py
```

### **고급 옵션**

```bash
# 1. 매 리밸런싱마다 재학습 (가장 정확, 느림)
python scripts/run_ml_backtest.py \
    --retrain-freq every \
    --window expanding \
    --top-k 20

# 2. Rolling 3년 window
python scripts/run_ml_backtest.py \
    --retrain-freq quarterly \
    --window rolling \
    --window-size 3

# 3. 연도마다만 재학습 (빠름)
python scripts/run_ml_backtest.py \
    --retrain-freq yearly \
    --window expanding

# 4. 한 번만 학습 (현재 방식, 비추천)
python scripts/run_ml_backtest.py \
    --retrain-freq once
```

### **파라미터 설명**

| 파라미터 | 옵션 | 설명 |
|---------|------|------|
| `--retrain-freq` | `every` | 매 리밸런싱마다 재학습 (가장 정확) |
|  | `quarterly` | 분기마다 재학습 (권장) |
|  | `yearly` | 연도마다 재학습 (빠름) |
|  | `once` | 한 번만 학습 (비추천) |
| `--window` | `expanding` | 점점 증가하는 윈도우 (권장) |
|  | `rolling` | 고정 크기 윈도우 |
| `--window-size` | 숫자 | Rolling window 크기 (년) |
| `--top-k` | 숫자 | 선택할 종목 수 (기본값: 20) |
| `--rebalance-period` | 숫자 | 리밸런싱 주기 (개월, 기본값: 3) |

---

## 📊 출력 결과

### **콘솔 출력**

```
================================================================================
ML Walk-Forward Backtest Starting
================================================================================
Rebalance period: 3 months
Top K: 20
Retrain frequency: quarterly
Window type: expanding

📅 Rebalance dates: 8
   2023-03-13
   2023-06-13
   2023-09-13
   ...

================================================================================
Rebalance #1: 2023-03-13
================================================================================
📊 Available data until 2023-03-13: 45230 rows
🔧 Training models with data until 2023-03-13
   Training samples: 45230
   Training classifier...
   Training regressor...
   ✅ Model saved: MODELS_WALKFORWARD/model_20230313.pkl
📊 Selected 20 stocks
💰 Period return: 5.23%

================================================================================
Rebalance #2: 2023-06-13
================================================================================
📊 Available data until 2023-06-13: 48450 rows
🔧 Training models with data until 2023-06-13
...

================================================================================
BACKTEST SUMMARY
================================================================================
Total Periods: 8
Total Return: 32.45%
Average Return: 3.82%
Std Dev: 4.15%
Sharpe Ratio: 1.85
Max Drawdown: -8.32%
Win Rate: 75.0%
Models Retrained: 4 times

✅ Results saved: ./reports/ml_backtest_20251120_142530.csv
```

### **CSV 파일**

`./reports/ml_backtest_YYYYMMDD_HHMMSS.csv`:

```csv
date,selected_symbols,return,retrained
2023-03-13,"['AAPL', 'MSFT', 'GOOGL', ...]",0.0523,True
2023-06-13,"['NVDA', 'META', 'TSLA', ...]",0.0412,True
2023-09-13,"['AAPL', 'AMZN', 'MSFT', ...]",-0.0132,False
...
```

---

## 🎯 기존 방식과의 비교

### **기존: regressor.py (정적 모델)**

```python
# 한 번만 학습
TRAIN: 2020-2022 → 모델 A 학습
TEST: 2023-2024 → 모델 A로 예측

문제점:
❌ 2024년에도 2022년 데이터로 학습한 모델 사용
❌ 2023년 데이터를 학습에 활용 못 함
❌ "예측 정확도"만 측정 (실제 수익률 X)
```

### **신규: ml_backtest.py (Walk-Forward)**

```python
# 각 시점마다 재학습
2023-03: 2020-2022 → 모델 A → 예측 → 실제 수익률 계산
2023-06: 2020-2023Q1 → 모델 B → 예측 → 실제 수익률 계산
2023-09: 2020-2023Q2 → 모델 C → 예측 → 실제 수익률 계산
...

장점:
✓ 최신 데이터로 계속 업데이트
✓ 실제 투자 시나리오 반영
✓ 진짜 수익률 측정 가능
✓ 미래 유출 완전 차단
```

---

## 💡 권장 설정

### **개발/테스트 단계**

```bash
# 빠른 테스트 (연도마다만 재학습)
python scripts/run_ml_backtest.py \
    --retrain-freq yearly \
    --window expanding \
    --top-k 10
```

### **실전 검증**

```bash
# 분기마다 재학습 (현실적인 균형점)
python scripts/run_ml_backtest.py \
    --retrain-freq quarterly \
    --window expanding \
    --top-k 20
```

### **최대 정확도**

```bash
# 매번 재학습 (가장 정확하지만 느림)
python scripts/run_ml_backtest.py \
    --retrain-freq every \
    --window expanding \
    --top-k 20
```

---

## 🔍 트러블슈팅

### **문제: "No data available until YYYY-MM-DD"**

**원인:** ML 데이터 파일(`rnorm_ml_*.parquet`)이 없음

**해결:**
```bash
# ML 데이터 먼저 생성
cd /home/user/Quant/Quant-refactoring
python main.py  # with ML.RUN_REGRESSION: Y
```

### **문제: "Prediction file not found"**

**원인:** 예측용 파일(`rnorm_fs_*.parquet`)이 없음

**해결:**
- `make_mldata.py`가 `rnorm_fs_*` 파일도 생성하는지 확인
- 또는 `rnorm_ml_*` 파일을 대신 사용하도록 코드 수정

### **문제: 백테스트가 너무 느림**

**해결:**
```bash
# 재학습 빈도 줄이기
python scripts/run_ml_backtest.py --retrain-freq yearly

# 또는 Rolling window 사용
python scripts/run_ml_backtest.py \
    --window rolling \
    --window-size 2
```

---

## 📚 추가 자료

- [WORKFLOW_GUIDE.md](./WORKFLOW_GUIDE.md) - 전체 시스템 워크플로우
- [README.md](../README.md) - 프로젝트 개요
- [config/conf.yaml.template](../config/conf.yaml.template) - 설정 템플릿

---

## 🤝 기여

버그 리포트나 개선 제안은 GitHub Issues로 제출해주세요.

---

**작성:** Quant Trading Team
**날짜:** 2025-11-20
**버전:** 1.0.0
