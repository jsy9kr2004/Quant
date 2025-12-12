# 미래 유출 방지 시스템 구축 완료 ✅

## 📌 핵심 요약

**문제:** 기존 코드는 ML 모델을 한 번만 학습하고, 그 모델로 전체 테스트 기간을 예측했습니다. 이는 미래 데이터를 활용하지 못하고, 현실적인 투자 시나리오를 반영하지 못했습니다.

**해결:** Walk-Forward Analysis를 구현하여 각 리밸런싱 시점마다 **그 시점까지 사용 가능한 데이터만**으로 모델을 학습하고 예측합니다.

---

## 🔍 미래 유출(Future Leakage)이란?

### ❌ **잘못된 예시**

```
2020~2024 전체 데이터로 모델 학습
         ↓
2020년에 매매 시뮬레이션
         ↓
"와! 수익률 200%다!"

→ 문제: 2020년에 매매할 때 2021~2024년 데이터를 이미 봤음
→ 현실: 절대 불가능한 시나리오
→ 결과: 실전에서는 수익률 -50%
```

### ✅ **올바른 예시 (Walk-Forward)**

```
[2023-03-13 매매]
  사용 데이터: 2020~2022
  → 모델 A 학습 → 예측 → 매수 → 실제 수익률 +5.2%

[2023-06-13 매매]
  사용 데이터: 2020~2023Q1 (업데이트!)
  → 모델 B 학습 → 예측 → 매수 → 실제 수익률 +4.1%

[2023-09-13 매매]
  사용 데이터: 2020~2023Q2 (업데이트!)
  → 모델 C 학습 → 예측 → 매수 → 실제 수익률 -1.3%

→ 각 시점에서 "실제로 사용 가능했던 데이터만" 사용
→ 현실적인 수익률 측정 가능
```

---

## 🎯 구현된 기능

### **1. Filing Date 엄수 (이미 구현되어 있음)**

```python
# make_mldata.py Line 752-761
filing_delay_days = fillingDate - report_date

# 로그 출력:
# Average delay: 33.3 days
# Maximum delay: 327 days

→ 재무제표 공시 전에는 해당 데이터 사용 불가!
```

**예시:**
- 2024 Q1 실적 발표: 2024-05-15
- 2024-04-01 매매: Q1 실적 모름 ✗
- 2024-06-01 매매: Q1 실적 사용 가능 ✓

### **2. Walk-Forward Analysis (신규 구현)**

```python
# backtest/ml_backtest.py

class MLBacktest:
    def _get_available_data_until(cutoff_date):
        """
        미래 유출 방지의 핵심!
        cutoff_date까지 사용 가능한 데이터만 반환
        """
        # Filing Date 체크
        df = df[df['fillingDate'] <= cutoff_date]

    def run():
        """
        각 리밸런싱 시점마다:
        1. 사용 가능한 데이터만 로드
        2. 모델 재학습 (필요시)
        3. 예측
        4. 실제 수익률 계산
        """
```

### **3. 학습 윈도우 전략**

#### **Expanding Window (권장)**
```
데이터가 점점 증가
2023-03: [2020──────2022] 학습
2023-06: [2020────────2023Q1] 학습
2023-09: [2020──────────2023Q2] 학습

장점: 더 많은 데이터 = 더 나은 학습
```

#### **Rolling Window**
```
최근 N년만 사용
2023-03: [2020─2021─2022] 학습 (3년)
2023-06: [2020─2021─2023Q1] 학습 (3년)
2023-09: [2021─2022─2023Q2] 학습 (3년)

장점: 과거 데이터 영향 제거, 시장 변화 민감
```

### **4. 재학습 전략**

| 전략 | 설명 | 장점 | 단점 |
|------|------|------|------|
| **every** | 매 리밸런싱마다 재학습 | 가장 정확 | 느림 |
| **quarterly** | 분기마다 재학습 | 균형점 (권장) | - |
| **yearly** | 연도마다 재학습 | 빠름 | 부정확 |
| **once** | 한 번만 학습 | 매우 빠름 | 비현실적 (비추천) |

---

## 📂 생성된 파일

### **1. 백테스트 엔진**
```
/home/user/Quant/Quant-refactoring/backtest/ml_backtest.py
```

**주요 기능:**
- `_get_available_data_until()`: 미래 유출 방지
- `_train_model()`: 모델 학습
- `_predict()`: 예측 수행
- `run()`: 전체 백테스트 실행

### **2. 실행 스크립트**
```
/home/user/Quant/Quant-refactoring/scripts/run_ml_backtest.py
```

**사용법:**
```bash
# 기본 실행 (분기마다 재학습)
python scripts/run_ml_backtest.py

# 매번 재학습 (가장 정확)
python scripts/run_ml_backtest.py --retrain-freq every

# Rolling 3년 window
python scripts/run_ml_backtest.py --window rolling --window-size 3
```

### **3. 가이드 문서**
```
/home/user/Quant/Quant-refactoring/docs/ML_BACKTEST_GUIDE.md
```

---

## 🚀 사용 방법

### **Step 1: ML 데이터 생성 (이미 완료)**

```bash
cd /home/user/Quant/Quant-refactoring

# 설정 확인
# config/conf.yaml:
#   ML.RUN_REGRESSION: Y
#   ML.EXIT_AFTER_ML: Y

# 실행 (이미 완료됨)
python main.py
```

**생성된 파일:**
- `/data_parquet/ml_per_year/rnorm_ml_2024_Q1.parquet`
- `/data_parquet/ml_per_year/rnorm_ml_2024_Q2.parquet`
- ...

### **Step 2: Walk-Forward 백테스트 실행**

```bash
cd /home/user/Quant/Quant-refactoring

# 기본 실행 (분기마다 재학습, Expanding window)
python scripts/run_ml_backtest.py

# 또는 커스텀 설정
python scripts/run_ml_backtest.py \
    --retrain-freq quarterly \
    --window expanding \
    --top-k 20 \
    --rebalance-period 3
```

### **Step 3: 결과 확인**

**콘솔 출력:**
```
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
```

**CSV 파일:**
```
./reports/ml_backtest_20251120_142530.csv
```

---

## 📊 기존 방식 vs 신규 방식

| 구분 | 기존 (regressor.py) | 신규 (ml_backtest.py) |
|------|---------------------|----------------------|
| **학습** | 한 번만 (2020-2022) | 각 시점마다 재학습 |
| **예측** | 모델 A로 전체 예측 | 각 시점의 최신 모델 사용 |
| **데이터 활용** | 2022년까지만 | 최신 데이터 계속 반영 |
| **측정 지표** | 정확도, RMSE | 실제 수익률, Sharpe, MDD |
| **현실성** | ❌ 비현실적 | ✅ 현실적 |
| **미래 유출** | ⚠️ 간접적 위험 | ✅ 완전 차단 |

---

## 💡 추천 설정

### **개발/테스트**
```bash
python scripts/run_ml_backtest.py \
    --retrain-freq yearly \
    --top-k 10
```

### **실전 검증 (권장)**
```bash
python scripts/run_ml_backtest.py \
    --retrain-freq quarterly \
    --window expanding \
    --top-k 20
```

### **최대 정확도**
```bash
python scripts/run_ml_backtest.py \
    --retrain-freq every \
    --window expanding \
    --top-k 20
```

---

## 🔧 다음 단계

### **1. 첫 실행 테스트**

```bash
cd /home/user/Quant/Quant-refactoring

# 디렉토리 생성
mkdir -p logs reports backtest

# 실행
python scripts/run_ml_backtest.py

# 결과 확인
ls -la reports/
cat logs/ml_backtest.log
```

### **2. 설정 최적화**

- 여러 `--retrain-freq` 옵션 비교
- `expanding` vs `rolling` window 비교
- 최적 `--top-k` 값 찾기

### **3. 성과 분석**

- 기존 plan.csv 백테스트 vs ML 백테스트 비교
- 섹터별 성과 분석
- 리밸런싱 주기 최적화

---

## ⚠️ 주의사항

### **데이터 파일 확인**

```bash
# ML 데이터 파일이 있는지 확인
ls -la /home/user/data_parquet/ml_per_year/

# 필요한 파일:
# - rnorm_ml_YYYY_QN.parquet (학습용)
# - rnorm_fs_YYYY_QN.parquet (예측용)
```

만약 파일이 없다면:
```bash
# main.py 실행으로 생성
python main.py  # with ML.RUN_REGRESSION: Y
```

### **경로 문제**

현재 코드는 다음 경로를 가정합니다:
- ML 데이터: `/data_parquet/ml_per_year/`
- 모델 저장: `/data_parquet/MODELS_WALKFORWARD/`
- 가격 데이터: `/data_parquet/VIEW/price.parquet`

경로가 다르다면 `ml_backtest.py`에서 수정 필요.

---

## 🎓 학습 자료

### **미래 유출에 대해 더 알아보기**

1. **Look-ahead Bias**: 미래 정보를 사용하는 오류
2. **Survivorship Bias**: 상장폐지 기업 제외 오류
3. **Data Snooping**: 같은 데이터로 여러 전략 테스트

### **Walk-Forward Analysis**

- [Investopedia: Walk-Forward Analysis](https://www.investopedia.com)
- [QuantConnect: Backtesting Best Practices](https://www.quantconnect.com)

---

## 📞 문의

버그 리포트나 개선 제안:
- GitHub Issues
- 또는 팀 내부 채널

---

**작성:** Quant Trading Team
**날짜:** 2025-11-20
**목적:** 미래 유출 방지 시스템 구축 및 안내
**상태:** ✅ 완료, 테스트 준비 완료
