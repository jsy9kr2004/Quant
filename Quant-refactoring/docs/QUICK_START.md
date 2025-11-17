# 🚀 빠른 시작 가이드

실전 트레이딩 시스템 사용법

---

## 📖 이 문서는 누구를 위한 것인가?

- ✅ **처음 이 프로젝트를 실행하는 분** - 가장 먼저 읽으세요!
- ✅ **빠르게 결과를 확인하고 싶은 분** - 5-10분이면 충분합니다
- ✅ **정기적으로 시스템을 실행하는 분** - 매번 참고하기 좋습니다

**다음 단계:**
- 시스템 구조를 이해하고 싶다면 → [../README.md](../README.md)
- 아키텍처를 깊이 이해하고 싶다면 → [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md)
- 고급 기능을 활용하고 싶다면 → [ADVANCED_FEATURES_GUIDE.md](ADVANCED_FEATURES_GUIDE.md)

---

## 📋 사전 준비

### 1. 데이터 준비

```bash
# 데이터 위치
./VIEW/
├── symbol_list.csv                    # 심볼 목록 (sector 정보 포함)
├── price.csv                          # 가격 데이터
├── financial_statement_2018.csv       # 재무제표
├── financial_statement_2019.csv
├── ...
└── financial_statement_2023.csv
```

### 2. 필수 컬럼

**symbol_list.csv**:
- `symbol`: 주식 심볼
- `sector`: 섹터 정보 (Technology, Financial, Healthcare, Consumer, Industrial 등)

**financial_statement_YYYY.csv**:
- `symbol`: 주식 심볼
- `date`: 날짜
- `close`: 종가
- 재무 지표들 (revenue, netIncome, totalAssets, ...)

---

## 🎯 사용 방법

### Option 1: 전체 파이프라인 실행 (권장)

```bash
cd Quant-refactoring/scripts
python run_full_pipeline.py
```

이 스크립트는 3단계를 순차적으로 실행합니다:
1. 리밸런싱 기간 최적화 (선택 사항)
2. 모델 성능 비교 (선택 사항)
3. 섹터별 주식 선택 (필수)

**설정 변경**:
```python
# run_full_pipeline.py 파일에서
RUN_REBALANCE_OPTIMIZATION = False  # 최초 1회만 True
RUN_MODEL_COMPARISON = False         # 모델 변경 시에만 True
RUN_SECTOR_TRADING = True            # 매번 True
```

---

### Option 2: 개별 실행

#### 1️⃣ 리밸런싱 기간 최적화 (최초 1회)

```bash
python scripts/run_rebalance_optimization.py
```

**출력**:
- 최적 리밸런싱 기간
- 각 기간별 성능 메트릭
- 결과 저장: `./results/rebalancing_optimization/`
- 설정 저장: `./config/optimal_rebalance_period.txt`

#### 2️⃣ 모델 성능 비교 (모델 변경 시)

```bash
python scripts/run_model_comparison.py
```

**출력**:
- 여러 모델 버전 비교
- 통계적 유의성 검정
- 최고 성능 모델 선택
- 결과 저장: `./results/model_comparison/`
- 설정 저장: `./config/best_model.txt`

#### 3️⃣ 섹터별 주식 선택 (매 리밸런싱마다)

```bash
python scripts/run_sector_trading.py
```

**출력**:
- 선택된 top N 주식
- 섹터별 분포
- 예측 점수
- 결과 저장: `./results/sector_trading/selected_stocks_YYYYMMDD_HHMMSS.csv`

---

## 📊 실행 흐름

```
┌─────────────────────────────────────────┐
│  1. 리밸런싱 기간 최적화 (최초 1회)       │
│                                         │
│  • 1개월, 2개월, 3개월, 6개월 등 테스트  │
│  • 최적 기간 선택 (예: 3개월)            │
│  • ./config/optimal_rebalance_period.txt│
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  2. 모델 성능 비교 (모델 변경 시)         │
│                                         │
│  • XGBoost v1 vs v2                     │
│  • LightGBM vs CatBoost                 │
│  • 통계적 검증                          │
│  • ./config/best_model.txt              │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  3. 섹터별 주식 선택 (매 리밸런싱)        │
│                                         │
│  ┌──────────┐  ┌──────────┐             │
│  │Technology│  │Financial │             │
│  │다른 피처  │  │다른 피처  │             │
│  │다른 모델  │  │다른 모델  │             │
│  │  ↓       │  │  ↓       │             │
│  │예측 점수  │  │예측 점수  │   ...       │
│  └──────────┘  └──────────┘             │
│         ↓           ↓                   │
│    ┌───────────────────┐                │
│    │ 전체에서 top 10   │                │
│    │ 선택 (섹터 무관)  │                │
│    └───────────────────┘                │
│                                         │
│  결과: TECH_001, FIN_002, HLTH_003, ... │
└─────────────────────────────────────────┘
```

---

## 🔧 설정 커스터마이징

### 리밸런싱 기간 변경

`scripts/run_rebalance_optimization.py`:
```python
PERIODS_TO_TEST = [1, 2, 3, 4, 6, 12]  # 테스트할 기간들
```

### 선택할 주식 개수 변경

`scripts/run_sector_trading.py`:
```python
TOP_K = 10  # 5개 or 10개 or 20개
```

### 섹터별 피처 변경

`strategy/sector_ensemble.py`의 `create_default_sector_configs()` 함수 수정:

```python
'Technology': {
    'feature_cols': [
        'revenue',
        'OverMC_researchAndDevelopmentExpenses',  # Tech에 중요!
        'priceToBookRatio',
        # ... 추가/제거
    ],
    'model_params': {'n_estimators': 100, 'max_depth': 6}
}
```

---

## 📈 실전 사용 예시

### 시나리오: 2024년 1월 1일 리밸런싱

```bash
# 1. (최초 1회) 최적 리밸런싱 기간 찾기
python scripts/run_rebalance_optimization.py
# 출력: 최적 기간 = 3개월

# 2. (모델 변경 시) 모델 성능 비교
python scripts/run_model_comparison.py
# 출력: XGBoost_v2가 v1보다 3% 개선

# 3. (매 리밸런싱) 주식 선택
python scripts/run_sector_trading.py
# 출력:
# 1. TECH_001 (Technology, 점수: 0.92)
# 2. FIN_002 (Financial, 점수: 0.88)
# 3. HLTH_003 (Healthcare, 점수: 0.85)
# ...
```

### 결과 확인

```bash
cat ./results/sector_trading/selected_stocks_20240101_090000.csv
```

---

## 🔄 정기 실행 (자동화)

### Cron Job 설정

```bash
# 매달 1일 오전 9시에 실행
0 9 1 * * cd /path/to/Quant-refactoring && python scripts/run_sector_trading.py >> logs/cron.log 2>&1
```

### Python 스케줄러

```python
import schedule
import time

def job():
    os.system('python scripts/run_sector_trading.py')

# 매달 1일 실행
schedule.every().month.at("09:00").do(job)

while True:
    schedule.run_pending()
    time.sleep(3600)  # 1시간마다 체크
```

---

## 📂 출력 파일 구조

```
results/
├── rebalancing_optimization/
│   ├── optimization_results_20240101_080000.csv
│   ├── optimization_results_20240101_080000_summary.txt
│   └── optimization_chart_20240101_080000.png
├── model_comparison/
│   ├── model_comparison_20240101_083000/
│   │   ├── model_comparison_20240101_083000.csv
│   │   ├── model_comparison_20240101_083000_metadata.json
│   │   └── model_comparison_20240101_083000_summary.txt
│   └── comparison_chart_20240101_083000.png
└── sector_trading/
    ├── selected_stocks_20240101_090000.csv      ← 이 파일 확인!
    ├── sector_config_20240101_090000.json
    └── sector_summary_20240101_090000.csv
```

---

## 🐛 문제 해결

### 데이터 없음 오류

```
ERROR: Price 파일을 찾을 수 없습니다
```

**해결**: `./VIEW/` 디렉토리에 데이터 파일 확인

### 섹터 정보 없음

```
WARNING: 섹터 정보가 없는 주식 제외
```

**해결**: `symbol_list.csv`에 `sector` 컬럼 추가

### 선택된 주식 없음

```
ERROR: 선택된 주식이 없습니다!
```

**해결**:
1. 데이터 기간 확인
2. 피처 NaN 비율 확인
3. 모델 학습 로그 확인

---

## 💡 팁

### 1. 최초 실행 순서

```bash
# 1단계: 샘플 데이터로 테스트
cd examples
python comprehensive_example.py

# 2단계: 실제 데이터로 리밸런싱 기간 최적화
cd ../scripts
python run_rebalance_optimization.py

# 3단계: 모델 비교
python run_model_comparison.py

# 4단계: 주식 선택
python run_sector_trading.py
```

### 2. 빠른 테스트

```python
# run_sector_trading.py에서
TRAIN_YEARS = 1  # 3 → 1로 변경 (빠른 테스트)
```

### 3. 로그 확인

```bash
tail -f logs/sector_trading.log
```

---

## 📚 추가 문서

- **ROBUST_VALIDATION_GUIDE.md**: 백테스팅 및 검증 시스템
- **ADVANCED_FEATURES_GUIDE.md**: 3가지 핵심 기능 상세 설명
- **examples/**: 예제 스크립트

---

## 🤝 문의

버그 리포트 및 개선 제안은 이슈로 등록해주세요.
