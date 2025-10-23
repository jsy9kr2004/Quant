# Quick Start Guide - Quant-refactoring

## 🚀 바로 실행하기

### 1. 설정 파일 생성

```bash
cd /home/user/Quant/Quant-refactoring

# 템플릿 복사
cp config/conf.yaml.template config/conf.yaml

# 설정 편집 (YOUR_FMP_API_KEY_HERE를 실제 API 키로 변경)
nano config/conf.yaml  # 또는 vim, vi
```

**최소 필수 설정**:
```yaml
DATA:
  ROOT_PATH: /home/user/Quant/data  # 데이터 저장 경로
  API_KEY: YOUR_ACTUAL_API_KEY      # FMP API 키
  GET_FMP: N                         # 처음엔 N (기존 데이터 사용)

ML:
  RUN_REGRESSION: Y                  # 모델 학습
  EXIT_AFTER_ML: Y                   # ML만 실행

LOG_LVL: 20                          # 로그 레벨 (INFO)
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. 실행

```bash
# Quant-refactoring 디렉토리에서
python main.py
```

---

## 📋 실행 시나리오

### 시나리오 1: 기존 데이터로 ML 학습 (가장 빠름) ⭐

```yaml
DATA:
  GET_FMP: N  # 기존 데이터 사용

ML:
  RUN_REGRESSION: Y  # 학습 실행
  USE_NEW_MODELS: N  # 기존 regressor.py 사용
  EXIT_AFTER_ML: Y   # ML만 하고 종료
```

```bash
python main.py
```

**예상 시간**: 데이터 크기에 따라 10분 ~ 1시간

---

### 시나리오 2: 새 데이터 수집 + ML 학습

```yaml
DATA:
  GET_FMP: Y  # FMP에서 데이터 수집
  API_KEY: your_actual_api_key  # ⚠️ 필수!

ML:
  RUN_REGRESSION: Y
  EXIT_AFTER_ML: Y
```

```bash
python main.py
```

**예상 시간**: 3-4시간 (데이터 수집이 오래 걸림)

---

### 시나리오 3: 백테스팅만 실행

```yaml
DATA:
  GET_FMP: N

ML:
  RUN_REGRESSION: N  # ML 스킵
  EXIT_AFTER_ML: N

BACKTEST:
  RUN_BACKTEST: Y  # 백테스트 실행
```

```bash
python main.py
```

**주의**: plan.csv 파일이 필요합니다.

---

## 🔍 실행 확인

### 성공적인 실행

```
================================================================================
Quant Trading System - Refactored Version
================================================================================
[2025-10-23 11:00:00][INFO] ✅ Configuration loaded from: config/conf.yaml
[2025-10-23 11:00:00][INFO] ================================================================================
[2025-10-23 11:00:00][INFO] Quant Trading System - Refactored
[2025-10-23 11:00:00][INFO] ================================================================================
[2025-10-23 11:00:00][INFO] Data period: 2015 - 2023
[2025-10-23 11:00:00][INFO] Root path: /home/user/Quant/data
[2025-10-23 11:00:00][INFO] ================================================================================
[2025-10-23 11:00:00][INFO] ✅ Configuration validated
...
```

### 흔한 에러와 해결

#### 1. Config file not found

```
❌ Config file not found. Please create config/conf.yaml
```

**해결**:
```bash
cp config/conf.yaml.template config/conf.yaml
```

#### 2. ROOT_PATH not set

```
❌ ROOT_PATH not set in config
```

**해결**: conf.yaml에서 `DATA.ROOT_PATH` 설정

#### 3. Module not found

```
ModuleNotFoundError: No module named 'xxx'
```

**해결**:
```bash
pip install -r requirements.txt
```

#### 4. Legacy regressor not found

```
⚠️ Legacy regressor not found, using new models only
```

**문제 아님**: regressor.py가 없어도 동작합니다. 새 모델 사용.

---

## 📂 디렉토리 구조 (실행 후)

```
Quant-refactoring/
├── config/
│   ├── conf.yaml          # 여기에 실제 설정
│   └── conf.yaml.template # 템플릿
├── data/                  # ROOT_PATH 설정에 따라
│   ├── VIEW/              # 처리된 데이터
│   ├── ml_per_year/       # ML 학습 데이터
│   └── MODELS/            # 학습된 모델
├── reports/               # 백테스트 리포트
└── log.txt               # 실행 로그
```

---

## 🐛 디버깅

### 로그 확인

```bash
# 실시간 로그
tail -f log.txt

# 에러만 보기
grep ERROR log.txt

# 마지막 100줄
tail -100 log.txt
```

### Verbose 로그

conf.yaml:
```yaml
LOG_LVL: 10  # DEBUG 레벨
```

---

## 💡 다음 단계

1. **기본 실행 성공** → ML 모델 학습 확인
2. **새 모델 시도** → `USE_NEW_MODELS: Y` + `USE_MLFLOW: Y`
3. **하이퍼파라미터 튜닝** → `USE_OPTUNA: Y`
4. **백테스팅** → `RUN_BACKTEST: Y`

---

## 📞 도움말

- **에러 발생 시**: log.txt 파일 확인
- **설정 참고**: conf.yaml.template 참조
- **자세한 문서**: README.md 참조
- **통합 가이드**: INTEGRATION_ANALYSIS.md 참조

---

## ⚡ 빠른 테스트

설정 없이 바로 테스트:

```bash
# 기존 루트 디렉토리로 돌아가서
cd /home/user/Quant

# 기존 설정 사용
python main.py  # 기존 버전

# 리팩토링 버전 테스트
cd Quant-refactoring
# (config/conf.yaml 설정 후)
python main.py  # 새 버전
```

두 버전 모두 같은 `data/` 디렉토리 사용 가능!
