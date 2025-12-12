# 🔧 문제 해결 가이드 (Troubleshooting)

> **목적**: 시스템 설치 및 실행 중 발생하는 문제 해결
> **작성일**: 2025-11-17
> **대상**: 모든 사용자

---

## 📖 이 문서는 누구를 위한 것인가?

- ✅ **에러가 발생한 분** - 문제 해결 방법을 찾고 있는 분
- ✅ **처음 설치하는 분** - 흔한 설치 에러 사전 방지
- ✅ **의존성 문제를 겪는 분** - PyArrow, Datasets 등 패키지 호환성
- ✅ **Linux/Mac 사용자** - 크로스 플랫폼 지원

**다른 문제가 있나요?**
- 시스템 이해 → [../README.md](../README.md)
- 빠른 시작 → [QUICK_START.md](QUICK_START.md)

---

## 📋 목차

1. [일반적인 에러](#1-일반적인-에러)
2. [설치 문제](#2-설치-문제)
3. [실행 문제](#3-실행-문제)
4. [데이터 관련 에러](#4-데이터-관련-에러)
5. [플랫폼별 가이드](#5-플랫폼별-가이드)
6. [성능 문제](#6-성능-문제)
7. [빠른 참조](#7-빠른-참조)

---

## 1. 일반적인 에러

### 1-1. PyArrow 호환성 에러

**에러 메시지:**
```
AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'
```

**해결 방법:**

<details>
<summary><b>Windows (PowerShell/CMD)</b></summary>

```powershell
# PyArrow 및 Datasets 업그레이드
pip install --upgrade pyarrow>=14.0.0 datasets>=2.14.0

# 또는 requirements 재설치
pip install -r requirements.txt
```
</details>

<details>
<summary><b>Linux (Ubuntu/Debian)</b></summary>

```bash
# PyArrow 및 Datasets 업그레이드
pip install --upgrade pyarrow>=14.0.0 datasets>=2.14.0

# 또는 pip3 사용
pip3 install --upgrade pyarrow>=14.0.0 datasets>=2.14.0
```
</details>

<details>
<summary><b>macOS</b></summary>

```bash
# PyArrow 및 Datasets 업그레이드
pip install --upgrade pyarrow>=14.0.0 datasets>=2.14.0

# 또는 pip3 사용
pip3 install --upgrade pyarrow>=14.0.0 datasets>=2.14.0
```
</details>

### 1-2. Module Import 에러

**에러 메시지:**
```
ModuleNotFoundError: No module named 'xxxxx'
```

**해결 방법:**

모든 플랫폼:
```bash
# 전체 의존성 설치
pip install -r requirements.txt

# 또는 개별 모듈 설치
pip install [모듈명]
```

### 1-3. Python 버전 에러

**에러 메시지:**
```
SyntaxError: ...
```

**확인 및 해결:**

```bash
# Python 버전 확인
python --version
# 또는
python3 --version

# Python 3.8 이상 필요 (권장: 3.10)
```

**Python 버전이 낮은 경우:**
- Windows: python.org에서 최신 버전 설치
- Linux: `sudo apt install python3.10` (Ubuntu/Debian)
- macOS: `brew install python@3.10`

---

## 2. 설치 문제

### 2-1. Permission 에러

**에러 메시지:**
```
PermissionError: [Errno 13] Permission denied
```

**해결 방법:**

<details>
<summary><b>Windows</b></summary>

```powershell
# 관리자 권한으로 PowerShell 실행
# 또는 --user 옵션 사용
pip install --user pyarrow
```
</details>

<details>
<summary><b>Linux/macOS</b></summary>

```bash
# --user 옵션 사용 (권장)
pip install --user pyarrow

# 또는 sudo 사용 (주의)
sudo pip install pyarrow
```
</details>

### 2-2. 가상 환경 설정 (권장)

깨끗한 환경에서 시작하려면 가상 환경을 사용하세요:

<details>
<summary><b>모든 플랫폼</b></summary>

```bash
# 가상 환경 생성
python -m venv venv_quant

# 활성화 (Windows)
venv_quant\Scripts\activate

# 활성화 (Linux/macOS)
source venv_quant/bin/activate

# 패키지 설치
pip install -r requirements.txt

# 비활성화
deactivate
```
</details>

---

## 3. 실행 문제

### 3-1. FMP API 키 에러

**에러 메시지:**
```
API Error: Invalid API Key
Authentication failed
```

**해결 방법:**

1. `config/conf.yaml` 파일 확인:
```yaml
DATA:
  API_KEY: "your_actual_api_key_here"  # 실제 키로 변경
```

2. API 키 발급: https://financialmodelingprep.com/developer/docs

### 3-2. 스크립트 실행 문제

**문제**: 어떤 스크립트를 실행해야 할지 모름

**해결:**

#### ✅ 올바른 방법

<details>
<summary><b>Windows</b></summary>

```powershell
cd Quant-refactoring\scripts
python run_sector_trading.py
```
</details>

<details>
<summary><b>Linux/macOS</b></summary>

```bash
cd Quant-refactoring/scripts
python run_sector_trading.py
# 또는
python3 run_sector_trading.py
```
</details>

#### ❌ 잘못된 방법
```bash
python main.py  # 이것은 레거시 코드입니다
```

### 3-3. 메모리 부족 에러

**에러 메시지:**
```
MemoryError
numpy.core._exceptions.MemoryError
```

**해결 방법:**

1. **배치 크기 줄이기** (`config/conf.yaml`):
```yaml
ML:
  BATCH_SIZE: 1000  # 기본값보다 작게
  CHUNK_SIZE: 500
```

2. **데이터 필터링**:
```python
# 특정 연도만 로드
DATA:
  START_YEAR: 2020  # 2015 대신
  END_YEAR: 2023
```

3. **시스템 메모리 확인**:
```bash
# Windows
systeminfo | findstr /C:"Total Physical Memory"

# Linux
free -h

# macOS
top -l 1 | grep PhysMem
```

### 3-4. GPU/CUDA 에러

**에러 메시지:**
```
CUDA error: out of memory
CUDARuntimeError
```

**해결 방법:**

1. **CPU로 전환** (`models/config.py`):
```python
# XGBoost
'tree_method': 'hist',  # 'gpu_hist' 대신

# LightGBM
'device': 'cpu',  # 'gpu' 대신
```

2. **GPU 메모리 확인**:
```bash
# NVIDIA GPU
nvidia-smi
```

### 3-5. XGBoost 3.1+ 호환성 에러

XGBoost 3.0에서 3.1로 업그레이드하면서 GPU 관련 파라미터가 변경되었습니다.

#### 에러 1: gpu_id 파라미터 에러

**에러 메시지:**
```
XGBoostError: Invalid Parameter format for gpu_id
XGBoostError: gpu_id has been removed since 3.1
```

**원인**: XGBoost 3.1 이상에서는 `gpu_id` 파라미터가 제거됨

**해결 방법:**

`models/config.py` 또는 모델 설정 파일 수정:

```python
# ❌ 잘못된 설정 (XGBoost < 3.1)
xgb_params = {
    'gpu_id': 0,
    'tree_method': 'gpu_hist'
}

# ✅ 올바른 설정 (XGBoost >= 3.1)
xgb_params = {
    'device': 'cuda:0',      # 첫 번째 GPU
    'tree_method': 'hist'    # 'gpu_hist' 대신 'hist' 사용
}

# CPU 사용
xgb_params = {
    'device': 'cpu',
    'tree_method': 'hist'
}
```

#### 에러 2: tree_method 에러

**에러 메시지:**
```
XGBoostError: Invalid Input: 'gpu_hist', valid values are: {'approx', 'auto', 'exact', 'hist'}
```

**원인**: XGBoost 3.1+에서는 `gpu_hist`가 제거되고 `hist`로 통합됨

**해결 방법:**

```python
# ❌ 잘못된 설정
'tree_method': 'gpu_hist'

# ✅ 올바른 설정
'tree_method': 'hist'     # GPU는 device 파라미터로 지정
'device': 'cuda:0'        # GPU 사용
# 또는
'device': 'cpu'           # CPU 사용
```

#### 에러 3: 무한대 값으로 인한 학습 실패

**에러 메시지:**
```
XGBoostError: Invalid input - contains inf or -inf
XGBoostError: Invalid input - data contains NaN
ValueError: Input contains infinity or a value too large
```

**원인**:
- VIEW 데이터에 포함된 무한대 값 (division by zero)
- RobustScaler가 IQR=0인 경우 무한대 생성
- 특정 재무 지표의 극단적인 값

**해결 방법**:

**이 문제는 PR #56, #60, #62에서 자동으로 해결되었습니다:**
- ✅ VIEW 데이터 로드 시 무한대 값을 자동으로 NaN으로 변환
- ✅ 스케일링 전후로 무한대 값 자동 제거
- ✅ 제거된 데이터의 상세 리포트 CSV 파일로 출력 (선택적)

**진단 활성화** (선택 사항):

`config/conf.yaml` 파일에서:
```yaml
ML:
  EXPORT_NAN_REMOVAL_DETAILS: Y  # 제거된 데이터 추적 활성화
```

출력 파일 위치:
```
NAN_REMOVAL_DETAILS/
├── nan_removal_rows_2024_Q1.csv      # 제거된 실제 row 데이터
├── nan_removal_summary_2024_Q1.csv   # 제거 통계 요약
└── ...
```

**로그 확인**:
```
INFO: Converted 1234 infinite values to NaN in fs_table
INFO: Converted 567 infinite values to NaN in metrics_table
INFO: Removed 89 rows containing infinite values before scaling
INFO: Removed 12 rows containing infinite values after scaling
```

#### XGBoost 버전 확인

```bash
python -c "import xgboost; print(xgboost.__version__)"
# 3.1.0 이상이어야 함
```

**XGBoost 업그레이드**:
```bash
pip install --upgrade xgboost>=3.1.0
```

---

## 4. 데이터 관련 에러

### 4-1. 데이터 파일 누락

**에러 메시지:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'VIEW/price.csv'
```

**해결 방법:**

1. **데이터 경로 확인** (`scripts/run_sector_trading.py`):
```python
# 25번째 줄 근처
DATA_PATH = './VIEW'  # 실제 경로로 수정

# 예시:
DATA_PATH = '/home/user/Quant/data/VIEW'  # Linux/macOS
DATA_PATH = 'C:/Users/jsy9k/PycharmProjects/Quant/VIEW'  # Windows
```

2. **데이터 존재 확인**:
```bash
# Windows
dir VIEW\*.csv

# Linux/macOS
ls VIEW/*.csv
```

### 4-2. 데이터 형식 에러

**에러 메시지:**
```
ValueError: could not convert string to float
pandas.errors.ParserError
```

**해결 방법:**

1. CSV 파일 인코딩 확인 (UTF-8이어야 함)
2. 데이터 검증 실행:
```python
from src.storage.data_validator import DataValidator

validator = DataValidator()
results = validator.validate_all_tables()
```

### 4-3. 2024-2025년 데이터 사용 문제 (해결됨)

#### 문제

2024 Q1부터 2025 Q3까지 9개 분기에서 27개 컬럼에 무한대 값이 포함되어 ML 데이터 생성이 실패하거나 XGBoost 학습 에러 발생

**영향받은 컬럼 예시**:
- `OverMC_researchAndDevelopmentExpenses`
- `OverMC_operatingExpenses`
- `OverMC_netIncome`
- 기타 24개 컬럼

#### 원인

VIEW 파일의 원본 메트릭 데이터에 division by zero로 생성된 무한대 값 존재
- 예: `researchAndDevelopmentExpenses / marketCap` 계산 시 marketCap=0인 경우

#### 해결 방법 (자동)

**PR #56에서 자동으로 해결됨** - 별도 조치 불필요

시스템이 자동으로:
1. `fs_table`과 `metrics_table` 로드 직후 무한대 값을 NaN으로 변환
2. 변환된 무한대 값 개수를 로그로 출력
3. 이후 NaN 처리 로직에서 안전하게 제거

**확인 방법**:

로그 파일에서 다음 메시지 확인:
```
INFO: Converted 1,234 infinite values to NaN in fs_table
INFO: Converted 567 infinite values to NaN in metrics_table
```

이 메시지가 보이면 2024-2025 데이터를 정상적으로 사용할 수 있습니다.

### 4-4. Q1 분기 파일 누락 문제 (해결됨)

#### 문제

다음 Q1 분기 파일이 생성되지 않음:
- `rnorm_ml_2015_Q1.parquet`
- `rnorm_ml_2016_Q1.parquet`
- `rnorm_ml_2019_Q1.parquet`
- `rnorm_ml_2020_Q1.parquet`
- `rnorm_ml_2021_Q1.parquet`
- `rnorm_ml_2022_Q1.parquet`

총 6개 분기 (1.5년 치) 데이터 손실

#### 원인

리밸런싱 날짜를 찾을 수 없을 때 파일 저장을 건너뛰는 로직
- Q1의 경우 특정 년도에서 리밸런싱 날짜 데이터 누락
- 과거 로직: 날짜 없으면 해당 분기 전체 스킵

#### 해결 방법 (자동)

**PR #62에서 자동으로 해결됨** - 별도 조치 불필요

시스템이 자동으로:
1. 리밸런싱 날짜가 없을 때 **분기 말일을 기본값**으로 사용
   - Q1: 3월 31일
   - Q2: 6월 30일
   - Q3: 9월 30일
   - Q4: 12월 31일
2. 예측 전용 Parquet 파일 생성 (학습용 라벨 없음)
3. 6개 Q1 분기 추가 확보 → 총 1.5년 치 예측 데이터 확보

**확인 방법**:

```bash
# Windows
dir data\ml_per_year\*_Q1.parquet

# Linux/macOS
ls data/ml_per_year/*_Q1.parquet
```

모든 Q1 분기 파일이 존재하는지 확인:
```
rnorm_ml_2015_Q1.parquet  ✅
rnorm_ml_2016_Q1.parquet  ✅
rnorm_ml_2017_Q1.parquet
rnorm_ml_2018_Q1.parquet
rnorm_ml_2019_Q1.parquet  ✅
rnorm_ml_2020_Q1.parquet  ✅
rnorm_ml_2021_Q1.parquet  ✅
rnorm_ml_2022_Q1.parquet  ✅
...
```

**로그 확인**:
```
INFO: No rebalance_date found for 2015 Q1, using quarter end date: 2015-03-31
INFO: Saved prediction-only file: rnorm_ml_2015_Q1.parquet (no label)
```

---

## 5. 플랫폼별 가이드

### 5-1. Windows

#### PowerShell 실행
```powershell
# 1. 디렉토리 이동
cd C:\Users\jsy9k\PycharmProjects\Quant\Quant-refactoring

# 2. 라이브러리 업데이트
pip install --upgrade pyarrow datasets

# 3. 스크립트 실행
cd scripts
python run_sector_trading.py
```

#### PyCharm 설정
1. **Run** → **Edit Configurations**
2. **+** → **Python**
3. 설정:
   - **Script path**: `...\Quant-refactoring\scripts\run_sector_trading.py`
   - **Working directory**: `...\Quant-refactoring\scripts`

### 5-2. Linux (Ubuntu/Debian)

#### 기본 설정
```bash
# 1. Python 및 pip 설치
sudo apt update
sudo apt install python3 python3-pip python3-venv

# 2. 프로젝트 이동
cd ~/Quant/Quant-refactoring

# 3. 가상 환경 생성 (권장)
python3 -m venv venv
source venv/bin/activate

# 4. 의존성 설치
pip install -r requirements.txt

# 5. 스크립트 실행
cd scripts
python run_sector_trading.py
```

#### Cron 스케줄링 (자동 실행)
```bash
# crontab 편집
crontab -e

# 매일 오전 9시 실행
0 9 * * * cd ~/Quant/Quant-refactoring/scripts && /home/user/Quant/venv/bin/python run_sector_trading.py >> ~/quant_cron.log 2>&1
```

### 5-3. macOS

#### 기본 설정
```bash
# 1. Homebrew 설치 (없는 경우)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Python 설치
brew install python@3.10

# 3. 프로젝트 이동
cd ~/Quant/Quant-refactoring

# 4. 가상 환경 생성
python3 -m venv venv
source venv/bin/activate

# 5. 의존성 설치
pip install -r requirements.txt

# 6. 스크립트 실행
cd scripts
python run_sector_trading.py
```

#### launchd 스케줄링 (자동 실행)
`~/Library/LaunchAgents/com.quant.trading.plist` 생성:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.quant.trading</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/yourname/Quant/venv/bin/python</string>
        <string>/Users/yourname/Quant/Quant-refactoring/scripts/run_sector_trading.py</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>9</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
</dict>
</plist>
```

로드:
```bash
launchctl load ~/Library/LaunchAgents/com.quant.trading.plist
```

---

## 6. 성능 문제

### 6-1. 느린 실행 속도

**해결 방법:**

1. **GPU 사용** (NVIDIA GPU가 있는 경우):
```yaml
# config/conf.yaml
ML:
  USE_GPU: Y
```

2. **멀티프로세싱 활성화**:
```yaml
ML:
  N_JOBS: -1  # 모든 CPU 코어 사용
```

3. **데이터 캐싱**:
```yaml
DATA:
  USE_CACHE: Y
```

### 6-2. 높은 메모리 사용

**해결 방법:**

1. **데이터 청크 처리**:
```python
# 큰 데이터를 청크로 나누어 처리
CHUNK_SIZE = 10000
for chunk in pd.read_csv('data.csv', chunksize=CHUNK_SIZE):
    process(chunk)
```

2. **Feature Selection**:
```python
from feature_engineering.feature_selector import FeatureSelector

selector = FeatureSelector(method='importance', threshold=0.01)
X_selected = selector.fit_transform(X, y)
```

---

## 7. 빠른 참조

### 7-1. 문제별 해결 방법 요약

| 문제 | 해결 |
|------|------|
| PyArrow 에러 | `pip install --upgrade pyarrow>=14.0.0` |
| Module 없음 | `pip install -r requirements.txt` |
| XGBoost gpu_id 에러 | `'gpu_id': 0` → `'device': 'cuda:0'` |
| XGBoost gpu_hist 에러 | `'tree_method': 'gpu_hist'` → `'tree_method': 'hist'` |
| XGBoost infinite 에러 | 자동 해결됨 (PR #60, #62), 로그 확인 |
| API 키 에러 | `config/conf.yaml`에서 `API_KEY` 확인 |
| 파일 없음 | 데이터 경로 확인 (`DATA_PATH`) |
| 메모리 부족 | 배치 크기 줄이기, 데이터 필터링 |
| GPU 에러 | CPU 모드로 전환 (`device: cpu`) |
| Permission 에러 | `pip install --user` 또는 관리자 권한 |

### 7-2. 플랫폼별 명령어 비교

| 작업 | Windows | Linux/macOS |
|------|---------|-------------|
| Python 실행 | `python` | `python3` |
| pip 실행 | `pip` | `pip3` 또는 `pip` |
| 가상환경 활성화 | `venv\Scripts\activate` | `source venv/bin/activate` |
| 경로 구분자 | `\` | `/` |
| 환경변수 | `$env:VAR` (PS) | `$VAR` |

### 7-3. 빠른 테스트 (1분)

**모든 플랫폼:**
```bash
# 1. 라이브러리 업데이트
pip install --upgrade pyarrow datasets

# 2. 샘플 데이터로 테스트
cd examples
python comprehensive_example.py

# 3. 실제 실행
cd ../scripts
python run_sector_trading.py
```

### 7-4. 로그 확인

**모든 플랫폼:**
```bash
# 로그 파일 위치
logs/quant_trading.log
logs/sector_trading.log

# 로그 확인 (Windows)
type logs\sector_trading.log

# 로그 확인 (Linux/macOS)
cat logs/sector_trading.log
tail -f logs/sector_trading.log  # 실시간
```

---

## 📞 추가 지원

문제가 해결되지 않으면 다음 정보와 함께 문의하세요:

1. **에러 메시지 전체**
2. **Python 버전**: `python --version`
3. **OS 정보**:
   - Windows: `systeminfo | findstr /C:"OS Name"`
   - Linux: `uname -a`
   - macOS: `sw_vers`
4. **실행한 명령어**
5. **로그 파일 내용**: `logs/*.log`

**관련 문서:**
- [QUICK_START.md](QUICK_START.md) - 빠른 시작 가이드
- [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md) - 시스템 아키텍처
- [API_REFERENCE.md](API_REFERENCE.md) - API 문서

---

**문서 버전**: 1.0
**마지막 업데이트**: 2025-11-17
**작성자**: Claude AI + Development Team
