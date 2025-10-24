# 🔧 에러 해결 가이드

## 문제: PyArrow 호환성 에러

### 에러 메시지
```
AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'
```

---

## 해결 방법

### Option 1: PyArrow 및 Datasets 업그레이드 (권장)

```bash
# Windows PowerShell 또는 CMD
pip install --upgrade pyarrow datasets

# 또는 특정 버전 설치
pip install pyarrow>=14.0.0
pip install datasets>=2.14.0
```

### Option 2: 전체 requirements 재설치

```bash
pip install -r requirements_fix.txt
```

---

## 새로운 스크립트 실행 방법

### ⚠️ 중요: 기존 main.py가 아닌 새 스크립트 사용

우리가 만든 새로운 시스템은 `scripts` 디렉토리에 있습니다!

```bash
# 잘못된 방법 (기존)
python main.py  # ❌ 이것은 기존 코드

# 올바른 방법 (새로운 시스템)
cd scripts
python run_sector_trading.py  # ✅ 섹터별 주식 선택
```

---

## 빠른 실행 가이드

### Windows PowerShell

```powershell
# 1. 디렉토리 이동
cd C:\Users\jsy9k\PycharmProjects\Quant\Quant-refactoring

# 2. 라이브러리 업데이트
pip install --upgrade pyarrow datasets

# 3. 새로운 스크립트 실행
cd scripts
python run_sector_trading.py
```

### Windows CMD

```cmd
:: 1. 디렉토리 이동
cd C:\Users\jsy9k\PycharmProjects\Quant\Quant-refactoring

:: 2. 라이브러리 업데이트
pip install --upgrade pyarrow datasets

:: 3. 새로운 스크립트 실행
cd scripts
python run_sector_trading.py
```

---

## 실행할 스크립트 선택

### 1️⃣ 전체 파이프라인 (권장 - 처음 실행)

```bash
cd scripts
python run_full_pipeline.py
```

이 스크립트는:
- 리밸런싱 기간 최적화 (선택)
- 모델 성능 비교 (선택)
- 섹터별 주식 선택 (필수)

모두 자동으로 실행합니다.

### 2️⃣ 섹터별 주식 선택만 (매 리밸런싱)

```bash
cd scripts
python run_sector_trading.py
```

---

## 데이터 경로 문제 해결

### 에러: "Price 파일을 찾을 수 없습니다"

스크립트에서 데이터 경로를 수정하세요:

**scripts/run_sector_trading.py** 파일 열기:

```python
# 25번째 줄 근처
DATA_PATH = './VIEW'  # 이 부분을 실제 데이터 경로로 수정

# 예시:
# DATA_PATH = 'C:/Users/jsy9k/PycharmProjects/Quant/VIEW'
# 또는
# DATA_PATH = '../VIEW'  # scripts 폴더에서 실행하는 경우
```

---

## PyCharm에서 실행하기

### 방법 1: 터미널 사용

1. PyCharm 하단의 **Terminal** 탭 클릭
2. 다음 명령어 입력:

```bash
cd scripts
python run_sector_trading.py
```

### 방법 2: Run Configuration 설정

1. PyCharm 메뉴: **Run** → **Edit Configurations**
2. **+** 버튼 클릭 → **Python** 선택
3. 설정:
   - **Name**: `Sector Trading`
   - **Script path**: `C:\Users\jsy9k\PycharmProjects\Quant\Quant-refactoring\scripts\run_sector_trading.py`
   - **Working directory**: `C:\Users\jsy9k\PycharmProjects\Quant\Quant-refactoring\scripts`
4. **OK** 클릭
5. 실행 버튼 (▶️) 클릭

---

## 테스트 실행 (데이터 없이)

실제 데이터가 없는 경우 샘플 데이터로 테스트:

```bash
cd examples
python comprehensive_example.py
```

이 스크립트는 샘플 데이터를 자동으로 생성하여 모든 기능을 테스트합니다.

---

## 전체 해결 순서 (Windows)

```powershell
# 1. 디렉토리로 이동
cd C:\Users\jsy9k\PycharmProjects\Quant\Quant-refactoring

# 2. 라이브러리 문제 해결
pip install --upgrade pyarrow>=14.0.0 datasets>=2.14.0

# 3. 필요한 디렉토리 생성
mkdir logs
mkdir results
mkdir config

# 4. 테스트 실행 (샘플 데이터)
cd examples
python comprehensive_example.py

# 5. 실제 실행 (실제 데이터)
cd ..\scripts
python run_sector_trading.py
```

---

## 여전히 에러가 발생하면

### 1. Python 버전 확인

```bash
python --version
```

Python 3.8 이상이어야 합니다. (권장: 3.10)

### 2. 가상 환경 생성 (깨끗한 환경)

```bash
# 가상 환경 생성
python -m venv venv_quant

# 활성화 (Windows)
venv_quant\Scripts\activate

# 패키지 설치
pip install -r requirements_fix.txt

# 스크립트 실행
cd scripts
python run_sector_trading.py
```

### 3. 특정 에러 메시지 확인

로그 파일을 확인하세요:
```bash
type logs\sector_trading.log
```

---

## 빠른 테스트 (1분 안에)

```bash
# 1. 라이브러리 업데이트
pip install --upgrade pyarrow datasets

# 2. 샘플 데이터로 테스트
cd examples
python comprehensive_example.py

# 성공하면 ✅
# 실제 데이터로 실행:
cd ..\scripts
python run_sector_trading.py
```

---

## 문제별 해결 방법

| 문제 | 해결 |
|------|------|
| PyArrow 에러 | `pip install --upgrade pyarrow>=14.0.0` |
| Datasets 에러 | `pip install --upgrade datasets>=2.14.0` |
| 파일 없음 에러 | 데이터 경로 확인 (`DATA_PATH` 수정) |
| 모듈 없음 에러 | `pip install [모듈명]` |
| 권한 에러 | 관리자 권한으로 CMD/PowerShell 실행 |

---

## 연락처

추가 문제가 발생하면 다음 정보와 함께 문의하세요:
1. 에러 메시지 전체
2. Python 버전 (`python --version`)
3. 실행한 명령어
4. 로그 파일 내용 (`logs/*.log`)
