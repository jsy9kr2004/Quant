# 실험 추적 시스템 사용 가이드

## 📋 목차

1. [개요](#개요)
2. [초기 설정](#초기-설정)
3. [사용 방법](#사용-방법)
4. [트러블슈팅](#트러블슈팅)
5. [FAQ](#faq)

---

## 개요

### 목적

팀 협업 환경에서 ML 실험 결과를 중앙 집중식으로 관리하기 위한 시스템

### 주요 기능

- ✅ 구글 시트에 실험 결과 자동 기록 (수익률, Sharpe, 예측도 등)
- ✅ 구글 드라이브에 Config 파일 자동 업로드 (재현성 보장)
- ✅ Template 대비 변경사항만 추출하여 간결하게 표시
- ✅ API_KEY 등 민감 정보 자동 마스킹
- ✅ Git 정보 자동 수집 (commit, branch, user)
- ✅ Master 브랜치만 업로드 (Feature 브랜치는 로컬 테스트)
- ✅ 백테스트 실행 시에만 업로드 (regressor만 돌리면 스킵)
- ✅ 에러 발생 시에도 프로그램 정상 실행 (경고만 출력)

### 시스템 구조

```
실험 실행 (ml_backtest.py)
    ↓
upload_experiment_result() 호출
    ↓
조건 체크 (ENABLED? Master branch? Backtest 결과 있음?)
    ↓
Config 마스킹 (API_KEY → ***MASKED***)
    ↓
변경사항 추출 (Template 대비 diff)
    ↓
┌─────────────────┬──────────────────┐
│ Google Drive    │ Google Sheets    │
│ Config 업로드   │ 결과 기록        │
│ (전체 파라미터) │ (요약 + 링크)    │
└─────────────────┴──────────────────┘
```

---

## 초기 설정

### 1단계: Google Cloud Console 설정

#### 1-1. 프로젝트 생성

1. [Google Cloud Console](https://console.cloud.google.com/) 접속
2. 프로젝트 생성: "Quant Experiment Tracking"
3. 프로젝트 선택

#### 1-2. API 활성화

1. "API 및 서비스" → "라이브러리" 이동
2. 다음 API 검색 후 활성화:
   - **Google Sheets API**
   - **Google Drive API**

#### 1-3. Service Account 생성

1. "API 및 서비스" → "사용자 인증 정보" 이동
2. "사용자 인증 정보 만들기" → "서비스 계정" 선택
3. 서비스 계정 세부정보:
   - 이름: `quant-experiment-bot`
   - ID: `quant-experiment-bot` (자동 생성)
   - 설명: "Quant trading experiment tracker"
4. "만들기 및 계속하기" 클릭
5. 역할 선택: "편집자" (또는 "Sheets 편집자" + "Drive 파일 편집자")
6. "완료" 클릭

#### 1-4. JSON Key 다운로드

1. 생성된 서비스 계정 클릭
2. "키" 탭 → "키 추가" → "새 키 만들기"
3. 키 유형: **JSON** 선택
4. "만들기" → JSON 파일 자동 다운로드
   - 파일명 예: `quant-experiment-tracking-abc123.json`

**⚠️ 중요**: 이 파일은 절대 Git에 커밋하지 마세요!

---

### 2단계: 구글 시트/드라이브 설정

#### 2-1. 구글 시트 생성

1. [Google Sheets](https://sheets.google.com/) 접속
2. 새 스프레드시트 생성: "Quant Experiments"
3. 첫 번째 탭 이름: "Experiments"
4. 헤더 행 작성 (1행):

| A열 | B열 | C열 | D열 | E열 | F열 | G열 | H열 | I열 | J열 | K열 | L열 | M열 | N열 | O열 | P열 | Q열 | R열 | S열 | T열 |
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| Date | Name | Git User | Git Commit | Branch | Config File | Changes | Total Return | Sharpe Ratio | Max Drawdown | Win Rate | Pred RMSE | Pred MAE | Pred R² | Pred Accuracy | Pred Precision | Pred Recall | Train Time | Status | Notes |

5. 시트 ID 복사:
   - URL: `https://docs.google.com/spreadsheets/d/{SHEET_ID}/edit`
   - `SHEET_ID` 부분 복사

#### 2-2. 구글 드라이브 폴더 생성

1. [Google Drive](https://drive.google.com/) 접속
2. 새 폴더 생성: "Quant Configs"
3. 폴더 ID 복사:
   - 폴더 우클릭 → "링크 복사"
   - URL: `https://drive.google.com/drive/folders/{FOLDER_ID}`
   - `FOLDER_ID` 부분 복사

#### 2-3. Service Account에 권한 부여

**중요**: Service Account는 일반 사용자가 아니므로 명시적으로 공유해야 합니다!

1. **시트 공유**:
   - 구글 시트 열기
   - 우측 상단 "공유" 버튼 클릭
   - Service Account 이메일 입력 (JSON 파일에서 확인)
     - 예: `quant-experiment-bot@project-id.iam.gserviceaccount.com`
   - 권한: **편집자**
   - "완료" 클릭

2. **드라이브 폴더 공유**:
   - "Quant Configs" 폴더 우클릭 → "공유"
   - Service Account 이메일 입력
   - 권한: **편집자**
   - "완료" 클릭

---

### 3단계: 로컬 환경 설정

#### 3-1. 라이브러리 설치

```bash
cd /home/user/Quant
pip install -r requirements.txt
```

설치되는 주요 라이브러리:
- `gspread>=6.0.0` - Google Sheets API
- `google-auth>=2.25.0` - Google 인증
- `google-auth-oauthlib>=1.2.0` - OAuth2
- `google-auth-httplib2>=0.2.0` - HTTP 지원

#### 3-2. JSON Key 파일 설정

```bash
# 1. 디렉토리 생성
mkdir -p ~/.credentials

# 2. JSON 파일 복사 (다운로드한 파일)
cp ~/Downloads/quant-experiment-tracking-abc123.json ~/.credentials/quant_sheets.json

# 3. 권한 설정 (보안 강화)
chmod 600 ~/.credentials/quant_sheets.json
```

**팀원들**: Slack이나 이메일로 JSON 파일을 공유받아 동일하게 설정

#### 3-3. Config 파일 설정

`config/conf.yaml` 수정:

```yaml
EXPERIMENT_TRACKING:
  ENABLED: Y  # 활성화!

  UPLOAD_CONDITIONS:
    ONLY_MASTER_BRANCH: Y  # Master만 업로드
    REQUIRE_BACKTEST_RESULTS: Y  # 백테스트 필수

  GOOGLE_SHEETS:
    SHEET_ID: "1abc123..."  # ← 2-1단계에서 복사한 시트 ID
    SHEET_NAME: "Experiments"
    KEY_PATH: "~/.credentials/quant_sheets.json"
    RETRY_ON_FAILURE: 3
    TIMEOUT_SECONDS: 30

  GOOGLE_DRIVE:
    FOLDER_ID: "1xyz789..."  # ← 2-2단계에서 복사한 폴더 ID
    CONFIG_PREFIX: "config_"

  MASK_KEYS:
    - API_KEY
    - FMP_API_KEY
    - DATABASE_PASSWORD
```

---

## 사용 방법

### 자동 업로드 (권장)

백테스트 실행 시 자동으로 구글 시트에 기록됩니다.

```bash
# Master 브랜치에서 실행
git checkout master
python src/backtest/ml_backtest.py
```

**동작**:
1. 백테스트 완료
2. 자동으로 Config 마스킹
3. 변경사항 추출 (Template 대비)
4. Drive에 Config 업로드
5. Sheets에 결과 기록
6. ✅ 완료!

### Feature 브랜치 실험

Feature 브랜치에서는 자동으로 스킵됩니다:

```bash
git checkout feature/test-topk10
python src/backtest/ml_backtest.py

# 출력:
# ⏭️  Skipping sheet upload: Not on master branch: feature/test-topk10
```

실험 완료 후 master에 merge하면 자동 기록:

```bash
git checkout master
git merge feature/test-topk10
python src/backtest/ml_backtest.py

# 출력:
# 📊 Uploading experiment to Google Sheets...
# ✅ Experiment uploaded successfully!
```

### 수동 업로드

코드에서 직접 호출도 가능:

```python
from src.experiment import upload_experiment_result
import yaml

# Config 로드
with open('config/conf.yaml') as f:
    config = yaml.safe_load(f)

# 백테스트 결과
backtest_results = {
    "total_return": 163.81,
    "sharpe_ratio": 1.24,
    "max_drawdown": -16.86,
    "win_rate": 81.8
}

# 예측 품질 지표
prediction_metrics = {
    "rmse": 0.123,
    "mae": 0.089,
    "r2": 0.456,
    "accuracy": 61.2,
    "precision": 72.9,
    "recall": 54.3,
    "train_time_hours": 6.5
}

# 업로드
upload_experiment_result(
    config=config,
    backtest_results=backtest_results,
    prediction_metrics=prediction_metrics,
    experiment_name="custom_experiment_name"  # Optional
)
```

---

## 트러블슈팅

### 에러 1: Service account key not found

```
⚠️  EXPERIMENT TRACKING ERROR: Service account key not found
    File: quant_sheets.json
    Expected path: ~/.credentials/quant_sheets.json
```

**해결**:
```bash
# 파일이 존재하는지 확인
ls -la ~/.credentials/quant_sheets.json

# 없으면 다시 복사
cp ~/Downloads/키파일.json ~/.credentials/quant_sheets.json
chmod 600 ~/.credentials/quant_sheets.json
```

---

### 에러 2: Google Sheets API error (Permission denied)

```
⚠️  EXPERIMENT TRACKING ERROR: Google Sheets API error
    Error: Insufficient Permission
    Possible causes:
    - No permission to access the sheet
```

**해결**:
1. 구글 시트 열기
2. 우측 상단 "공유" 클릭
3. Service Account 이메일 확인:
   ```bash
   # JSON 파일에서 확인
   cat ~/.credentials/quant_sheets.json | grep client_email
   ```
4. 해당 이메일이 **편집자** 권한으로 공유되어 있는지 확인
5. 없으면 추가 → "편집자" 권한 부여

---

### 에러 3: Network connection failed

```
⚠️  EXPERIMENT TRACKING ERROR: Network connection failed
    → Check your internet connection
```

**해결**:
```bash
# 인터넷 연결 확인
ping google.com

# VPN 사용 중이면 잠시 끄기
```

---

### 에러 4: API quota exceeded

```
⚠️  EXPERIMENT TRACKING ERROR: Google Sheets API error
    Error: Quota exceeded for quota metric 'Write requests'
```

**해결**:
- Google Sheets API는 **분당 100회** 쓰기 제한
- 너무 빠르게 여러 실험 실행하면 발생
- 1-2분 후 재시도

---

### 에러 5: Google API libraries not installed

```
⚠️  EXPERIMENT TRACKING ERROR: Google API libraries not installed
    Install with: pip install gspread google-auth google-auth-httplib2
```

**해결**:
```bash
pip install gspread google-auth google-auth-oauthlib google-auth-httplib2
```

---

## FAQ

### Q1: 백테스트 없이 regressor만 돌리면 어떻게 되나요?

**A**: 자동으로 스킵됩니다.

```python
# regressor.py만 실행
python src/training/regressor.py

# 출력:
# ⏭️  Skipping sheet upload: No backtest results (백테스트 미실행)
```

`REQUIRE_BACKTEST_RESULTS: Y` 설정 때문에 백테스트 결과가 없으면 업로드하지 않습니다.

---

### Q2: Feature 브랜치에서 테스트하고 싶은데 시트에 안 올라가요.

**A**: 정상 동작입니다! `ONLY_MASTER_BRANCH: Y` 설정 때문입니다.

**Feature 브랜치 워크플로우**:
1. Feature 브랜치에서 실험 (로컬에만 기록)
2. 결과 확인 → 로그 파일 분석
3. 좋은 결과면 master에 merge
4. Master에서 다시 실행 → 시트에 자동 기록

**모든 브랜치에서 기록하고 싶다면**:
```yaml
UPLOAD_CONDITIONS:
  ONLY_MASTER_BRANCH: N  # 모든 브랜치 업로드
```

---

### Q3: API_KEY가 Drive에 올라가면 어떡하죠?

**A**: 걱정 마세요! 자동으로 마스킹됩니다.

**원본 config** (로컬):
```yaml
DATA:
  API_KEY: "abc123secret"
```

**Drive에 업로드되는 config**:
```yaml
DATA:
  API_KEY: "***MASKED***"
```

마스킹 대상 키는 `MASK_KEYS`에서 설정:
```yaml
MASK_KEYS:
  - API_KEY
  - FMP_API_KEY
  - DATABASE_PASSWORD
  - AWS_SECRET_KEY
```

---

### Q4: 시트에 업로드 실패해도 백테스트는 정상 실행되나요?

**A**: 네! 절대 프로그램이 종료되지 않습니다.

```
[14:30:00] Starting backtest...
[15:45:00] Backtest complete: +163.81%
[15:45:01] 📊 Uploading experiment to Google Sheets...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  EXPERIMENT TRACKING ERROR: Network connection failed
    → Continuing without experiment tracking...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[15:45:02] All results saved to log.txt
[15:45:02] ✅ Program completed successfully!
```

백테스트 결과는 로컬 `log.txt`에 저장되므로 안전합니다.

---

### Q5: 여러 팀원이 동시에 실험하면 충돌나나요?

**A**: 아니요! 구글 시트는 동시 쓰기를 지원합니다.

각 팀원의 실험은 독립적으로 새 행에 추가됩니다:

```
| Date | Name | Git User | ...
|------|------|----------|----
| 12-30 14:30 | exp_A | 팀원A | ...
| 12-30 14:32 | exp_B | 팀원B | ...  ← 동시 실행 OK
| 12-30 14:35 | exp_C | 팀원A | ...
```

---

### Q6: 변경사항이 너무 많으면 "Changes" 컬럼이 너무 길지 않나요?

**A**: 자동으로 간결하게 요약됩니다!

**우선순위 파라미터**만 표시 (최대 5개):
- `BACKTEST.TOPK`
- `BACKTEST.START_MONTH`
- `ML.LOSS_THRESHOLD`
- `ML.USE_SECTOR_MODEL`
- ...

**예시**:
```
Changes: TOPK=10, START_MONTH=3, ...+3 more
```

전체 Config는 Drive 링크 클릭하면 확인 가능합니다.

---

### Q7: 실험 재현은 어떻게 하나요?

**A**: 시트에서 Config 링크 클릭 → 다운로드 → 실행

```bash
# 1. 시트에서 "Config File" 컬럼의 📄 클릭
# 2. Drive에서 config_2025-12-30_baseline.yaml 다운로드
# 3. 로컬 config에 덮어쓰기
cp ~/Downloads/config_2025-12-30_baseline.yaml config/conf.yaml

# 4. API_KEY만 원래대로 복원 (마스킹되어 있음)
vim config/conf.yaml
# DATA:
#   API_KEY: "***MASKED***"  ← 실제 키로 변경

# 5. Git commit으로 코드 버전 맞추기
git checkout {Git Commit}  # 시트의 "Git Commit" 컬럼 값

# 6. 재실행
python src/backtest/ml_backtest.py
```

---

### Q8: 비용이 얼마나 드나요?

**A**: 완전 무료입니다!

- Google Sheets API: **무료** (분당 100회 쓰기)
- Google Drive API: **무료** (15GB 스토리지)
- Service Account: **무료**

Config 파일은 수 KB이므로 수천 개 실험해도 괜찮습니다.

---

## 추가 자료

- [Google Sheets API 문서](https://developers.google.com/sheets/api)
- [Google Drive API 문서](https://developers.google.com/drive/api)
- [gspread 라이브러리 문서](https://docs.gspread.org/)

---

**Last Updated**: 2025-12-30
**Version**: 1.0.0
