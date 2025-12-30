"""
Google Sheets 실험 추적기

백테스트 결과를 구글 시트에 자동 기록하고 Config 파일을 Drive에 업로드
"""

import os
import subprocess
import yaml
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

# Google API 라이브러리
try:
    import gspread
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from googleapiclient.errors import HttpError
    import requests
    GOOGLE_LIBS_AVAILABLE = True
except ImportError:
    GOOGLE_LIBS_AVAILABLE = False

from src.experiment.config_masker import ConfigMasker

logger = logging.getLogger(__name__)


class SheetsTracker:
    """실험 결과를 구글 시트/드라이브에 자동 기록하는 클래스"""

    # Google Sheets API 스코프
    SCOPES = [
        'https://www.googleapis.com/auth/spreadsheets',  # Sheets 읽기/쓰기
        'https://www.googleapis.com/auth/drive.file',    # Drive 파일 업로드
    ]

    def __init__(self, config: Dict[str, Any]):
        """
        초기화

        Args:
            config: 전체 config dict (EXPERIMENT_TRACKING 섹션 포함)
        """
        self.config = config
        self.tracking_config = config.get('EXPERIMENT_TRACKING', {})

        # Config 마스커 초기화
        mask_keys = self.tracking_config.get('MASK_KEYS', ['API_KEY'])
        self.masker = ConfigMasker(mask_keys)

        # Google API 클라이언트 (lazy initialization)
        self.sheets_client = None
        self.drive_client = None

        logger.debug("SheetsTracker initialized")

    def upload_experiment(
        self,
        backtest_results: Optional[Dict[str, Any]] = None,
        prediction_metrics: Optional[Dict[str, Any]] = None,
        experiment_name: Optional[str] = None
    ) -> None:
        """
        실험 결과를 구글 시트/드라이브에 업로드

        ⚠️ 중요: 이 메서드는 절대 예외를 throw하지 않음!
        에러 발생 시 경고 로그만 출력하고 정상 종료

        Args:
            backtest_results: 백테스트 결과 dict
            prediction_metrics: 예측 품질 지표 dict
            experiment_name: 실험 이름 (None이면 자동 생성)
        """
        try:
            # 1. 업로드 여부 체크
            should_upload, reason = self._should_upload(backtest_results)
            if not should_upload:
                logger.info(f"⏭️  Skipping sheet upload: {reason}")
                return

            # 2. Google API 라이브러리 확인
            if not GOOGLE_LIBS_AVAILABLE:
                logger.warning("━" * 60)
                logger.warning("⚠️  EXPERIMENT TRACKING ERROR: Google API libraries not installed")
                logger.warning("    Install with: pip install gspread google-auth google-auth-httplib2")
                logger.warning("    → Continuing without experiment tracking...")
                logger.warning("━" * 60)
                return

            # 3. Git 정보 수집
            git_user, git_email, git_commit, git_branch = self._get_git_info()

            # 4. 실험 이름 생성
            if experiment_name is None:
                experiment_name = self._generate_experiment_name()

            logger.info("━" * 60)
            logger.info("📊 Uploading experiment to Google Sheets...")
            logger.info(f"   Experiment: {experiment_name}")
            logger.info(f"   User: {git_user} ({git_email})")
            logger.info(f"   Commit: {git_commit}")
            logger.info(f"   Branch: {git_branch}")
            logger.info("━" * 60)

            # 5. Config 마스킹 & 변경사항 추출
            masked_config = self.masker.mask_config(self.config)
            changes = self._extract_config_changes()
            changes_str = self.masker.format_changes_for_display(changes)

            # 6. Config 파일 Drive에 업로드
            config_url = self._upload_config_to_drive(
                masked_config,
                experiment_name
            )

            # 7. 시트에 결과 기록
            self._append_to_sheet(
                experiment_name=experiment_name,
                git_user=git_user,
                git_commit=git_commit,
                git_branch=git_branch,
                config_url=config_url,
                changes_str=changes_str,
                backtest_results=backtest_results,
                prediction_metrics=prediction_metrics
            )

            logger.info("✅ Experiment uploaded successfully!")
            logger.info("━" * 60)

        except FileNotFoundError as e:
            # Service account key 파일 없음
            logger.warning("━" * 60)
            logger.warning("⚠️  EXPERIMENT TRACKING ERROR: Service account key not found")
            logger.warning(f"    File: {e.filename}")
            key_path = self.tracking_config.get('GOOGLE_SHEETS', {}).get('KEY_PATH', 'N/A')
            logger.warning(f"    Expected path: {key_path}")
            logger.warning("    → Please download the key file and place it in the correct location")
            logger.warning("    → Continuing without experiment tracking...")
            logger.warning("━" * 60)

        except gspread.exceptions.APIError as e:
            # Google API 에러 (권한, quota, 시트 ID 오류 등)
            logger.warning("━" * 60)
            logger.warning("⚠️  EXPERIMENT TRACKING ERROR: Google Sheets API error")
            logger.warning(f"    Error: {e}")
            logger.warning("    Possible causes:")
            logger.warning("    - No permission to access the sheet")
            logger.warning("    - Invalid SHEET_ID or SHEET_NAME in config")
            logger.warning("    - API quota exceeded")
            logger.warning("    → Check your configuration and permissions")
            logger.warning("    → Continuing without experiment tracking...")
            logger.warning("━" * 60)

        except requests.exceptions.ConnectionError as e:
            # 인터넷 연결 끊김
            logger.warning("━" * 60)
            logger.warning("⚠️  EXPERIMENT TRACKING ERROR: Network connection failed")
            logger.warning(f"    Error: {str(e)[:100]}...")
            logger.warning("    → Check your internet connection")
            logger.warning("    → Continuing without experiment tracking...")
            logger.warning("━" * 60)

        except requests.exceptions.Timeout as e:
            # 네트워크 타임아웃
            logger.warning("━" * 60)
            logger.warning("⚠️  EXPERIMENT TRACKING ERROR: Network timeout")
            logger.warning(f"    Error: {e}")
            logger.warning("    → Google API is slow or unreachable")
            logger.warning("    → Continuing without experiment tracking...")
            logger.warning("━" * 60)

        except PermissionError as e:
            # Drive/Sheet 접근 권한 없음
            logger.warning("━" * 60)
            logger.warning("⚠️  EXPERIMENT TRACKING ERROR: Permission denied")
            logger.warning(f"    Error: {e}")
            logger.warning("    → Share the sheet/folder with service account email")
            logger.warning("    → Continuing without experiment tracking...")
            logger.warning("━" * 60)

        except Exception as e:
            # 예상 못한 에러
            logger.warning("━" * 60)
            logger.warning("⚠️  EXPERIMENT TRACKING ERROR: Unexpected error")
            logger.warning(f"    Error type: {type(e).__name__}")
            logger.warning(f"    Error message: {str(e)[:200]}")
            logger.warning("    → Please report this issue")
            logger.warning("    → Continuing without experiment tracking...")
            logger.warning("━" * 60)

        # 모든 경우에 정상 종료 (예외 throw 안 함)

    def _should_upload(self, backtest_results: Optional[Dict]) -> Tuple[bool, str]:
        """
        업로드 여부 결정

        Returns:
            (업로드 여부, 이유)
        """
        # 1. ENABLED 체크
        if not self.tracking_config.get('ENABLED', False):
            return False, "EXPERIMENT_TRACKING.ENABLED=False"

        # 2. 백테스트 결과 체크
        upload_conditions = self.tracking_config.get('UPLOAD_CONDITIONS', {})
        require_backtest = upload_conditions.get('REQUIRE_BACKTEST_RESULTS', True)

        if require_backtest and backtest_results is None:
            return False, "No backtest results (백테스트 미실행)"

        # 3. Branch 체크
        only_master = upload_conditions.get('ONLY_MASTER_BRANCH', True)
        if only_master:
            _, _, _, current_branch = self._get_git_info()
            if current_branch != "master":
                return False, f"Not on master branch: {current_branch}"

        return True, "OK"

    def _get_git_info(self) -> Tuple[str, str, str, str]:
        """
        Git 정보 수집

        Returns:
            (git_user, git_email, git_commit, git_branch)
        """
        try:
            git_user = subprocess.check_output(
                ["git", "config", "user.name"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
        except:
            git_user = "unknown"

        try:
            git_email = subprocess.check_output(
                ["git", "config", "user.email"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
        except:
            git_email = "unknown"

        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
        except:
            git_commit = "unknown"

        try:
            git_branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
        except:
            git_branch = "unknown"

        return git_user, git_email, git_commit, git_branch

    def _generate_experiment_name(self) -> str:
        """
        실험 이름 자동 생성

        Returns:
            실험 이름 (예: "exp_20251230_143000")
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"exp_{timestamp}"

    def _extract_config_changes(self) -> Dict[str, tuple]:
        """
        Template config와 현재 config를 비교하여 변경사항 추출

        Returns:
            변경사항 dict
        """
        try:
            # Template config 로드
            root_path = self.config.get('DATA', {}).get('ROOT_PATH', './data')
            project_root = Path(root_path).parent  # data의 상위 폴더 = 프로젝트 루트
            template_path = project_root / "config" / "conf.yaml.template"

            if not template_path.exists():
                logger.warning(f"Template config not found: {template_path}")
                return {}

            with open(template_path, 'r', encoding='utf-8') as f:
                template_config = yaml.safe_load(f)

            # 변경사항 추출
            changes = self.masker.extract_changes(template_config, self.config)
            return changes

        except Exception as e:
            logger.warning(f"Failed to extract config changes: {e}")
            return {}

    def _get_credentials(self) -> Credentials:
        """
        Service Account credentials 로드

        Returns:
            Google API credentials

        Raises:
            FileNotFoundError: Key 파일이 없으면
        """
        key_path = self.tracking_config.get('GOOGLE_SHEETS', {}).get('KEY_PATH')
        if not key_path:
            raise ValueError("GOOGLE_SHEETS.KEY_PATH not configured")

        # ~ 확장
        key_path = os.path.expanduser(key_path)

        if not os.path.exists(key_path):
            raise FileNotFoundError(
                f"Service account key file not found: {key_path}"
            )

        creds = Credentials.from_service_account_file(key_path, scopes=self.SCOPES)
        return creds

    def _get_sheets_client(self) -> gspread.Client:
        """
        Google Sheets 클라이언트 가져오기 (lazy initialization)

        Returns:
            gspread client
        """
        if self.sheets_client is None:
            creds = self._get_credentials()
            self.sheets_client = gspread.authorize(creds)
        return self.sheets_client

    def _get_drive_client(self):
        """
        Google Drive 클라이언트 가져오기 (lazy initialization)

        Returns:
            Google Drive service
        """
        if self.drive_client is None:
            creds = self._get_credentials()
            self.drive_client = build('drive', 'v3', credentials=creds)
        return self.drive_client

    def _upload_config_to_drive(
        self,
        masked_config: Dict[str, Any],
        experiment_name: str
    ) -> str:
        """
        마스킹된 Config를 Google Drive에 업로드

        Args:
            masked_config: 마스킹된 config dict
            experiment_name: 실험 이름

        Returns:
            업로드된 파일의 URL
        """
        # 임시 파일로 저장
        timestamp = datetime.now().strftime("%Y-%m-%d")
        prefix = self.tracking_config.get('GOOGLE_DRIVE', {}).get('CONFIG_PREFIX', 'config_')
        filename = f"{prefix}{timestamp}_{experiment_name}.yaml"

        temp_dir = Path("/tmp/quant_experiment")
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / filename

        # YAML로 저장
        with open(temp_path, 'w', encoding='utf-8') as f:
            yaml.dump(masked_config, f, default_flow_style=False, allow_unicode=True)

        logger.debug(f"Config saved to temp file: {temp_path}")

        # Drive에 업로드
        folder_id = self.tracking_config.get('GOOGLE_DRIVE', {}).get('FOLDER_ID')
        if not folder_id:
            raise ValueError("GOOGLE_DRIVE.FOLDER_ID not configured")

        drive_service = self._get_drive_client()

        file_metadata = {
            'name': filename,
            'parents': [folder_id]
        }
        media = MediaFileUpload(str(temp_path), mimetype='text/yaml')

        file = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id,webViewLink'
        ).execute()

        file_id = file.get('id')
        file_url = file.get('webViewLink')

        logger.debug(f"Config uploaded to Drive: {file_url}")

        # 임시 파일 삭제
        temp_path.unlink()

        return file_url

    def _append_to_sheet(
        self,
        experiment_name: str,
        git_user: str,
        git_commit: str,
        git_branch: str,
        config_url: str,
        changes_str: str,
        backtest_results: Optional[Dict],
        prediction_metrics: Optional[Dict]
    ) -> None:
        """
        구글 시트에 실험 결과 추가

        Args:
            experiment_name: 실험 이름
            git_user: Git 사용자 이름
            git_commit: Git commit hash
            git_branch: Git branch 이름
            config_url: Drive에 업로드된 config URL
            changes_str: 변경사항 요약 문자열
            backtest_results: 백테스트 결과
            prediction_metrics: 예측 품질 지표
        """
        # Sheets 클라이언트
        client = self._get_sheets_client()

        # 시트 열기
        sheet_id = self.tracking_config.get('GOOGLE_SHEETS', {}).get('SHEET_ID')
        sheet_name = self.tracking_config.get('GOOGLE_SHEETS', {}).get('SHEET_NAME', 'Experiments')

        if not sheet_id:
            raise ValueError("GOOGLE_SHEETS.SHEET_ID not configured")

        spreadsheet = client.open_by_key(sheet_id)
        worksheet = spreadsheet.worksheet(sheet_name)

        # 현재 날짜/시간
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 백테스트 결과 추출
        total_return = backtest_results.get('total_return', 0) if backtest_results else 0
        sharpe_ratio = backtest_results.get('sharpe_ratio', 0) if backtest_results else 0
        max_drawdown = backtest_results.get('max_drawdown', 0) if backtest_results else 0
        win_rate = backtest_results.get('win_rate', 0) if backtest_results else 0

        # 예측 품질 지표 추출
        pred_rmse = prediction_metrics.get('rmse', 0) if prediction_metrics else 0
        pred_mae = prediction_metrics.get('mae', 0) if prediction_metrics else 0
        pred_r2 = prediction_metrics.get('r2', 0) if prediction_metrics else 0
        pred_accuracy = prediction_metrics.get('accuracy', 0) if prediction_metrics else 0
        pred_precision = prediction_metrics.get('precision', 0) if prediction_metrics else 0
        pred_recall = prediction_metrics.get('recall', 0) if prediction_metrics else 0
        train_time = prediction_metrics.get('train_time_hours', 0) if prediction_metrics else 0

        # 상태 결정
        if sharpe_ratio > 1.0:
            status = "✅"
        elif sharpe_ratio < 0:
            status = "❌"
        else:
            status = "🔍"

        # 행 데이터 구성
        row = [
            now,                # Date
            experiment_name,    # Name
            git_user,           # Git User
            git_commit,         # Git Commit
            git_branch,         # Branch
            f'=HYPERLINK("{config_url}", "📄")',  # Config File (링크)
            changes_str,        # Changes
            f"{total_return:.2f}%",      # Total Return
            f"{sharpe_ratio:.2f}",       # Sharpe Ratio
            f"{max_drawdown:.2f}%",      # Max Drawdown
            f"{win_rate:.1f}%",          # Win Rate
            f"{pred_rmse:.4f}",          # Pred RMSE
            f"{pred_mae:.4f}",           # Pred MAE
            f"{pred_r2:.4f}",            # Pred R²
            f"{pred_accuracy:.1f}%",     # Pred Accuracy
            f"{pred_precision:.1f}%",    # Pred Precision
            f"{pred_recall:.1f}%",       # Pred Recall
            f"{train_time:.1f}h",        # Train Time
            status,             # Status
            "",                 # Notes (empty)
        ]

        # 시트에 추가
        worksheet.append_row(row, value_input_option='USER_ENTERED')
        logger.debug(f"Row appended to sheet: {experiment_name}")


# 편의 함수
def upload_experiment_result(
    config: Dict[str, Any],
    backtest_results: Optional[Dict[str, Any]] = None,
    prediction_metrics: Optional[Dict[str, Any]] = None,
    experiment_name: Optional[str] = None
) -> None:
    """
    실험 결과를 구글 시트/드라이브에 업로드 (편의 함수)

    이 함수는 절대 예외를 throw하지 않습니다.
    에러 발생 시 경고 로그만 출력하고 정상 종료합니다.

    Args:
        config: 전체 config dict
        backtest_results: 백테스트 결과 (None이면 업로드 스킵)
        prediction_metrics: 예측 품질 지표
        experiment_name: 실험 이름 (None이면 자동 생성)

    Example:
        >>> # ml_backtest.py에서 호출
        >>> backtest_results = {
        ...     "total_return": 163.81,
        ...     "sharpe_ratio": 1.24,
        ...     "max_drawdown": -16.86,
        ...     "win_rate": 81.8
        ... }
        >>> prediction_metrics = {
        ...     "rmse": 0.123,
        ...     "mae": 0.089,
        ...     "r2": 0.456,
        ...     "accuracy": 61.2,
        ...     "precision": 72.9,
        ...     "recall": 54.3,
        ...     "train_time_hours": 6.5
        ... }
        >>> upload_experiment_result(config, backtest_results, prediction_metrics)
    """
    tracker = SheetsTracker(config)
    tracker.upload_experiment(backtest_results, prediction_metrics, experiment_name)
