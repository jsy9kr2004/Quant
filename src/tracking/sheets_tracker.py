"""
Google Sheets 실험 추적기

백테스트 결과를 구글 시트에 자동 기록하고 Config 파일을 GitHub Gist에 업로드
"""

import os
import subprocess
import yaml
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List, TYPE_CHECKING
from pathlib import Path

# Google API 라이브러리 (타입 힌트용)
if TYPE_CHECKING:
    import gspread
    from google.oauth2.service_account import Credentials

# Google API 라이브러리 (런타임)
try:
    import gspread
    from google.oauth2.service_account import Credentials
    import requests
    GOOGLE_LIBS_AVAILABLE = True
except ImportError:
    gspread = None  # type: ignore
    Credentials = None  # type: ignore
    requests = None  # type: ignore
    GOOGLE_LIBS_AVAILABLE = False
    # 타입 힌트용 더미 클래스 (실제로 사용되지 않음)
    Credentials = None
    gspread = None

from src.tracking.config_masker import ConfigMasker
from src.infra.context_loader import ConfigLoader, ContextLoader

logger = logging.getLogger(__name__)


class SheetsTracker:
    """실험 결과를 구글 시트에 기록하고 Config를 GitHub Gist에 업로드하는 클래스"""

    # Google Sheets API 스코프
    SCOPES = [
        'https://www.googleapis.com/auth/spreadsheets',  # Sheets 읽기/쓰기
    ]

    # GitHub Gist API URL
    GITHUB_GIST_API = "https://api.github.com/gists"

    @staticmethod
    def _log_tracking_error(
        error_message: str,
        error: Exception,
        suggestions: List[str],
    ) -> None:
        """실험 추적 에러를 일관된 형식으로 로깅합니다.

        Args:
            error_message: 에러 요약 메시지
            error: 발생한 예외 객체
            suggestions: 사용자에게 보여줄 해결 방법 리스트
        """
        logger.warning("━" * 60)
        logger.warning(f"⚠️  EXPERIMENT TRACKING ERROR: {error_message}")
        if hasattr(error, 'filename'):
            logger.warning(f"    File: {error.filename}")
        else:
            error_str = str(error)
            if len(error_str) > 200:
                error_str = error_str[:200] + "..."
            logger.warning(f"    Error: {error_str}")
        for suggestion in suggestions:
            logger.warning(f"    → {suggestion}")
        logger.warning("    → Continuing without experiment tracking...")
        logger.warning("━" * 60)

    def __init__(self, config: Dict[str, Any]):
        """
        초기화

        Args:
            config: 전체 config dict (EXPERIMENT_TRACKING 섹션 포함)
        """
        self.config = config
        self.tracking_config = config.get('EXPERIMENT_TRACKING', {})

        # Config 마스커 초기화
        # 1. conf.yaml의 MASK_KEYS에서 명시적으로 지정한 키
        mask_keys = list(self.tracking_config.get('MASK_KEYS', ['API_KEY']))

        # 2. secrets.yaml에서 관리되는 모든 키도 자동으로 마스킹 대상에 추가
        #    → 추후 secrets.yaml에 새 키가 추가되어도 자동 보호
        for secret_key, config_path in ConfigLoader.SECRETS_KEY_MAPPING.items():
            # secret_key 자체 (예: FMP_API_KEY, GITHUB_PAT)
            if secret_key not in mask_keys:
                mask_keys.append(secret_key)
            # config 경로의 leaf 키 (예: API_KEY, PAT, SHEET_ID)
            leaf_key = config_path[-1]
            if leaf_key not in mask_keys:
                mask_keys.append(leaf_key)

        self.masker = ConfigMasker(mask_keys)

        # Google API 클라이언트 (lazy initialization)
        self.sheets_client = None

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

            # 6. Config 파일을 GitHub Gist에 업로드
            config_url = self._upload_config_to_gist(
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
            key_path = self.tracking_config.get('GOOGLE_SHEETS', {}).get('KEY_PATH', 'N/A')
            self._log_tracking_error(
                "Service account key not found", e,
                [f"Expected path: {key_path}",
                 "Please download the key file and place it in the correct location"],
            )

        except gspread.exceptions.APIError as e:
            self._log_tracking_error(
                "Google Sheets API error", e,
                ["Check your configuration and permissions",
                 "Possible: no permission, invalid SHEET_ID/SHEET_NAME, or API quota exceeded"],
            )

        except requests.exceptions.ConnectionError as e:
            self._log_tracking_error(
                "Network connection failed", e,
                ["Check your internet connection"],
            )

        except requests.exceptions.Timeout as e:
            self._log_tracking_error(
                "Network timeout", e,
                ["Google API is slow or unreachable"],
            )

        except PermissionError as e:
            self._log_tracking_error(
                "Permission denied", e,
                ["Share the sheet/folder with service account email"],
            )

        except Exception as e:
            self._log_tracking_error(
                f"Unexpected error ({type(e).__name__})", e,
                ["Please report this issue"],
            )

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

    @staticmethod
    def _run_git_command(args: list, default: str = "unknown") -> str:
        """Git 명령어를 안전하게 실행하고 결과를 반환합니다.

        Args:
            args: git 서브커맨드 인자 리스트 (예: ["config", "user.name"])
            default: 실패 시 반환할 기본값

        Returns:
            명령어 출력 문자열 또는 default 값
        """
        try:
            return subprocess.check_output(
                ["git"] + args,
                stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            return default

    def _get_git_info(self) -> Tuple[str, str, str, str]:
        """
        Git 정보 수집

        Returns:
            (git_user, git_email, git_commit, git_branch)
        """
        git_user = self._run_git_command(["config", "user.name"])
        git_email = self._run_git_command(["config", "user.email"])
        git_commit = self._run_git_command(["rev-parse", "--short", "HEAD"])
        git_branch = self._run_git_command(["rev-parse", "--abbrev-ref", "HEAD"])

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

    def _get_sheets_client(self) -> "gspread.Client":
        """
        Google Sheets 클라이언트 가져오기 (lazy initialization)

        Returns:
            gspread client
        """
        if self.sheets_client is None:
            creds = self._get_credentials()
            self.sheets_client = gspread.authorize(creds)
        return self.sheets_client

    def _upload_config_to_gist(
        self,
        masked_config: Dict[str, Any],
        experiment_name: str
    ) -> str:
        """
        마스킹된 Config를 GitHub Gist에 업로드

        Args:
            masked_config: 마스킹된 config dict
            experiment_name: 실험 이름

        Returns:
            업로드된 Gist의 URL (실패 시 빈 문자열)
        """
        # GitHub PAT 확인
        gist_config = self.tracking_config.get('GITHUB_GIST', {})
        github_pat = gist_config.get('PAT')

        if not github_pat:
            logger.warning("━" * 60)
            logger.warning("⚠️  GITHUB_GIST.PAT not configured - skipping config upload")
            logger.warning("    Possible causes:")
            logger.warning("    1. secrets.yaml missing or not loaded")
            logger.warning("    2. GITHUB_PAT in secrets.yaml is placeholder (YOUR_..._HERE)")
            logger.warning("    3. GITHUB_PAT key not present in secrets.yaml")
            logger.warning(f"    GITHUB_GIST config keys: {list(gist_config.keys())}")
            logger.warning("    → Set GITHUB_PAT in config/secrets.yaml")
            logger.warning("━" * 60)
            return ""

        # 파일 이름 생성
        timestamp = datetime.now().strftime("%Y-%m-%d")
        prefix = gist_config.get('CONFIG_PREFIX', 'config_')
        filename = f"{prefix}{timestamp}_{experiment_name}.yaml"

        # YAML 문자열로 변환
        yaml_content = yaml.dump(masked_config, default_flow_style=False, allow_unicode=True)

        # Gist 생성 요청
        headers = {
            'Authorization': f'token {github_pat}',
            'Accept': 'application/vnd.github.v3+json',
        }

        # public=False로 설정하면 secret gist (URL 아는 사람만 접근 가능)
        is_public = gist_config.get('PUBLIC', False)
        description = f"Quant Experiment Config: {experiment_name}"

        payload = {
            'description': description,
            'public': is_public,
            'files': {
                filename: {
                    'content': yaml_content
                }
            }
        }

        try:
            response = requests.post(
                self.GITHUB_GIST_API,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            gist_data = response.json()
            gist_url = gist_data.get('html_url', '')

            logger.debug(f"Config uploaded to Gist: {gist_url}")
            return gist_url

        except requests.exceptions.RequestException as e:
            # 상세 진단 로그
            pat_len = len(github_pat) if github_pat else 0
            pat_prefix = github_pat[:4] + "..." if pat_len > 4 else "***"
            status_code = getattr(getattr(e, 'response', None), 'status_code', 'N/A')
            response_text = ""
            if hasattr(e, 'response') and e.response is not None:
                response_text = e.response.text[:200] if e.response.text else ""

            logger.warning("━" * 60)
            logger.warning(f"⚠️  Failed to upload config to Gist")
            logger.warning(f"    HTTP Status: {status_code}")
            logger.warning(f"    PAT length: {pat_len}, prefix: {pat_prefix}")
            logger.warning(f"    Response: {response_text}")
            if status_code == 401:
                logger.warning("    → PAT may be expired or invalid")
                logger.warning("    → Check: https://github.com/settings/tokens")
                logger.warning("    → Required scope: 'gist'")
            elif status_code == 403:
                logger.warning("    → PAT may lack 'gist' scope permission")
            logger.warning("━" * 60)
            return ""

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
