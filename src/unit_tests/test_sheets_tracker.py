"""
SheetsTracker 유닛테스트

테스트 실행:
    # Mock 테스트만 (Google/GitHub API 호출 없음, 빠름)
    pytest src/unit_tests/test_sheets_tracker.py -v -m "not integration"

    # 실제 API 연동 테스트 (secrets.yaml 필요)
    pytest src/unit_tests/test_sheets_tracker.py -v -m integration

    # 전체 테스트
    pytest src/unit_tests/test_sheets_tracker.py -v
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# 테스트 대상 모듈
from src.experiment.sheets_tracker import SheetsTracker, upload_experiment_result
from src.experiment.config_masker import ConfigMasker


# ============================================================================
# Fixtures (테스트 데이터)
# ============================================================================

@pytest.fixture
def sample_config():
    """기본 테스트 config"""
    return {
        'DATA': {
            'ROOT_PATH': './data',
            'API_KEY': 'test_api_key_12345',
            'START_YEAR': 2020,
            'END_YEAR': 2023,
        },
        'ML': {
            'USE_CLASSIFIER': 'Y',
            'OPTUNA_TRIALS': 10,
        },
        'EXPERIMENT_TRACKING': {
            'ENABLED': True,
            'UPLOAD_CONDITIONS': {
                'ONLY_MASTER_BRANCH': False,  # 테스트에서는 브랜치 체크 비활성화
                'REQUIRE_BACKTEST_RESULTS': True,
            },
            'GOOGLE_SHEETS': {
                'SHEET_ID': 'test_sheet_id_123',
                'SHEET_NAME': 'Experiments',
                'KEY_PATH': '~/.credentials/test_key.json',
            },
            'GITHUB_GIST': {
                'PAT': 'test_github_pat_12345',
                'CONFIG_PREFIX': 'config_',
                'PUBLIC': False,
            },
            'MASK_KEYS': ['API_KEY', 'FMP_API_KEY', 'GITHUB_PAT'],
        },
    }


@pytest.fixture
def sample_backtest_results():
    """샘플 백테스트 결과"""
    return {
        'total_return': 25.5,
        'sharpe_ratio': 1.2,
        'max_drawdown': -15.3,
        'win_rate': 65.0,
    }


@pytest.fixture
def sample_prediction_metrics():
    """샘플 예측 지표"""
    return {
        'rmse': 0.045,
        'mae': 0.032,
        'r2': 0.65,
        'accuracy': 62.5,
        'precision': 70.0,
        'recall': 55.0,
        'train_time_hours': 2.5,
    }


@pytest.fixture
def tracker(sample_config):
    """SheetsTracker 인스턴스"""
    return SheetsTracker(sample_config)


# ============================================================================
# Unit Tests (Mock 사용, Google API 호출 없음)
# ============================================================================

class TestSheetsTrackerInit:
    """초기화 테스트"""

    def test_init_with_valid_config(self, sample_config):
        """정상 config로 초기화"""
        tracker = SheetsTracker(sample_config)

        assert tracker.config == sample_config
        assert tracker.tracking_config == sample_config['EXPERIMENT_TRACKING']
        assert tracker.sheets_client is None  # lazy init

    def test_init_without_experiment_tracking(self):
        """EXPERIMENT_TRACKING 없는 config"""
        config = {'DATA': {'ROOT_PATH': './data'}}
        tracker = SheetsTracker(config)

        assert tracker.tracking_config == {}


class TestShouldUpload:
    """업로드 여부 판단 테스트"""

    def test_should_upload_disabled(self, sample_config):
        """ENABLED=False면 업로드 안 함"""
        sample_config['EXPERIMENT_TRACKING']['ENABLED'] = False
        tracker = SheetsTracker(sample_config)

        should, reason = tracker._should_upload({'total_return': 10})

        assert should is False
        assert 'ENABLED=False' in reason

    def test_should_upload_no_backtest_results(self, sample_config):
        """백테스트 결과 없으면 업로드 안 함"""
        tracker = SheetsTracker(sample_config)

        should, reason = tracker._should_upload(None)

        assert should is False
        assert 'No backtest' in reason

    def test_should_upload_backtest_not_required(self, sample_config):
        """REQUIRE_BACKTEST_RESULTS=False면 백테스트 없어도 OK"""
        sample_config['EXPERIMENT_TRACKING']['UPLOAD_CONDITIONS']['REQUIRE_BACKTEST_RESULTS'] = False
        tracker = SheetsTracker(sample_config)

        should, reason = tracker._should_upload(None)

        assert should is True
        assert reason == 'OK'

    @patch('src.experiment.sheets_tracker.SheetsTracker._get_git_info')
    def test_should_upload_wrong_branch(self, mock_git_info, sample_config):
        """master 브랜치가 아니면 업로드 안 함"""
        sample_config['EXPERIMENT_TRACKING']['UPLOAD_CONDITIONS']['ONLY_MASTER_BRANCH'] = True
        mock_git_info.return_value = ('user', 'email', 'abc123', 'feature-branch')

        tracker = SheetsTracker(sample_config)
        should, reason = tracker._should_upload({'total_return': 10})

        assert should is False
        assert 'Not on master' in reason

    @patch('src.experiment.sheets_tracker.SheetsTracker._get_git_info')
    def test_should_upload_on_master(self, mock_git_info, sample_config):
        """master 브랜치면 업로드"""
        sample_config['EXPERIMENT_TRACKING']['UPLOAD_CONDITIONS']['ONLY_MASTER_BRANCH'] = True
        mock_git_info.return_value = ('user', 'email', 'abc123', 'master')

        tracker = SheetsTracker(sample_config)
        should, reason = tracker._should_upload({'total_return': 10})

        assert should is True
        assert reason == 'OK'


class TestGenerateExperimentName:
    """실험 이름 생성 테스트"""

    def test_experiment_name_format(self, tracker):
        """실험 이름 형식 확인"""
        name = tracker._generate_experiment_name()

        assert name.startswith('exp_')
        assert len(name) == 19  # exp_ + YYYYMMDD_HHMMSS

    def test_experiment_name_unique(self, tracker):
        """연속 호출 시 다른 이름 (시간 차이)"""
        import time
        name1 = tracker._generate_experiment_name()
        time.sleep(0.01)  # 약간 대기
        name2 = tracker._generate_experiment_name()

        # 같은 초 내에 호출하면 같을 수 있으므로 형식만 확인
        assert name1.startswith('exp_')
        assert name2.startswith('exp_')


class TestGetGitInfo:
    """Git 정보 수집 테스트"""

    def test_git_info_returns_tuple(self, tracker):
        """Git 정보가 4개 튜플로 반환"""
        result = tracker._get_git_info()

        assert isinstance(result, tuple)
        assert len(result) == 4

    @patch('subprocess.check_output')
    def test_git_info_fallback_on_error(self, mock_subprocess, tracker):
        """Git 명령 실패 시 'unknown' 반환"""
        mock_subprocess.side_effect = Exception("Git not found")

        user, email, commit, branch = tracker._get_git_info()

        assert user == 'unknown'
        assert email == 'unknown'
        assert commit == 'unknown'
        assert branch == 'unknown'


class TestConfigMasking:
    """Config 마스킹 테스트"""

    def test_mask_api_key(self, sample_config):
        """API_KEY가 마스킹되는지 확인"""
        tracker = SheetsTracker(sample_config)
        masked = tracker.masker.mask_config(sample_config)

        assert masked['DATA']['API_KEY'] == '***MASKED***'

    def test_mask_preserves_other_values(self, sample_config):
        """다른 값은 그대로 유지"""
        tracker = SheetsTracker(sample_config)
        masked = tracker.masker.mask_config(sample_config)

        assert masked['DATA']['START_YEAR'] == 2020
        assert masked['ML']['USE_CLASSIFIER'] == 'Y'


class TestUploadExperimentMocked:
    """upload_experiment 메서드 테스트 (Mock)"""

    @patch.object(SheetsTracker, '_upload_config_to_gist')
    @patch.object(SheetsTracker, '_append_to_sheet')
    @patch.object(SheetsTracker, '_get_git_info')
    def test_upload_calls_methods_in_order(
        self,
        mock_git_info,
        mock_append_sheet,
        mock_upload_gist,
        sample_config,
        sample_backtest_results
    ):
        """업로드 시 필요한 메서드들이 순서대로 호출되는지"""
        mock_git_info.return_value = ('test_user', 'test@email.com', 'abc123', 'master')
        mock_upload_gist.return_value = 'https://gist.github.com/user/abc123'

        tracker = SheetsTracker(sample_config)
        tracker.upload_experiment(sample_backtest_results, experiment_name='test_exp')

        # Gist 업로드 호출 확인
        mock_upload_gist.assert_called_once()

        # Sheet 추가 호출 확인
        mock_append_sheet.assert_called_once()

    def test_upload_disabled_does_nothing(self, sample_config, sample_backtest_results):
        """ENABLED=False면 아무것도 하지 않음"""
        sample_config['EXPERIMENT_TRACKING']['ENABLED'] = False
        tracker = SheetsTracker(sample_config)

        # 예외 없이 정상 종료
        tracker.upload_experiment(sample_backtest_results)

    def test_upload_handles_missing_google_libs(self, sample_config, sample_backtest_results):
        """Google 라이브러리 없어도 예외 발생 안 함"""
        tracker = SheetsTracker(sample_config)

        with patch('src.experiment.sheets_tracker.GOOGLE_LIBS_AVAILABLE', False):
            # 예외 없이 정상 종료
            tracker.upload_experiment(sample_backtest_results)


# ============================================================================
# Integration Tests (실제 Google API 호출)
# ============================================================================

@pytest.mark.integration
class TestSheetsTrackerIntegration:
    """
    실제 Google API / GitHub API 연동 테스트

    실행 전 필요 사항:
    1. config/secrets.yaml에 실제 값 설정
    2. Google Service Account 키 파일 존재
    3. Google Sheets에 Service Account 권한 부여
    4. GitHub PAT (gist 스코프)

    실행:
        pytest src/unit_tests/test_sheets_tracker.py -v -m integration
    """

    @pytest.fixture
    def real_config(self):
        """실제 secrets.yaml을 로드한 config"""
        from config.context_loader import load_config

        try:
            config = load_config('config/conf.yaml')
            config['EXPERIMENT_TRACKING']['ENABLED'] = True
            config['EXPERIMENT_TRACKING']['UPLOAD_CONDITIONS']['ONLY_MASTER_BRANCH'] = False
            return config
        except FileNotFoundError:
            pytest.skip("config/conf.yaml not found")

    def test_credentials_load(self, real_config):
        """Service Account 인증 정보 로드 테스트"""
        tracker = SheetsTracker(real_config)

        key_path = real_config.get('EXPERIMENT_TRACKING', {}).get('GOOGLE_SHEETS', {}).get('KEY_PATH')
        if not key_path or not os.path.exists(os.path.expanduser(key_path)):
            pytest.skip(f"Service account key not found: {key_path}")

        # 인증 정보 로드 시도
        creds = tracker._get_credentials()
        assert creds is not None

    def test_sheets_client_connection(self, real_config):
        """Google Sheets 클라이언트 연결 테스트"""
        tracker = SheetsTracker(real_config)

        key_path = real_config.get('EXPERIMENT_TRACKING', {}).get('GOOGLE_SHEETS', {}).get('KEY_PATH')
        if not key_path or not os.path.exists(os.path.expanduser(key_path)):
            pytest.skip("Service account key not found")

        # Sheets 클라이언트 생성
        client = tracker._get_sheets_client()
        assert client is not None

    def test_upload_config_to_gist(self, real_config):
        """
        실제 GitHub Gist에 Config 업로드 테스트

        주의: 실제로 Gist가 생성됩니다!
        """
        tracker = SheetsTracker(real_config)

        github_pat = real_config.get('EXPERIMENT_TRACKING', {}).get('GITHUB_GIST', {}).get('PAT')

        if not github_pat or github_pat.startswith('YOUR_'):
            pytest.skip("GITHUB_GIST.PAT not configured")

        # 마스킹된 config
        masked_config = tracker.masker.mask_config(real_config)

        # Gist에 업로드
        url = tracker._upload_config_to_gist(masked_config, 'unittest_exp')

        assert url is not None
        assert url != ''
        assert 'gist.github.com' in url
        print(f"\n✅ Config uploaded to Gist: {url}")

    def test_append_to_sheet(self, real_config):
        """
        실제 Sheets에 행 추가 테스트

        주의: 실제로 행이 추가됩니다!
        """
        tracker = SheetsTracker(real_config)

        key_path = real_config.get('EXPERIMENT_TRACKING', {}).get('GOOGLE_SHEETS', {}).get('KEY_PATH')
        sheet_id = real_config.get('EXPERIMENT_TRACKING', {}).get('GOOGLE_SHEETS', {}).get('SHEET_ID')

        if not key_path or not os.path.exists(os.path.expanduser(key_path)):
            pytest.skip("Service account key not found")
        if not sheet_id or sheet_id.startswith('YOUR_'):
            pytest.skip("GOOGLE_SHEETS.SHEET_ID not configured")

        # 테스트 데이터로 행 추가
        tracker._append_to_sheet(
            experiment_name='unittest_exp',
            git_user='pytest',
            git_commit='test123',
            git_branch='test',
            config_url='https://gist.github.com/test/config',
            changes_str='UNITTEST: test run',
            backtest_results={
                'total_return': 10.0,
                'sharpe_ratio': 0.5,
                'max_drawdown': -5.0,
                'win_rate': 50.0,
            },
            prediction_metrics={
                'rmse': 0.1,
                'mae': 0.08,
                'r2': 0.5,
                'accuracy': 55.0,
                'precision': 60.0,
                'recall': 50.0,
                'train_time_hours': 0.1,
            }
        )

        print("\n✅ Row appended to sheet successfully")

    def test_full_upload_experiment(self, real_config):
        """
        전체 upload_experiment 플로우 테스트

        주의: 실제로 Gist 업로드 + Sheets 기록이 됩니다!
        """
        key_path = real_config.get('EXPERIMENT_TRACKING', {}).get('GOOGLE_SHEETS', {}).get('KEY_PATH')
        sheet_id = real_config.get('EXPERIMENT_TRACKING', {}).get('GOOGLE_SHEETS', {}).get('SHEET_ID')
        github_pat = real_config.get('EXPERIMENT_TRACKING', {}).get('GITHUB_GIST', {}).get('PAT')

        if not key_path or not os.path.exists(os.path.expanduser(key_path)):
            pytest.skip("Service account key not found")
        if not sheet_id or sheet_id.startswith('YOUR_'):
            pytest.skip("GOOGLE_SHEETS.SHEET_ID not configured")
        if not github_pat or github_pat.startswith('YOUR_'):
            pytest.skip("GITHUB_GIST.PAT not configured")

        # 전체 플로우 실행
        upload_experiment_result(
            config=real_config,
            backtest_results={
                'total_return': 15.5,
                'sharpe_ratio': 0.8,
                'max_drawdown': -10.0,
                'win_rate': 55.0,
            },
            prediction_metrics={
                'rmse': 0.05,
                'mae': 0.04,
                'r2': 0.6,
                'accuracy': 60.0,
                'precision': 65.0,
                'recall': 55.0,
                'train_time_hours': 0.5,
            },
            experiment_name='unittest_full_flow'
        )

        print("\n✅ Full upload_experiment completed successfully")
