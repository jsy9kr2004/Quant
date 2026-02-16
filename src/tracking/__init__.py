"""ML 실험 관리를 위한 실험 추적 모듈입니다.

이 패키지는 다음 도구들을 제공합니다:
1. Google Sheets/Gist를 통한 실험 추적
2. 모델 성능 모니터링
3. 통합 보고서 생성

주요 컴포넌트:
    SheetsTracker: Google Sheets 기반 실험 추적기
    ConfigMasker: 안전한 공유를 위한 설정 마스킹
    PerformanceMonitor: 모델 성능 모니터링
    IntegratedReportWriter: 종합 Excel 보고서 생성

사용 예시:
    실험 결과 추적::

        from src.tracking import SheetsTracker, IntegratedReportWriter

        # 실험 결과 업로드
        tracker = SheetsTracker(config)
        tracker.upload_result(experiment_data)

        # 통합 보고서 생성
        writer = IntegratedReportWriter(output_path)
        writer.write_report(results)
"""

from .sheets_tracker import SheetsTracker, upload_experiment_result
from .config_masker import ConfigMasker
from .performance_monitor import PerformanceMonitor
from .integrated_report import IntegratedReportWriter

__all__ = [
    'SheetsTracker',
    'upload_experiment_result',
    'ConfigMasker',
    'PerformanceMonitor',
    'IntegratedReportWriter',
]

__version__ = '1.0.0'
__author__ = 'Quant Trading Team'
__description__ = '실험 추적 및 보고서 생성'
