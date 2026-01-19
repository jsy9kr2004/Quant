"""
pytest 설정 파일

마커:
    - integration: 실제 외부 서비스(Google API 등) 연동 테스트
    - slow: 오래 걸리는 테스트

사용 예:
    # integration 테스트 제외
    pytest src/unit_tests/ -v -m "not integration"

    # integration 테스트만
    pytest src/unit_tests/ -v -m integration

    # 빠른 테스트만
    pytest src/unit_tests/ -v -m "not slow"
"""

import pytest


def pytest_configure(config):
    """pytest 마커 등록"""
    config.addinivalue_line(
        "markers", "integration: 실제 외부 서비스(Google API 등) 연동 테스트"
    )
    config.addinivalue_line(
        "markers", "slow: 오래 걸리는 테스트"
    )
