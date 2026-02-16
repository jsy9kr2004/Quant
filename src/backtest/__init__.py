"""퀀트 트레이딩 전략을 위한 백테스팅 모듈입니다.

이 패키지는 과거 데이터를 기반으로 트레이딩 전략을 검증하기 위한
백테스팅 인프라를 제공합니다.

주요 구성요소:
    MLBacktest: ML 기반 Walk-Forward 백테스트 엔진 (권장)
    Backtest: 레거시 plan 기반 백테스트 엔진 (deprecated)
    PlanHandler: 레거시 전략 plan 실행 및 종목 점수 산정 (deprecated)

사용 예시:
    ML Walk-Forward 백테스트 워크플로우 (권장)::

        from src.backtest import MLBacktest
        from src.infra.context_loader import load_config, MainContext

        # 설정 로드
        config = load_config('config/conf.yaml')
        main_ctx = MainContext(config)

        # ML 기반 Walk-Forward 백테스트 실행 (regressor.py 예측 캐시 필요)
        ml_bt = MLBacktest(
            config=config,
            main_ctx=main_ctx,
            rebalance_period=3,
            top_k=20,
        )

        # 백테스트 실행
        results = ml_bt.run()

        # 결과는 reports 디렉토리에 자동 저장됩니다

참고:
    - docs/ML_BACKTEST_GUIDE.md: ML 백테스팅 종합 가이드
    - docs/WORKFLOW_GUIDE.md: 시스템 워크플로우 문서
"""

from .backtest import Backtest, PlanHandler
from .ml_backtest import MLBacktest

__all__ = ['MLBacktest', 'Backtest', 'PlanHandler']

__version__ = '1.0.0'
__author__ = 'Quant Trading Team'
__description__ = '퀀트 트레이딩을 위한 백테스팅 프레임워크'
