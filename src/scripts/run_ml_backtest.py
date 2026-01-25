"""
ML 모델 기반 Walk-Forward 백테스트 실행 스크립트

미래 유출을 방지하면서 ML 예측의 실제 수익률을 계산합니다.

사용법:
    python scripts/run_ml_backtest.py

    또는 설정 변경:
    python scripts/run_ml_backtest.py --retrain-freq quarterly --window expanding
"""

import sys
import os
from pathlib import Path

# 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
import argparse
from datetime import datetime

from src.infra.context_loader import load_config, MainContext
from src.infra.logger import get_logger
from src.backtest.ml_backtest import MLBacktest


def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(
        description='ML Walk-Forward Backtest',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 분기마다 재학습, Expanding window
  python run_ml_backtest.py --retrain-freq quarterly --window expanding

  # 매 리밸런싱마다 재학습, Rolling 3년 window
  python run_ml_backtest.py --retrain-freq every --window rolling --window-size 3

  # 한 번만 학습 (현재 방식, 비추천)
  python run_ml_backtest.py --retrain-freq once
        """
    )

    parser.add_argument(
        '--retrain-freq',
        choices=['every', 'quarterly', 'yearly', 'once'],
        default='quarterly',
        help='모델 재학습 빈도 (기본값: quarterly)'
    )

    parser.add_argument(
        '--window',
        choices=['expanding', 'rolling'],
        default='expanding',
        help='학습 윈도우 타입 (기본값: expanding)'
    )

    parser.add_argument(
        '--window-size',
        type=int,
        default=3,
        help='Rolling window 크기 (년 단위, 기본값: 3)'
    )

    parser.add_argument(
        '--top-k',
        type=int,
        default=20,
        help='선택할 종목 수 (기본값: 20)'
    )

    parser.add_argument(
        '--rebalance-period',
        type=int,
        default=3,
        help='리밸런싱 주기 (개월, 기본값: 3)'
    )

    parser.add_argument(
        '--use-sector',
        action='store_true',
        help='섹터별 모델 사용 (기본값: Config 파일 설정 따름)'
    )

    parser.add_argument(
        '--no-sector',
        action='store_true',
        help='통합 모델 사용 (섹터 구분 없음)'
    )

    return parser.parse_args()


def main():
    """메인 실행 함수"""
    # 인자 파싱
    args = parse_args()

    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('./logs/ml_backtest.log'),
            logging.StreamHandler()
        ]
    )

    logger = get_logger('main')

    logger.info("="*80)
    logger.info("ML Walk-Forward Backtest")
    logger.info("="*80)

    # 섹터 모델 사용 여부 결정
    use_sector_model = None
    if args.use_sector:
        use_sector_model = True
        logger.info("Model type: SECTOR-BASED (forced by --use-sector)")
    elif args.no_sector:
        use_sector_model = False
        logger.info("Model type: UNIFIED (forced by --no-sector)")
    else:
        logger.info("Model type: From config file")

    logger.info(f"Retrain frequency: {args.retrain_freq}")
    logger.info(f"Window type: {args.window}")
    if args.window == 'rolling':
        logger.info(f"Window size: {args.window_size} years")
    logger.info(f"Top K: {args.top_k}")
    logger.info(f"Rebalance period: {args.rebalance_period} months")

    # 설정 로드
    try:
        # 여러 경로에서 설정 파일 찾기
        config_paths = [
            'config/conf.yaml',
            '../config/conf.yaml'
        ]

        config = None
        for path in config_paths:
            if os.path.exists(path):
                config = load_config(path)
                logger.info(f"✅ Loaded config from: {path}")
                break

        if config is None:
            raise FileNotFoundError("Config file not found in any of the expected locations")

    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        logger.info("\nPlease create config/conf.yaml")
        logger.info("See config/conf.yaml.template for reference")
        sys.exit(1)

    # 컨텍스트 초기화
    main_ctx = MainContext(config)

    # 디렉토리 생성
    Path('./logs').mkdir(exist_ok=True)
    Path('outputs/reports').mkdir(exist_ok=True)

    # 백테스트 설정 확인
    backtest_config = config.get('BACKTEST', {})
    if not backtest_config:
        logger.warning("⚠️ No BACKTEST section in config, using defaults")
        backtest_config = {
            'START_YEAR': 2023,
            'END_YEAR': 2024,
            'START_MONTH': 3,
            'START_DATE': 13
        }
        config['BACKTEST'] = backtest_config

    # ML 백테스트 실행
    logger.info("\n" + "="*80)
    logger.info("Starting ML Backtest...")
    logger.info("="*80)

    try:
        backtest = MLBacktest(
            config=config,
            main_ctx=main_ctx,
            rebalance_period=args.rebalance_period,
            top_k=args.top_k,
            retrain_frequency=args.retrain_freq,
            window_type=args.window,
            window_size=args.window_size if args.window == 'rolling' else None,
            use_sector_model=use_sector_model
        )

        # 실행
        results = backtest.run()

        logger.info("\n" + "="*80)
        logger.info("✅ Backtest Completed!")
        logger.info("="*80)
        logger.info(f"Results saved to: ./reports/")
        logger.info(f"Logs saved to: ./logs/ml_backtest.log")

        return 0

    except Exception as e:
        logger.error(f"\n❌ Backtest failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
