"""
전체 파이프라인 실행 스크립트

1단계: 리밸런싱 기간 최적화 (선택 사항)
2단계: 모델 성능 비교 (선택 사항)
3단계: 섹터별 주식 선택 (필수)
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import logging
from pathlib import Path
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/full_pipeline.log'),
        logging.StreamHandler()
    ]
)


def run_rebalance_optimization():
    """1단계: 리밸런싱 기간 최적화"""
    logging.info("\n" + "="*60)
    logging.info("1단계: 리밸런싱 기간 최적화")
    logging.info("="*60)

    try:
        from run_rebalance_optimization import main as rebalance_main
        rebalance_main()
        return True
    except Exception as e:
        logging.error(f"리밸런싱 기간 최적화 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_model_comparison():
    """2단계: 모델 성능 비교"""
    logging.info("\n" + "="*60)
    logging.info("2단계: 모델 성능 비교")
    logging.info("="*60)

    try:
        from run_model_comparison import main as comparison_main
        comparison_main()
        return True
    except Exception as e:
        logging.error(f"모델 성능 비교 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_sector_trading():
    """3단계: 섹터별 주식 선택"""
    logging.info("\n" + "="*60)
    logging.info("3단계: 섹터별 주식 선택")
    logging.info("="*60)

    try:
        from run_sector_trading import main as trading_main
        trading_main()
        return True
    except Exception as e:
        logging.error(f"섹터별 주식 선택 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """메인 함수"""
    start_time = datetime.now()

    logging.info("="*60)
    logging.info("전체 파이프라인 시작")
    logging.info("="*60)
    logging.info(f"시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # ===== 설정 =====
    RUN_REBALANCE_OPTIMIZATION = False  # 최초 1회만 실행 (시간 오래 걸림)
    RUN_MODEL_COMPARISON = False         # 모델 변경 시에만 실행
    RUN_SECTOR_TRADING = True            # 매번 실행 (실제 주식 선택)

    results = {
        'rebalance_optimization': None,
        'model_comparison': None,
        'sector_trading': None
    }

    # ===== 1단계: 리밸런싱 기간 최적화 =====
    if RUN_REBALANCE_OPTIMIZATION:
        results['rebalance_optimization'] = run_rebalance_optimization()

        if not results['rebalance_optimization']:
            logging.warning("⚠️ 리밸런싱 기간 최적화 실패. 기본값 사용.")
    else:
        logging.info("\n1단계: 리밸런싱 기간 최적화 건너뛰기 (설정에서 비활성화)")

    # ===== 2단계: 모델 성능 비교 =====
    if RUN_MODEL_COMPARISON:
        results['model_comparison'] = run_model_comparison()

        if not results['model_comparison']:
            logging.warning("⚠️ 모델 성능 비교 실패. 기존 모델 사용.")
    else:
        logging.info("\n2단계: 모델 성능 비교 건너뛰기 (설정에서 비활성화)")

    # ===== 3단계: 섹터별 주식 선택 (필수) =====
    if RUN_SECTOR_TRADING:
        results['sector_trading'] = run_sector_trading()

        if not results['sector_trading']:
            logging.error("❌ 섹터별 주식 선택 실패!")
    else:
        logging.warning("\n3단계: 섹터별 주식 선택 건너뛰기")

    # ===== 최종 결과 요약 =====
    end_time = datetime.now()
    duration = end_time - start_time

    logging.info("\n" + "="*60)
    logging.info("전체 파이프라인 완료")
    logging.info("="*60)
    logging.info(f"종료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"소요 시간: {duration}")

    logging.info("\n실행 결과:")
    logging.info(f"  1. 리밸런싱 기간 최적화: {'✅ 성공' if results['rebalance_optimization'] else ('❌ 실패' if results['rebalance_optimization'] is False else '⏭️ 건너뜀')}")
    logging.info(f"  2. 모델 성능 비교: {'✅ 성공' if results['model_comparison'] else ('❌ 실패' if results['model_comparison'] is False else '⏭️ 건너뜀')}")
    logging.info(f"  3. 섹터별 주식 선택: {'✅ 성공' if results['sector_trading'] else ('❌ 실패' if results['sector_trading'] is False else '⏭️ 건너뜀')}")

    # 결과 파일 위치
    logging.info("\n결과 파일 위치:")

    if results['rebalance_optimization']:
        logging.info("  - ./results/rebalancing_optimization/")

    if results['model_comparison']:
        logging.info("  - ./results/model_comparison/")

    if results['sector_trading']:
        logging.info("  - ./results/sector_trading/")

    logging.info("\n" + "="*60)

    # 성공 여부 반환
    return results['sector_trading'] is True


if __name__ == '__main__':
    # 필요한 디렉토리 생성
    Path('./logs').mkdir(exist_ok=True)
    Path('./results').mkdir(exist_ok=True)
    Path('./config').mkdir(exist_ok=True)

    success = main()

    sys.exit(0 if success else 1)
