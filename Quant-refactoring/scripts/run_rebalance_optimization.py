"""
리밸런싱 기간 최적화 실행 스크립트

기존 백테스팅 시스템에서 최적의 리밸런싱 기간을 찾습니다.
"""

import sys
import os

# 상위 디렉토리를 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path

from optimization.rebalance_optimizer import RebalancingOptimizer

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/rebalance_optimization.log'),
        logging.StreamHandler()
    ]
)


def simple_backtest_wrapper(data, date_col, start_date, end_date, rebalance_months, top_k):
    """
    간단한 백테스팅 래퍼

    실제 백테스팅 로직을 여기에 구현하거나
    기존 백테스팅 시스템을 호출

    Args:
        data: 전체 데이터
        date_col: 날짜 컬럼
        start_date: 시작 날짜
        end_date: 종료 날짜
        rebalance_months: 리밸런싱 기간 (개월)
        top_k: 선택할 주식 개수

    Returns:
        결과 DataFrame (return 컬럼 필수)
    """
    from dateutil.relativedelta import relativedelta

    logging.info(f"백테스팅 시작: {rebalance_months}개월 기간")

    # 리밸런싱 날짜 생성
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date += relativedelta(months=rebalance_months)

    results = []

    for i in range(len(dates) - 1):
        period_start = dates[i]
        period_end = dates[i + 1]

        # 해당 기간 데이터
        period_mask = (data[date_col] >= period_start) & (data[date_col] < period_end)
        period_data = data[period_mask]

        if len(period_data) == 0:
            continue

        # ===== 여기에 실제 백테스팅 로직 구현 =====
        # 1. 모델 학습 (period_start 이전 데이터로)
        # 2. 예측 (period_start 데이터로)
        # 3. top_k 선택
        # 4. period_end에서 수익률 계산

        # 예시: 랜덤 수익률 (실제로는 모델 기반으로)
        # 실제 구현 시에는 여기서 모델을 학습하고 예측해야 함
        avg_return = np.random.uniform(-5, 15)  # -5% ~ +15%

        results.append({
            'period_start': period_start,
            'period_end': period_end,
            'return': avg_return  # 필수 컬럼!
        })

        logging.debug(f"  Period {i+1}: {period_start.strftime('%Y-%m-%d')} ~ {period_end.strftime('%Y-%m-%d')}, Return: {avg_return:.2f}%")

    return pd.DataFrame(results)


def load_data(data_path='./VIEW', start_year=2018, end_year=2023):
    """
    데이터 로드

    Args:
        data_path: 데이터 디렉토리 경로
        start_year: 시작 년도
        end_year: 종료 년도

    Returns:
        데이터프레임
    """
    logging.info(f"데이터 로드 중: {data_path}")

    try:
        # Price 데이터 로드
        price_path = Path(data_path) / 'price.csv'
        if not price_path.exists():
            logging.error(f"Price 파일을 찾을 수 없습니다: {price_path}")
            return None

        df = pd.read_csv(price_path)
        df['date'] = pd.to_datetime(df['date'])

        # 기간 필터링
        df = df[(df['date'].dt.year >= start_year) & (df['date'].dt.year <= end_year)]

        logging.info(f"데이터 로드 완료: {len(df):,} rows")
        logging.info(f"기간: {df['date'].min()} ~ {df['date'].max()}")

        return df

    except Exception as e:
        logging.error(f"데이터 로드 실패: {str(e)}")
        return None


def main():
    """메인 함수"""
    logging.info("="*60)
    logging.info("리밸런싱 기간 최적화 시작")
    logging.info("="*60)

    # ===== 설정 =====
    START_YEAR = 2018
    END_YEAR = 2023
    DATA_PATH = './VIEW'  # 실제 데이터 경로로 수정
    PERIODS_TO_TEST = [1, 2, 3, 4, 6]  # 테스트할 기간들 (개월)
    OPTIMIZATION_METRIC = 'total_return'  # 또는 'sharpe_ratio', 'win_rate'
    TOP_K = 5  # 매번 선택할 주식 개수

    # ===== 데이터 로드 =====
    data = load_data(DATA_PATH, START_YEAR, END_YEAR)

    if data is None:
        logging.error("데이터 로드 실패. 종료합니다.")
        logging.info("\n⚠️ 실제 데이터가 없는 경우 샘플 데이터로 테스트하려면:")
        logging.info("   python examples/comprehensive_example.py")
        return

    # ===== Optimizer 초기화 =====
    optimizer = RebalancingOptimizer(
        periods_to_test=PERIODS_TO_TEST,
        optimization_metric=OPTIMIZATION_METRIC,
        min_trades=5  # 최소 거래 횟수
    )

    # ===== 최적화 실행 =====
    start_date = datetime(START_YEAR, 1, 1)
    end_date = datetime(END_YEAR, 12, 31)

    result = optimizer.optimize(
        backtest_func=simple_backtest_wrapper,
        data=data,
        date_col='date',
        start_date=start_date,
        end_date=end_date,
        top_k=TOP_K
    )

    # ===== 결과 저장 =====
    output_dir = Path('./results/rebalancing_optimization')
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    optimizer.save_results(str(output_dir / f'optimization_results_{timestamp}.csv'))

    # 시각화 (선택 사항)
    try:
        optimizer.plot_results(str(output_dir / f'optimization_chart_{timestamp}.png'))
    except Exception as e:
        logging.warning(f"차트 생성 실패: {str(e)}")

    # ===== 최종 결과 출력 =====
    if result.get('optimal_period'):
        optimal = result['optimal_period']

        logging.info("\n" + "="*60)
        logging.info("🎯 최적화 완료!")
        logging.info("="*60)
        logging.info(f"\n최적 리밸런싱 기간: {optimal['period_months']}개월")
        logging.info(f"\n성능 지표:")
        logging.info(f"  - 총 수익률: {optimal['total_return']:.2f}%")
        logging.info(f"  - 연평균 수익률: {optimal['annualized_return']:.2f}%")
        logging.info(f"  - 샤프 비율: {optimal['sharpe_ratio']:.3f}")
        logging.info(f"  - 승률: {optimal['win_rate']:.2f}%")
        logging.info(f"  - 거래 횟수: {optimal['num_trades']}")
        logging.info(f"  - 최대 낙폭: {optimal['max_drawdown']:.2f}%")
        logging.info("\n" + "="*60)

        # 설정 파일에 저장 (다음 백테스팅에서 사용)
        config_file = Path('./config/optimal_rebalance_period.txt')
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w') as f:
            f.write(f"{optimal['period_months']}")

        logging.info(f"\n✅ 최적 기간이 {config_file}에 저장되었습니다.")
        logging.info(f"   다음 백테스팅에서 이 값을 사용하세요.")

    else:
        logging.error("최적화 실패!")

    logging.info("\n" + "="*60)
    logging.info("완료")
    logging.info("="*60)


if __name__ == '__main__':
    # 로그 디렉토리 생성
    Path('./logs').mkdir(exist_ok=True)

    main()
