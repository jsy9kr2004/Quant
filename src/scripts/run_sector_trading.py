"""
섹터별 실전 트레이딩 스크립트

섹터별로 다른 피처를 사용하여 모델을 학습하고,
전체에서 top N개 주식을 선택합니다.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path

from src.backtest.sector_ensemble import SectorEnsemble, create_default_sector_configs
from src.models.xgboost_model import XGBoostModel

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/sector_trading.log'),
        logging.StreamHandler()
    ]
)


def load_data_with_sector(data_path='./VIEW', start_year=2018, end_year=2023):
    """
    섹터 정보가 포함된 데이터 로드

    Args:
        data_path: 데이터 디렉토리
        start_year: 시작 년도
        end_year: 종료 년도

    Returns:
        데이터프레임
    """
    logging.info(f"데이터 로드 중: {data_path}")

    try:
        # Symbol list (섹터 정보 포함)
        symbol_path = Path(data_path) / 'symbol_list.csv'
        if not symbol_path.exists():
            logging.error(f"Symbol list 파일을 찾을 수 없습니다: {symbol_path}")
            return None

        symbol_df = pd.read_csv(symbol_path)

        # Financial statement 데이터 로드
        fs_dfs = []
        for year in range(start_year, end_year + 1):
            fs_path = Path(data_path) / f'financial_statement_{year}.csv'
            if fs_path.exists():
                fs_df = pd.read_csv(fs_path)
                fs_dfs.append(fs_df)
                logging.info(f"  ✓ {year} 재무제표 로드")

        if not fs_dfs:
            logging.error("재무제표 데이터를 찾을 수 없습니다.")
            return None

        # 재무제표 통합
        fs_df = pd.concat(fs_dfs, ignore_index=True)
        fs_df['date'] = pd.to_datetime(fs_df['date'])

        # Symbol과 재무제표 병합
        df = pd.merge(fs_df, symbol_df[['symbol', 'sector']], on='symbol', how='left')

        # 섹터 정보 없는 경우 제외
        df = df.dropna(subset=['sector'])

        logging.info(f"데이터 로드 완료: {len(df):,} rows")
        logging.info(f"섹터 수: {df['sector'].nunique()}")
        logging.info(f"섹터별 분포:\n{df['sector'].value_counts()}")

        return df

    except Exception as e:
        logging.error(f"데이터 로드 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def prepare_target(df, target_months=3):
    """
    타겟 변수 생성 (미래 수익률)

    Args:
        df: 데이터프레임
        target_months: 예측 기간 (개월)

    Returns:
        타겟이 추가된 데이터프레임
    """
    logging.info(f"타겟 변수 생성 중... (예측 기간: {target_months}개월)")

    df = df.sort_values(['symbol', 'date'])

    # 각 주식별로 미래 수익률 계산
    df['future_price'] = df.groupby('symbol')['close'].shift(-target_months)
    df['future_return'] = (df['future_price'] - df['close']) / df['close'] * 100

    # 타겟: 수익률이 양수면 1, 음수면 0
    df['target'] = (df['future_return'] > 0).astype(int)

    # 미래 데이터가 없는 행 제거
    df = df.dropna(subset=['future_return'])

    logging.info(f"타겟 생성 완료: {len(df):,} rows")
    logging.info(f"양성 클래스 비율: {df['target'].mean():.2%}")

    return df


def main():
    """메인 함수"""
    logging.info("="*60)
    logging.info("섹터별 실전 트레이딩 시작")
    logging.info("="*60)

    # ===== 설정 =====
    START_YEAR = 2018
    END_YEAR = 2023
    DATA_PATH = './VIEW'  # 실제 데이터 경로
    TRAIN_YEARS = 3  # 학습에 사용할 과거 년수
    TOP_K = 10  # 선택할 주식 개수
    REBALANCE_DATE = datetime(2024, 1, 1)  # 리밸런싱 날짜

    # ===== 데이터 로드 =====
    df = load_data_with_sector(DATA_PATH, START_YEAR, END_YEAR)

    if df is None:
        logging.error("데이터 로드 실패. 종료합니다.")
        logging.info("\n⚠️ 실제 데이터가 없는 경우 샘플 데이터로 테스트하려면:")
        logging.info("   python examples/comprehensive_example.py")
        return

    # ===== 타겟 생성 =====
    # 최적 리밸런싱 기간 파일이 있으면 사용
    optimal_period_file = Path('./config/optimal_rebalance_period.txt')
    if optimal_period_file.exists():
        with open(optimal_period_file, 'r') as f:
            target_months = int(f.read().strip())
        logging.info(f"최적 리밸런싱 기간 사용: {target_months}개월")
    else:
        target_months = 3  # 기본값
        logging.info(f"기본 리밸런싱 기간 사용: {target_months}개월")

    df = prepare_target(df, target_months=target_months)

    # ===== 학습/예측 데이터 분리 =====
    train_end = REBALANCE_DATE - relativedelta(days=1)
    train_start = REBALANCE_DATE - relativedelta(years=TRAIN_YEARS)

    train_df = df[(df['date'] >= train_start) & (df['date'] <= train_end)]
    predict_df = df[df['date'] == REBALANCE_DATE]

    if len(predict_df) == 0:
        # 리밸런싱 날짜에 데이터가 없으면 가장 최근 데이터 사용
        predict_df = df[df['date'] == df['date'].max()]
        logging.warning(f"리밸런싱 날짜 데이터 없음. 최근 데이터 사용: {predict_df['date'].iloc[0]}")

    logging.info(f"\n학습 데이터: {len(train_df):,} rows ({train_start.strftime('%Y-%m-%d')} ~ {train_end.strftime('%Y-%m-%d')})")
    logging.info(f"예측 데이터: {len(predict_df):,} rows ({predict_df['date'].iloc[0] if len(predict_df) > 0 else 'N/A'})")

    # ===== SectorEnsemble 초기화 =====
    ensemble = SectorEnsemble(sector_col='sector')

    # 기본 섹터 설정 사용
    default_configs = create_default_sector_configs(XGBoostModel)

    # 실제 데이터에 있는 섹터만 설정
    available_sectors = train_df['sector'].unique()

    for sector_name, config in default_configs.items():
        if sector_name in available_sectors:
            ensemble.configure_sector(
                sector_name=sector_name,
                **config
            )
        else:
            logging.info(f"섹터 '{sector_name}'는 데이터에 없어 건너뜁니다.")

    # ===== 학습 → 예측 → 선택 =====
    logging.info("\n" + "="*60)
    logging.info("섹터별 모델 학습 및 주식 선택")
    logging.info("="*60)

    top_stocks = ensemble.fit_predict_select(
        train_df=train_df,
        predict_df=predict_df,
        target_col='target',
        top_k=TOP_K,
        use_feature_selection=True,
        symbol_col='symbol'
    )

    # ===== 결과 출력 및 저장 =====
    if top_stocks.empty:
        logging.error("선택된 주식이 없습니다!")
        return

    logging.info("\n" + "="*60)
    logging.info(f"🎯 선택된 Top {TOP_K} 주식")
    logging.info("="*60)

    # 상세 정보 출력
    for idx, row in top_stocks.iterrows():
        logging.info(f"\n{idx+1}. {row['symbol']}")
        logging.info(f"   섹터: {row['sector']}")
        logging.info(f"   예측 점수: {row['predicted_score']:.4f}")
        logging.info(f"   예측 클래스: {row['predicted_class']}")

    # 섹터별 분포
    sector_dist = top_stocks['sector'].value_counts()
    logging.info(f"\n섹터별 분포:")
    for sector, count in sector_dist.items():
        logging.info(f"  {sector}: {count}개 ({count/TOP_K*100:.1f}%)")

    # ===== 결과 저장 =====
    output_dir = Path('./results/sector_trading')
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 선택된 주식 저장
    output_file = output_dir / f'selected_stocks_{timestamp}.csv'
    top_stocks.to_csv(output_file, index=False)
    logging.info(f"\n💾 선택된 주식 저장: {output_file}")

    # 섹터 설정 저장
    config_file = output_dir / f'sector_config_{timestamp}.json'
    ensemble.save_config(str(config_file))
    logging.info(f"💾 섹터 설정 저장: {config_file}")

    # 섹터 요약 저장
    sector_summary = ensemble.get_sector_summary()
    summary_file = output_dir / f'sector_summary_{timestamp}.csv'
    sector_summary.to_csv(summary_file, index=False)
    logging.info(f"💾 섹터 요약 저장: {summary_file}")

    # ===== 매수 추천 출력 =====
    logging.info("\n" + "="*60)
    logging.info("📈 매수 추천")
    logging.info("="*60)
    logging.info(f"\n리밸런싱 날짜: {REBALANCE_DATE.strftime('%Y-%m-%d')}")
    logging.info(f"다음 리밸런싱: {(REBALANCE_DATE + relativedelta(months=target_months)).strftime('%Y-%m-%d')}")
    logging.info(f"\n매수할 주식 ({TOP_K}개):")

    for idx, row in top_stocks.iterrows():
        logging.info(f"  {row['symbol']:<10} (섹터: {row['sector']:<15}, 점수: {row['predicted_score']:.4f})")

    logging.info("\n" + "="*60)
    logging.info("완료")
    logging.info("="*60)


if __name__ == '__main__':
    # 로그 디렉토리 생성
    Path('./logs').mkdir(exist_ok=True)

    main()
