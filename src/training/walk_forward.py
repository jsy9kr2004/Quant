"""
Walk-Forward 검증

시계열 데이터를 위한 walk-forward 검증 모듈입니다.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import logging
from typing import Dict, List, Any, Optional, Callable


class WalkForwardValidator:
    """
    Walk-Forward 검증을 수행합니다.

    사용 예시:
        (train_months=24, test_months=3인 경우):
        Period 1: Train(2020-01~2021-12) -> Test(2022-01~2022-03)
        Period 2: Train(2020-04~2022-03) -> Test(2022-04~2022-06)  <- 모델 재학습
        Period 3: Train(2020-07~2022-06) -> Test(2022-07~2022-09)  <- 모델 재학습
        ...

    매 기간마다 모델을 다시 학습하여 시장 변화에 적응합니다.
    """

    def __init__(self,
                 train_months: int = 24,
                 test_months: int = 3,
                 retrain_frequency: int = 3,
                 anchored: bool = False):
        """
        Args:
            train_months: 학습 데이터 기간 (개월)
            test_months: 테스트 데이터 기간 (개월)
            retrain_frequency: 재학습 빈도 (개월)
            anchored: True이면 학습 시작점 고정, False이면 롤링 윈도우
        """
        self.train_months = train_months
        self.test_months = test_months
        self.retrain_frequency = retrain_frequency
        self.anchored = anchored

    def generate_windows(self,
                        start_date: datetime,
                        end_date: datetime) -> List[Dict[str, datetime]]:
        """
        학습/테스트 윈도우를 생성합니다.

        Args:
            start_date: 시작 날짜
            end_date: 종료 날짜

        Returns:
            윈도우 정보 리스트
        """
        windows = []
        current_date = start_date + relativedelta(months=self.train_months)
        train_start = start_date

        while current_date < end_date:
            # 학습 기간
            if not self.anchored:
                # 롤링 윈도우: 학습 시작점도 이동
                train_start = current_date - relativedelta(months=self.train_months)
            # 고정 윈도우: 학습 시작점 고정

            train_end = current_date

            # 테스트 기간
            test_start = current_date
            test_end = current_date + relativedelta(months=self.test_months)

            if test_end > end_date:
                break

            windows.append({
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end
            })

            # 다음 윈도우로 이동
            current_date += relativedelta(months=self.retrain_frequency)

        return windows

    def validate(self,
                model_class: Any,
                df: pd.DataFrame,
                date_col: str,
                feature_cols: List[str],
                target_col: str,
                start_date: datetime,
                end_date: datetime,
                model_params: Optional[Dict] = None,
                verbose: bool = True) -> pd.DataFrame:
        """
        Walk-forward 검증을 수행합니다.

        Args:
            model_class: 모델 클래스 (인스턴스화 가능해야 함)
            df: 전체 DataFrame
            date_col: 날짜 컬럼명
            feature_cols: 피처 컬럼 리스트
            target_col: 타겟 컬럼명
            start_date: 시작 날짜
            end_date: 종료 날짜
            model_params: 모델 파라미터
            verbose: 상세 로그 출력 여부

        Returns:
            결과 DataFrame
        """
        if model_params is None:
            model_params = {}

        windows = self.generate_windows(start_date, end_date)

        if verbose:
            logging.info(f"\n{'='*60}")
            logging.info(f"Walk-Forward Validation 시작")
            logging.info(f"총 {len(windows)}개 윈도우")
            logging.info(f"학습 기간: {self.train_months}개월")
            logging.info(f"테스트 기간: {self.test_months}개월")
            logging.info(f"재학습 빈도: {self.retrain_frequency}개월")
            logging.info(f"윈도우 타입: {'Anchored' if self.anchored else 'Rolling'}")
            logging.info(f"{'='*60}\n")

        results = []

        for i, window in enumerate(windows, 1):
            if verbose:
                logging.info(f"\n{'='*60}")
                logging.info(f"Window {i}/{len(windows)}")
                logging.info(f"  Train: {window['train_start'].strftime('%Y-%m-%d')} ~ {window['train_end'].strftime('%Y-%m-%d')}")
                logging.info(f"  Test:  {window['test_start'].strftime('%Y-%m-%d')} ~ {window['test_end'].strftime('%Y-%m-%d')}")

            # 데이터 분할
            train_mask = (df[date_col] >= window['train_start']) & \
                        (df[date_col] < window['train_end'])
            test_mask = (df[date_col] >= window['test_start']) & \
                       (df[date_col] < window['test_end'])

            train_df = df[train_mask].copy()
            test_df = df[test_mask].copy()

            if len(train_df) == 0 or len(test_df) == 0:
                logging.warning(f"  ⚠️ Empty data in window {i}, skipping...")
                continue

            if verbose:
                logging.info(f"  Train samples: {len(train_df):,}")
                logging.info(f"  Test samples:  {len(test_df):,}")

            # 모델 재학습 (핵심!)
            try:
                model = model_class(**model_params)

                X_train = train_df[feature_cols]
                y_train = train_df[target_col]
                X_test = test_df[feature_cols]
                y_test = test_df[target_col]

                # NaN 값 제거
                train_valid_mask = ~(X_train.isnull().any(axis=1) | y_train.isnull())
                test_valid_mask = ~(X_test.isnull().any(axis=1) | y_test.isnull())

                X_train = X_train[train_valid_mask]
                y_train = y_train[train_valid_mask]
                X_test = X_test[test_valid_mask]
                y_test = y_test[test_valid_mask]

                if len(X_train) == 0 or len(X_test) == 0:
                    logging.warning(f"  ⚠️ No valid data after NaN removal, skipping...")
                    continue

                # 학습
                if verbose:
                    logging.info(f"  🔄 모델 학습 중...")

                model.fit(X_train, y_train, verbose=0)

                # 평가
                metrics = model.evaluate(X_test, y_test)

                result = {
                    'window': i,
                    'train_start': window['train_start'],
                    'train_end': window['train_end'],
                    'test_start': window['test_start'],
                    'test_end': window['test_end'],
                    'train_samples': len(X_train),
                    'test_samples': len(X_test),
                    **metrics
                }

                results.append(result)

                if verbose:
                    logging.info(f"  📊 Performance:")
                    for metric, value in metrics.items():
                        logging.info(f"     {metric}: {value:.4f}")

            except Exception as e:
                logging.error(f"  ❌ Error in window {i}: {str(e)}")
                continue

        # 결과 요약
        if not results:
            logging.error("No valid results from walk-forward validation!")
            return pd.DataFrame()

        results_df = pd.DataFrame(results)

        if verbose:
            logging.info(f"\n{'='*60}")
            logging.info("Walk-Forward Validation 결과 요약")
            logging.info(f"{'='*60}")
            logging.info(f"\n{results_df.to_string()}\n")

            # 평균 성능
            metric_cols = [col for col in results_df.columns if col not in [
                'window', 'train_start', 'train_end', 'test_start', 'test_end',
                'train_samples', 'test_samples'
            ]]

            logging.info("평균 성능:")
            for metric in metric_cols:
                mean_val = results_df[metric].mean()
                std_val = results_df[metric].std()
                logging.info(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")

        return results_df

    def validate_with_custom_model(self,
                                   train_func: Callable,
                                   evaluate_func: Callable,
                                   df: pd.DataFrame,
                                   date_col: str,
                                   start_date: datetime,
                                   end_date: datetime,
                                   **kwargs) -> pd.DataFrame:
        """
        사용자 정의 학습/평가 함수를 사용한 walk-forward 검증을 수행합니다.

        Args:
            train_func: 학습 함수 (train_df 입력, model 반환)
            evaluate_func: 평가 함수 (model, test_df 입력, metrics dict 반환)
            df: 전체 DataFrame
            date_col: 날짜 컬럼명
            start_date: 시작 날짜
            end_date: 종료 날짜
            **kwargs: train_func, evaluate_func에 전달할 추가 인자

        Returns:
            결과 DataFrame
        """
        windows = self.generate_windows(start_date, end_date)
        results = []

        for i, window in enumerate(windows, 1):
            logging.info(f"\nWindow {i}/{len(windows)}")
            logging.info(f"  Train: {window['train_start']} ~ {window['train_end']}")
            logging.info(f"  Test:  {window['test_start']} ~ {window['test_end']}")

            # 데이터 분할
            train_mask = (df[date_col] >= window['train_start']) & \
                        (df[date_col] < window['train_end'])
            test_mask = (df[date_col] >= window['test_start']) & \
                       (df[date_col] < window['test_end'])

            train_df = df[train_mask].copy()
            test_df = df[test_mask].copy()

            if len(train_df) == 0 or len(test_df) == 0:
                logging.warning(f"  ⚠️ Empty data, skipping...")
                continue

            try:
                # 사용자 정의 함수로 학습
                model = train_func(train_df, **kwargs)

                # 사용자 정의 함수로 평가
                metrics = evaluate_func(model, test_df, **kwargs)

                result = {
                    'window': i,
                    'train_start': window['train_start'],
                    'test_start': window['test_start'],
                    **metrics
                }

                results.append(result)

                logging.info(f"  📊 Performance: {metrics}")

            except Exception as e:
                logging.error(f"  ❌ Error: {str(e)}")
                continue

        return pd.DataFrame(results)
