"""
Time Series Cross-Validation

시계열 데이터를 위한 교차 검증
"""

import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import TimeSeriesSplit
from typing import Optional, List, Tuple, Dict, Any


class TimeSeriesCV:
    """
    시계열 데이터를 위한 교차 검증

    예시:
    2020-01 ~ 2023-12 데이터를 5-fold로 나누면:
    Fold 1: Train(2020-01~2021-06) → Test(2021-07~2021-12)
    Fold 2: Train(2020-01~2022-06) → Test(2022-07~2022-12)
    ...

    미래 데이터 누출(data leakage)을 방지하며 robust한 검증 수행
    """

    def __init__(self, n_splits: int = 5, test_size_ratio: float = 0.2):
        """
        Args:
            n_splits: Cross-validation fold 수
            test_size_ratio: 테스트 데이터 비율
        """
        self.n_splits = n_splits
        self.test_size_ratio = test_size_ratio

    def split(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        데이터를 시계열 방식으로 분할

        Args:
            X: 특징 데이터
            y: 타겟 데이터 (선택)

        Returns:
            (train_idx, val_idx) 튜플의 리스트
        """
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        splits = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            splits.append((train_idx, val_idx))

            logging.info(f"Fold {fold}/{self.n_splits}:")
            logging.info(f"  Train samples: {len(train_idx):,}")
            logging.info(f"  Val samples:   {len(val_idx):,}")

        return splits

    def split_by_date(self, df: pd.DataFrame, date_col: str = 'date') -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        날짜 컬럼 기반으로 데이터 분할

        Args:
            df: 데이터프레임
            date_col: 날짜 컬럼명

        Returns:
            (train_df, val_df) 튜플의 리스트
        """
        df_sorted = df.sort_values(date_col).reset_index(drop=True)
        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        splits = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(df_sorted), 1):
            train_data = df_sorted.iloc[train_idx]
            val_data = df_sorted.iloc[val_idx]

            logging.info(f"Fold {fold}/{self.n_splits}:")
            logging.info(f"  Train: {train_data[date_col].min()} ~ {train_data[date_col].max()} ({len(train_data):,} samples)")
            logging.info(f"  Val:   {val_data[date_col].min()} ~ {val_data[date_col].max()} ({len(val_data):,} samples)")

            splits.append((train_data, val_data))

        return splits

    def cross_validate_model(self,
                            model,
                            X: pd.DataFrame,
                            y: pd.Series,
                            dates: Optional[pd.Series] = None,
                            verbose: bool = True) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        모델 교차 검증 수행

        Args:
            model: 학습할 모델 (fit, evaluate 메서드 필요)
            X: 특징 데이터
            y: 타겟 데이터
            dates: 날짜 정보 (선택)
            verbose: 상세 로그 출력 여부

        Returns:
            (평균 점수, 각 fold 점수 리스트)
        """
        splits = self.split(X, y)
        all_scores = []

        for fold, (train_idx, val_idx) in enumerate(splits, 1):
            if verbose:
                logging.info(f"\n{'='*60}")
                logging.info(f"Training Fold {fold}/{self.n_splits}")
                logging.info(f"{'='*60}")

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # 모델 학습
            model.fit(X_train, y_train, X_val, y_val, verbose=0)

            # 평가
            metrics = model.evaluate(X_val, y_val)
            all_scores.append(metrics)

            if verbose:
                logging.info(f"Fold {fold} Results:")
                for metric, value in metrics.items():
                    logging.info(f"  {metric}: {value:.4f}")

        # 평균 성능 계산
        avg_scores = {}
        for metric in all_scores[0].keys():
            values = [score[metric] for score in all_scores]
            avg_scores[f'{metric}_mean'] = np.mean(values)
            avg_scores[f'{metric}_std'] = np.std(values)

        if verbose:
            logging.info(f"\n{'='*60}")
            logging.info("Cross-Validation Results (Average)")
            logging.info(f"{'='*60}")
            for metric, value in avg_scores.items():
                logging.info(f"  {metric}: {value:.4f}")

        return avg_scores, all_scores


class ExpandingWindowCV:
    """
    확장 윈도우 방식의 교차 검증

    학습 데이터는 계속 누적되고, 테스트 데이터만 이동

    예시:
    Fold 1: Train(2020-01~2020-12) → Test(2021-01~2021-03)
    Fold 2: Train(2020-01~2021-03) → Test(2021-04~2021-06)
    Fold 3: Train(2020-01~2021-06) → Test(2021-07~2021-09)
    """

    def __init__(self, initial_train_size: int, test_size: int, step_size: int = None):
        """
        Args:
            initial_train_size: 초기 학습 데이터 크기
            test_size: 테스트 데이터 크기
            step_size: 윈도우 이동 간격 (None이면 test_size와 동일)
        """
        self.initial_train_size = initial_train_size
        self.test_size = test_size
        self.step_size = step_size if step_size is not None else test_size

    def split(self, X: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """데이터를 확장 윈도우 방식으로 분할"""
        n_samples = len(X)
        splits = []

        current_train_end = self.initial_train_size
        fold = 1

        while current_train_end + self.test_size <= n_samples:
            train_idx = np.arange(0, current_train_end)
            test_idx = np.arange(current_train_end, current_train_end + self.test_size)

            splits.append((train_idx, test_idx))

            logging.info(f"Fold {fold}:")
            logging.info(f"  Train: [0:{current_train_end}] ({len(train_idx)} samples)")
            logging.info(f"  Test:  [{current_train_end}:{current_train_end + self.test_size}] ({len(test_idx)} samples)")

            current_train_end += self.step_size
            fold += 1

        return splits


class RollingWindowCV:
    """
    롤링 윈도우 방식의 교차 검증

    학습 데이터와 테스트 데이터 모두 이동 (고정된 크기 유지)

    예시:
    Fold 1: Train(2020-01~2020-12) → Test(2021-01~2021-03)
    Fold 2: Train(2020-04~2021-03) → Test(2021-04~2021-06)
    Fold 3: Train(2020-07~2021-06) → Test(2021-07~2021-09)
    """

    def __init__(self, train_size: int, test_size: int, step_size: int = None):
        """
        Args:
            train_size: 학습 데이터 크기 (고정)
            test_size: 테스트 데이터 크기
            step_size: 윈도우 이동 간격 (None이면 test_size와 동일)
        """
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size if step_size is not None else test_size

    def split(self, X: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """데이터를 롤링 윈도우 방식으로 분할"""
        n_samples = len(X)
        splits = []

        current_start = 0
        fold = 1

        while current_start + self.train_size + self.test_size <= n_samples:
            train_idx = np.arange(current_start, current_start + self.train_size)
            test_idx = np.arange(
                current_start + self.train_size,
                current_start + self.train_size + self.test_size
            )

            splits.append((train_idx, test_idx))

            logging.info(f"Fold {fold}:")
            logging.info(f"  Train: [{current_start}:{current_start + self.train_size}] ({len(train_idx)} samples)")
            logging.info(f"  Test:  [{current_start + self.train_size}:{current_start + self.train_size + self.test_size}] ({len(test_idx)} samples)")

            current_start += self.step_size
            fold += 1

        return splits
