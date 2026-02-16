"""
Feature 선택

피처 선택 및 과적합 방지를 수행하는 모듈입니다.
"""

import numpy as np
import pandas as pd
import logging
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression,
    mutual_info_classif, mutual_info_regression,
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from typing import List, Optional, Union


class FeatureSelector:
    """
    피처 선택 및 과적합 방지를 수행합니다.

    방법:
    1. 단변량 선택 (통계적 테스트)
    2. 재귀적 feature 제거 (RFE)
    3. Tree 기반 Feature Importance
    4. 상관관계 기반 중복 제거
    """

    def __init__(self,
                 method: str = 'mutual_info',
                 top_k: int = 50,
                 correlation_threshold: float = 0.95,
                 task: str = 'classification'):
        """
        Args:
            method: 피처 선택 방법
                   - 'mutual_info': 상호 정보량
                   - 'f_test': F-검정 (ANOVA)
                   - 'rfe': 재귀적 feature 제거
                   - 'tree_importance': Tree 기반 Feature Importance
            top_k: 선택할 피처 개수
            correlation_threshold: 상관관계 임계값 (이 값 이상이면 중복으로 간주)
            task: 'classification' 또는 'regression'
        """
        self.method = method
        self.top_k = top_k
        self.correlation_threshold = correlation_threshold
        self.task = task
        self.selected_features = None
        self.feature_scores = None

    def select_features(self,
                       X: Union[np.ndarray, pd.DataFrame],
                       y: Union[np.ndarray, pd.Series],
                       feature_names: Optional[List[str]] = None) -> List[str]:
        """
        피처 선택을 수행합니다.

        Args:
            X: feature 데이터
            y: 타겟 데이터
            feature_names: 피처 이름 리스트

        Returns:
            선택된 피처 이름 리스트
        """
        # DataFrame을 array로 변환
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = X.columns.tolist()
            X = X.values

        if isinstance(y, pd.Series):
            y = y.values

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        logging.info(f"{'='*60}")
        logging.info(f"피처 선택 시작")
        logging.info(f"{'='*60}")
        logging.info(f"원본 피처 수: {X.shape[1]}")
        logging.info(f"방법: {self.method}")
        logging.info(f"목표 피처 수: {self.top_k}")

        # 1단계: 상관관계 높은 피처 제거
        X_reduced, remaining_features = self._remove_correlated_features(
            pd.DataFrame(X, columns=feature_names)
        )

        logging.info(f"상관관계 제거 후: {len(remaining_features)}")

        # 2단계: 중요도 기반 선택
        if self.method == 'mutual_info':
            selected = self._select_by_mutual_info(X_reduced, y, remaining_features)
        elif self.method == 'f_test':
            selected = self._select_by_f_test(X_reduced, y, remaining_features)
        elif self.method == 'rfe':
            selected = self._select_by_rfe(X_reduced, y, remaining_features)
        elif self.method == 'tree_importance':
            selected = self._select_by_tree_importance(X_reduced, y, remaining_features)
        else:
            logging.warning(f"Unknown method '{self.method}', using first {self.top_k} features")
            selected = remaining_features[:self.top_k]

        self.selected_features = selected

        logging.info(f"✅ 최종 선택된 피처 수: {len(selected)}")
        logging.info(f"{'='*60}\n")

        return selected

    def _remove_correlated_features(self, df: pd.DataFrame) -> tuple:
        """
        상관관계가 높은 피처를 제거합니다.

        Args:
            df: 피처 DataFrame

        Returns:
            (축소된 데이터, 남은 피처 이름)
        """
        # 상관관계 행렬 계산
        corr_matrix = df.corr().abs()

        # 상삼각 행렬 (중복 제거)
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # 상관관계 높은 컬럼 찾기
        to_drop = [
            column for column in upper.columns
            if any(upper[column] > self.correlation_threshold)
        ]

        df_reduced = df.drop(columns=to_drop)

        logging.info(f"상관관계로 제거된 피처: {len(to_drop)}")
        if to_drop and len(to_drop) < 20:
            logging.debug(f"제거된 피처: {to_drop}")

        return df_reduced.values, df_reduced.columns.tolist()

    def _select_by_mutual_info(self,
                               X: np.ndarray,
                               y: np.ndarray,
                               feature_names: List[str]) -> List[str]:
        """
        상호 정보량(Mutual Information) 기반으로 피처를 선택합니다.

        Args:
            X: feature 데이터
            y: 타겟 데이터
            feature_names: 피처 이름

        Returns:
            선택된 피처 이름
        """
        k = min(self.top_k, X.shape[1])

        if self.task == 'classification':
            selector = SelectKBest(mutual_info_classif, k=k)
        else:
            selector = SelectKBest(mutual_info_regression, k=k)

        selector.fit(X, y)

        # 점수 저장
        self.feature_scores = pd.DataFrame({
            'feature': feature_names,
            'score': selector.scores_
        }).sort_values('score', ascending=False)

        selected_idx = selector.get_support(indices=True)
        selected = [feature_names[i] for i in selected_idx]

        logging.info(f"Mutual Information 상위 10개 피처:")
        for i, row in self.feature_scores.head(10).iterrows():
            logging.info(f"   {row['feature']}: {row['score']:.4f}")

        return selected

    def _select_by_f_test(self,
                         X: np.ndarray,
                         y: np.ndarray,
                         feature_names: List[str]) -> List[str]:
        """
        F-검정 기반으로 피처를 선택합니다.

        Args:
            X: feature 데이터
            y: 타겟 데이터
            feature_names: 피처 이름

        Returns:
            선택된 피처 이름
        """
        k = min(self.top_k, X.shape[1])

        if self.task == 'classification':
            selector = SelectKBest(f_classif, k=k)
        else:
            selector = SelectKBest(f_regression, k=k)

        selector.fit(X, y)

        # 점수 저장
        self.feature_scores = pd.DataFrame({
            'feature': feature_names,
            'score': selector.scores_
        }).sort_values('score', ascending=False)

        selected_idx = selector.get_support(indices=True)
        selected = [feature_names[i] for i in selected_idx]

        logging.info(f"F-test 상위 10개 피처:")
        for i, row in self.feature_scores.head(10).iterrows():
            logging.info(f"   {row['feature']}: {row['score']:.4f}")

        return selected

    def _select_by_rfe(self,
                      X: np.ndarray,
                      y: np.ndarray,
                      feature_names: List[str]) -> List[str]:
        """
        재귀적 feature 제거(RFE) 방식으로 피처를 선택합니다.

        Args:
            X: feature 데이터
            y: 타겟 데이터
            feature_names: 피처 이름

        Returns:
            선택된 피처 이름
        """
        k = min(self.top_k, X.shape[1])

        if self.task == 'classification':
            estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            estimator = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

        selector = RFE(estimator, n_features_to_select=k, step=max(1, X.shape[1] // 20))
        selector.fit(X, y)

        # 순위 저장
        self.feature_scores = pd.DataFrame({
            'feature': feature_names,
            'rank': selector.ranking_
        }).sort_values('rank')

        selected_idx = selector.get_support(indices=True)
        selected = [feature_names[i] for i in selected_idx]

        logging.info(f"RFE 상위 10개 피처:")
        for i, row in self.feature_scores.head(10).iterrows():
            logging.info(f"   {row['feature']}: rank {row['rank']}")

        return selected

    def _select_by_tree_importance(self,
                                   X: np.ndarray,
                                   y: np.ndarray,
                                   feature_names: List[str]) -> List[str]:
        """
        Tree 기반 중요도로 피처를 선택합니다.

        Args:
            X: feature 데이터
            y: 타겟 데이터
            feature_names: 피처 이름

        Returns:
            선택된 피처 이름
        """
        if self.task == 'classification':
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

        rf.fit(X, y)

        importances = rf.feature_importances_

        # 중요도 저장
        self.feature_scores = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        # 상위 k개 선택
        k = min(self.top_k, X.shape[1])
        selected = self.feature_scores.head(k)['feature'].tolist()

        logging.info(f"Tree Importance 상위 10개 피처:")
        for i, row in self.feature_scores.head(10).iterrows():
            logging.info(f"   {row['feature']}: {row['importance']:.4f}")

        return selected

    def transform(self,
                 X: Union[np.ndarray, pd.DataFrame],
                 feature_names: Optional[List[str]] = None) -> Union[np.ndarray, pd.DataFrame]:
        """
        선택된 피처만 추출합니다.

        Args:
            X: feature 데이터
            feature_names: 피처 이름 리스트

        Returns:
            선택된 피처만 포함된 데이터
        """
        if self.selected_features is None:
            raise ValueError("먼저 select_features()를 호출하세요")

        if isinstance(X, pd.DataFrame):
            return X[self.selected_features]
        else:
            if feature_names is None:
                # 인덱스로 선택
                return X[:, :len(self.selected_features)]
            else:
                # 이름으로 선택
                df = pd.DataFrame(X, columns=feature_names)
                return df[self.selected_features].values

    def fit_transform(self,
                     X: Union[np.ndarray, pd.DataFrame],
                     y: Union[np.ndarray, pd.Series],
                     feature_names: Optional[List[str]] = None) -> Union[np.ndarray, pd.DataFrame]:
        """
        피처 선택과 변환을 한번에 수행합니다.

        Args:
            X: feature 데이터
            y: 타겟 데이터
            feature_names: 피처 이름 리스트

        Returns:
            선택된 피처만 포함된 데이터
        """
        self.select_features(X, y, feature_names)
        return self.transform(X, feature_names)

    def get_feature_scores(self) -> pd.DataFrame:
        """피처 점수/순위를 반환합니다."""
        if self.feature_scores is None:
            raise ValueError("먼저 select_features()를 호출하세요")
        return self.feature_scores

    def save_selected_features(self, path: str):
        """선택된 피처를 저장합니다."""
        if self.selected_features is None:
            raise ValueError("먼저 select_features()를 호출하세요")

        df = pd.DataFrame({'feature': self.selected_features})
        df.to_csv(path, index=False)
        logging.info(f"💾 선택된 피처 저장: {path}")

    def load_selected_features(self, path: str):
        """선택된 피처를 로드합니다."""
        df = pd.read_csv(path)
        self.selected_features = df['feature'].tolist()
        logging.info(f"📂 선택된 피처 로드: {path} ({len(self.selected_features)}개)")
