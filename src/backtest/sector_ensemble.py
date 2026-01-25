"""섹터 기반 Ensemble 전략입니다.

섹터별로 다른 피처/모델을 사용하되, 최종 선택은 전체에서 top N개
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json


class SectorEnsemble:
    """섹터별 앙상블 전략입니다.

    각 섹터마다:
    1. 다른 피처 세트 사용
    2. 다른 모델 파라미터 사용
    3. 섹터별로 수익률 예측

    최종 선택:
    - 모든 섹터의 예측 점수를 하나로 합침
    - 전체에서 top N개 주식 선택
    """

    def __init__(self, sector_col: str = 'sector'):
        """
        Args:
            sector_col: 데이터프레임의 섹터 컬럼명
        """
        self.sector_col = sector_col
        self.sector_configs = {}
        self.sector_models = {}
        self.sector_feature_selectors = {}
        self.trained = False

    def configure_sector(self,
                        sector_name: str,
                        model_class: Any,
                        feature_cols: List[str],
                        model_params: Optional[Dict] = None,
                        feature_selection_params: Optional[Dict] = None):
        """
        섹터별 설정

        Args:
            sector_name: 섹터 이름 (예: 'Technology', 'Financial')
            model_class: 해당 섹터에 사용할 모델 클래스
            feature_cols: 해당 섹터에 중요한 피처 리스트
            model_params: 모델 하이퍼파라미터
            feature_selection_params: 피처 선택 파라미터
        """
        self.sector_configs[sector_name] = {
            'model_class': model_class,
            'feature_cols': feature_cols,
            'model_params': model_params or {},
            'feature_selection_params': feature_selection_params or {}
        }

        logging.info(f"✅ 섹터 설정: {sector_name}")
        logging.info(f"   모델: {model_class.__name__}")
        logging.info(f"   피처 수: {len(feature_cols)}")

    def fit(self,
            df: pd.DataFrame,
            target_col: str,
            use_feature_selection: bool = True):
        """
        섹터별로 모델 학습

        Args:
            df: 전체 데이터 (섹터 정보 포함)
            target_col: 타겟 컬럼명
            use_feature_selection: 피처 선택 사용 여부
        """
        if not self.sector_configs:
            raise ValueError("섹터 설정이 없습니다! configure_sector()를 먼저 호출하세요.")

        logging.info(f"\n{'='*60}")
        logging.info("섹터별 모델 학습 시작")
        logging.info(f"{'='*60}")
        logging.info(f"전체 데이터: {len(df):,} rows")
        logging.info(f"섹터 수: {len(self.sector_configs)}")
        logging.info(f"{'='*60}\n")

        for sector_name, config in self.sector_configs.items():
            logging.info(f"\n{'='*60}")
            logging.info(f"섹터: {sector_name}")
            logging.info(f"{'='*60}")

            # 해당 섹터 데이터만 추출
            sector_df = df[df[self.sector_col] == sector_name].copy()

            if len(sector_df) == 0:
                logging.warning(f"⚠️ {sector_name} 섹터에 데이터가 없습니다. 건너뜁니다.")
                continue

            logging.info(f"섹터 데이터: {len(sector_df):,} rows")

            try:
                # 피처 및 타겟 추출
                X = sector_df[config['feature_cols']]
                y = sector_df[target_col]

                # NaN 제거
                valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
                X = X[valid_mask]
                y = y[valid_mask]

                if len(X) == 0:
                    logging.warning(f"⚠️ {sector_name} 섹터에 유효한 데이터가 없습니다.")
                    continue

                logging.info(f"유효 데이터: {len(X):,} rows")

                # 피처 선택 (선택 사항)
                selected_features = config['feature_cols']
                if use_feature_selection and config.get('feature_selection_params'):
                    from src.training.feature_selector import FeatureSelector

                    selector = FeatureSelector(**config['feature_selection_params'])
                    selected_features = selector.select_features(X, y)
                    X = X[selected_features]

                    self.sector_feature_selectors[sector_name] = selector
                    logging.info(f"피처 선택: {len(config['feature_cols'])} → {len(selected_features)}")

                # 모델 생성 및 학습
                model = config['model_class'](**config['model_params'])
                model.build_model({})

                logging.info(f"모델 학습 중...")
                model.fit(X, y, verbose=0)

                # 모델 저장
                self.sector_models[sector_name] = {
                    'model': model,
                    'selected_features': selected_features,
                    'train_samples': len(X)
                }

                logging.info(f"✅ {sector_name} 섹터 학습 완료")

            except Exception as e:
                logging.error(f"❌ {sector_name} 섹터 학습 실패: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        self.trained = True

        logging.info(f"\n{'='*60}")
        logging.info(f"섹터별 모델 학습 완료")
        logging.info(f"학습된 섹터 수: {len(self.sector_models)}")
        logging.info(f"{'='*60}\n")

    def predict_by_sector(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        섹터별로 예측 수행

        Args:
            df: 예측할 데이터 (섹터 정보 포함)

        Returns:
            예측 점수가 추가된 데이터프레임
        """
        if not self.trained:
            raise ValueError("모델이 학습되지 않았습니다! fit()을 먼저 호출하세요.")

        logging.info(f"\n{'='*60}")
        logging.info("섹터별 예측 시작")
        logging.info(f"{'='*60}")

        result_df = df.copy()
        result_df['predicted_score'] = np.nan
        result_df['predicted_class'] = np.nan

        for sector_name, model_info in self.sector_models.items():
            logging.info(f"\n예측 중: {sector_name}")

            # 해당 섹터 데이터만 추출
            sector_mask = result_df[self.sector_col] == sector_name
            sector_df = result_df[sector_mask]

            if len(sector_df) == 0:
                logging.info(f"  {sector_name} 섹터 데이터 없음")
                continue

            try:
                # 피처 추출
                X = sector_df[model_info['selected_features']]

                # NaN 제거
                valid_mask = ~X.isnull().any(axis=1)
                X_valid = X[valid_mask]

                if len(X_valid) == 0:
                    logging.warning(f"  ⚠️ {sector_name} 섹터에 유효한 데이터가 없습니다.")
                    continue

                # 예측
                model = model_info['model']

                # 예측 클래스
                y_pred_class = model.predict(X_valid)

                # 예측 확률 (점수)
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_valid)[:, 1]  # 양성 클래스 확률
                else:
                    y_pred_proba = y_pred_class  # 확률이 없으면 클래스 사용

                # 결과 저장
                valid_indices = sector_df[valid_mask].index
                result_df.loc[valid_indices, 'predicted_score'] = y_pred_proba
                result_df.loc[valid_indices, 'predicted_class'] = y_pred_class

                logging.info(f"  예측 완료: {len(X_valid):,} rows")
                logging.info(f"  평균 점수: {y_pred_proba.mean():.4f}")

            except Exception as e:
                logging.error(f"  ❌ {sector_name} 섹터 예측 실패: {str(e)}")
                continue

        logging.info(f"\n{'='*60}")
        logging.info(f"섹터별 예측 완료")
        logging.info(f"총 예측: {result_df['predicted_score'].notna().sum():,} rows")
        logging.info(f"{'='*60}\n")

        return result_df

    def select_top_stocks(self,
                         df: pd.DataFrame,
                         top_k: int = 5,
                         symbol_col: str = 'symbol') -> pd.DataFrame:
        """
        전체 주식에서 top K개 선택

        Args:
            df: 예측 점수가 포함된 데이터프레임
            top_k: 선택할 주식 개수
            symbol_col: 심볼 컬럼명

        Returns:
            선택된 주식 데이터프레임
        """
        if 'predicted_score' not in df.columns:
            raise ValueError("predicted_score 컬럼이 없습니다! predict_by_sector()를 먼저 호출하세요.")

        logging.info(f"\n{'='*60}")
        logging.info(f"Top {top_k}개 주식 선택")
        logging.info(f"{'='*60}")

        # 예측 점수가 있는 주식만
        valid_df = df[df['predicted_score'].notna()].copy()

        if len(valid_df) == 0:
            logging.error("예측 점수가 있는 주식이 없습니다!")
            return pd.DataFrame()

        # 예측 점수 기준으로 정렬
        sorted_df = valid_df.sort_values('predicted_score', ascending=False)

        # Top K 선택
        top_stocks = sorted_df.head(top_k)

        logging.info(f"\n선택된 주식:")
        for idx, row in top_stocks.iterrows():
            logging.info(
                f"  {row[symbol_col]:<10} | {row[self.sector_col]:<15} | "
                f"Score: {row['predicted_score']:.4f}"
            )

        # 섹터별 분포
        sector_counts = top_stocks[self.sector_col].value_counts()
        logging.info(f"\n섹터별 분포:")
        for sector, count in sector_counts.items():
            logging.info(f"  {sector}: {count}개")

        logging.info(f"\n{'='*60}\n")

        return top_stocks

    def fit_predict_select(self,
                          train_df: pd.DataFrame,
                          predict_df: pd.DataFrame,
                          target_col: str,
                          top_k: int = 5,
                          use_feature_selection: bool = True,
                          symbol_col: str = 'symbol') -> pd.DataFrame:
        """
        학습 → 예측 → 선택을 한번에 수행

        Args:
            train_df: 학습 데이터
            predict_df: 예측할 데이터
            target_col: 타겟 컬럼명
            top_k: 선택할 주식 개수
            use_feature_selection: 피처 선택 사용 여부
            symbol_col: 심볼 컬럼명

        Returns:
            선택된 주식 데이터프레임
        """
        # 1. 학습
        self.fit(train_df, target_col, use_feature_selection)

        # 2. 예측
        predicted_df = self.predict_by_sector(predict_df)

        # 3. 선택
        top_stocks = self.select_top_stocks(predicted_df, top_k, symbol_col)

        return top_stocks

    def save_config(self, path: str):
        """섹터 설정 저장"""
        config_data = {}

        for sector_name, config in self.sector_configs.items():
            config_data[sector_name] = {
                'model_class': config['model_class'].__name__,
                'feature_cols': config['feature_cols'],
                'model_params': config['model_params'],
                'feature_selection_params': config['feature_selection_params']
            }

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(config_data, f, indent=2)

        logging.info(f"💾 섹터 설정 저장: {output_path}")

    def get_sector_summary(self) -> pd.DataFrame:
        """섹터별 모델 정보 요약"""
        if not self.sector_models:
            return pd.DataFrame()

        summary = []
        for sector_name, model_info in self.sector_models.items():
            summary.append({
                'sector': sector_name,
                'num_features': len(model_info['selected_features']),
                'train_samples': model_info['train_samples'],
                'model_type': type(model_info['model']).__name__
            })

        return pd.DataFrame(summary)


def create_default_sector_configs(model_class: Any) -> Dict[str, Dict]:
    """
    기본 섹터별 설정 생성 예제

    Args:
        model_class: 사용할 모델 클래스

    Returns:
        섹터별 설정 딕셔너리
    """
    configs = {
        'Technology': {
            'model_class': model_class,
            'feature_cols': [
                'revenue', 'netIncome', 'researchAndDevelopmentExpenses',
                'operatingCashFlow', 'freeCashFlow',
                'OverMC_researchAndDevelopmentExpenses',  # R&D 비중 중요
                'OverMC_revenue', 'OverMC_netIncome',
                'priceToBookRatio', 'priceToSalesRatio'
            ],
            'model_params': {'n_estimators': 100, 'max_depth': 6},
            'feature_selection_params': {'method': 'tree_importance', 'top_k': 8}
        },
        'Financial': {
            'model_class': model_class,
            'feature_cols': [
                'totalAssets', 'totalLiabilities', 'totalEquity',
                'netIncome', 'operatingIncome',
                'debtToEquity',  # 부채비율 중요
                'returnOnEquity', 'returnOnAssets',
                'priceToBookRatio', 'dividendYield'
            ],
            'model_params': {'n_estimators': 100, 'max_depth': 5},
            'feature_selection_params': {'method': 'mutual_info', 'top_k': 8}
        },
        'Healthcare': {
            'model_class': model_class,
            'feature_cols': [
                'revenue', 'grossProfit', 'netIncome',
                'researchAndDevelopmentExpenses',
                'operatingCashFlow', 'freeCashFlow',
                'OverMC_researchAndDevelopmentExpenses',  # R&D 비중 중요
                'grossProfitRatio', 'operatingIncomeRatio',
                'priceToSalesRatio'
            ],
            'model_params': {'n_estimators': 100, 'max_depth': 6},
            'feature_selection_params': {'method': 'tree_importance', 'top_k': 8}
        },
        'Consumer': {
            'model_class': model_class,
            'feature_cols': [
                'revenue', 'grossProfit', 'netIncome',
                'inventoryTurnover',  # 재고회전율 중요
                'operatingCashFlow', 'freeCashFlow',
                'OverMC_revenue', 'OverMC_netIncome',
                'returnOnEquity', 'priceToEarningsRatio'
            ],
            'model_params': {'n_estimators': 100, 'max_depth': 5},
            'feature_selection_params': {'method': 'mutual_info', 'top_k': 8}
        },
        'Industrial': {
            'model_class': model_class,
            'feature_cols': [
                'revenue', 'grossProfit', 'operatingIncome',
                'totalAssets', 'propertyPlantEquipmentNet',  # 고정자산 중요
                'operatingCashFlow', 'capitalExpenditure',
                'OverMC_revenue', 'assetTurnover',
                'returnOnAssets'
            ],
            'model_params': {'n_estimators': 100, 'max_depth': 5},
            'feature_selection_params': {'method': 'tree_importance', 'top_k': 8}
        }
    }

    return configs
