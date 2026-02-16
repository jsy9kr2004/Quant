"""
데이터 품질 보고서 및 검증 모듈입니다.

이 모듈은 다음 기능을 제공합니다:
1. 상세 데이터 품질 보고서 (NaN, Infinite, 분포 분석)
2. 데이터 클리닝 효과 검증을 위한 A/B 테스트
3. 모든 데이터 클리닝 작업에 대한 강화된 로깅

사용 예시:
    # 품질 보고서 생성
    report = DataQualityReport(X, y, config, logger)
    report.generate()
    report.save('outputs/data_quality_report.xlsx')

    # 클리닝 효과 A/B 테스트
    validator = DataCleaningValidator(X, y, config, logger)
    results = validator.validate_cleaning_effect()

Author: Claude Code Analysis
Created: 2026-01-17
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
import os


class DataQualityReport:
    """
    종합적인 데이터 품질 보고서를 생성하는 클래스입니다.

    보고서 내용:
    - 컬럼별 NaN 분석 및 원인
    - Infinite 값 분석 및 근본 원인
    - 분포 통계
    - 제거 영향 요약
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        config: Optional[Dict] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.X = X.copy()
        self.y = y.copy() if y is not None else None
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)

        # 보고서 데이터 저장소
        self.report_data = {
            'summary': {},
            'nan_analysis': {},
            'infinite_analysis': {},
            'distribution': {},
            'recommendations': []
        }

        self.generated = False

    def generate(self) -> Dict[str, Any]:
        """종합적인 데이터 품질 보고서를 생성합니다."""
        self.logger.info("=" * 60)
        self.logger.info("📊 DATA QUALITY REPORT")
        self.logger.info("=" * 60)

        self._analyze_basic_stats()
        self._analyze_nan_values()
        self._analyze_infinite_values()
        self._analyze_distribution()
        self._generate_recommendations()

        self.generated = True
        self._log_summary()

        return self.report_data

    def _analyze_basic_stats(self):
        """기본 데이터셋 통계를 분석합니다."""
        self.report_data['summary'] = {
            'timestamp': datetime.now().isoformat(),
            'total_rows': len(self.X),
            'total_columns': len(self.X.columns),
            'numeric_columns': len([c for c in self.X.columns
                                   if pd.api.types.is_numeric_dtype(self.X[c])]),
            'object_columns': len([c for c in self.X.columns
                                  if not pd.api.types.is_numeric_dtype(self.X[c])]),
            'memory_usage_mb': self.X.memory_usage(deep=True).sum() / 1024 / 1024
        }

        if self.y is not None:
            self.report_data['summary']['target_nan_count'] = int(self.y.isna().sum())
            self.report_data['summary']['target_nan_pct'] = float(self.y.isna().mean() * 100)

    def _analyze_nan_values(self):
        """상세 NaN 분석을 수행합니다."""
        nan_analysis = {
            'total_nan_cells': int(self.X.isna().sum().sum()),
            'total_cells': int(self.X.size),
            'nan_cell_pct': float(self.X.isna().sum().sum() / self.X.size * 100),
            'rows_with_nan': int(self.X.isna().any(axis=1).sum()),
            'rows_with_nan_pct': float(self.X.isna().any(axis=1).mean() * 100),
            'columns_with_nan': int((self.X.isna().sum() > 0).sum()),
            'columns_by_nan_ratio': {}
        }

        # 컬럼 수준 분석
        nan_ratio_per_col = self.X.isna().mean()

        # NaN 비율로 컬럼 분류
        nan_analysis['columns_by_nan_ratio'] = {
            'no_nan (0%)': int((nan_ratio_per_col == 0).sum()),
            'low_nan (0-10%)': int(((nan_ratio_per_col > 0) & (nan_ratio_per_col <= 0.1)).sum()),
            'medium_nan (10-50%)': int(((nan_ratio_per_col > 0.1) & (nan_ratio_per_col <= 0.5)).sum()),
            'high_nan (50-90%)': int(((nan_ratio_per_col > 0.5) & (nan_ratio_per_col <= 0.9)).sum()),
            'very_high_nan (>90%)': int((nan_ratio_per_col > 0.9).sum())
        }

        # NaN 비율 상위 컬럼
        top_nan_cols = nan_ratio_per_col.nlargest(20)
        nan_analysis['top_nan_columns'] = {
            col: {'nan_pct': float(pct * 100), 'nan_count': int(self.X[col].isna().sum())}
            for col, pct in top_nan_cols.items() if pct > 0
        }

        self.report_data['nan_analysis'] = nan_analysis

    def _analyze_infinite_values(self):
        """상세 Infinite 값 분석을 수행합니다."""
        numeric_cols = [c for c in self.X.columns
                       if pd.api.types.is_numeric_dtype(self.X[c])]

        if not numeric_cols:
            self.report_data['infinite_analysis'] = {'no_numeric_columns': True}
            return

        X_numeric = self.X[numeric_cols]
        inf_mask = np.isinf(X_numeric)

        inf_analysis = {
            'total_inf_cells': int(inf_mask.sum().sum()),
            'total_numeric_cells': int(X_numeric.size),
            'inf_cell_pct': float(inf_mask.sum().sum() / X_numeric.size * 100) if X_numeric.size > 0 else 0,
            'rows_with_inf': int(inf_mask.any(axis=1).sum()),
            'rows_with_inf_pct': float(inf_mask.any(axis=1).mean() * 100),
            'columns_with_inf': int((inf_mask.sum() > 0).sum()),
            'inf_columns_detail': {}
        }

        # 컬럼 수준 Infinite 분석
        inf_count_per_col = inf_mask.sum()
        inf_cols = inf_count_per_col[inf_count_per_col > 0]

        for col in inf_cols.index:
            col_data = X_numeric[col]
            pos_inf = np.sum(np.isposinf(col_data))
            neg_inf = np.sum(np.isneginf(col_data))

            # 원인 식별 시도
            cause = self._identify_inf_cause(col)

            inf_analysis['inf_columns_detail'][col] = {
                'total_inf': int(inf_cols[col]),
                'positive_inf': int(pos_inf),
                'negative_inf': int(neg_inf),
                'probable_cause': cause
            }

        self.report_data['infinite_analysis'] = inf_analysis

    def _identify_inf_cause(self, col_name: str) -> str:
        """Infinite 값의 추정 원인을 식별합니다."""
        col_lower = col_name.lower()

        if 'per' in col_lower or 'pe_' in col_lower or '_pe' in col_lower:
            return "Division by EPS (EPS=0)"
        elif 'pb' in col_lower or 'pbr' in col_lower:
            return "Division by Book Value (BV=0)"
        elif 'roe' in col_lower or 'roa' in col_lower:
            return "Division by Equity/Assets (=0)"
        elif 'ratio' in col_lower:
            return "Division by denominator (=0)"
        elif 'log' in col_lower:
            return "Log of zero or negative"
        elif 'growth' in col_lower or 'change' in col_lower:
            return "Division by previous value (=0)"
        else:
            return "Unknown (likely division by zero)"

    def _analyze_distribution(self):
        """이상치 탐지를 위한 값 분포를 분석합니다."""
        numeric_cols = [c for c in self.X.columns
                       if pd.api.types.is_numeric_dtype(self.X[c])]

        if not numeric_cols:
            self.report_data['distribution'] = {'no_numeric_columns': True}
            return

        dist_analysis = {
            'extreme_value_columns': [],
            'same_value_columns': [],
            'statistics': {}
        }

        for col in numeric_cols[:100]:  # 성능을 위해 처음 100개 컬럼으로 제한
            col_data = self.X[col].dropna()
            if len(col_data) == 0:
                continue

            # 극단값 확인
            max_val = col_data.abs().max()
            if max_val > 1e10:
                dist_analysis['extreme_value_columns'].append({
                    'column': col,
                    'max_abs_value': float(max_val),
                    'recommendation': 'Consider log transformation'
                })

            # 단일 값 컬럼 확인
            value_counts = col_data.value_counts(normalize=True)
            if len(value_counts) > 0 and value_counts.iloc[0] > 0.95:
                dist_analysis['same_value_columns'].append({
                    'column': col,
                    'top_value': str(value_counts.index[0]),
                    'top_value_pct': float(value_counts.iloc[0] * 100)
                })

        # 요약 통계
        dist_analysis['statistics'] = {
            'extreme_value_columns_count': len(dist_analysis['extreme_value_columns']),
            'same_value_columns_count': len(dist_analysis['same_value_columns'])
        }

        self.report_data['distribution'] = dist_analysis

    def _generate_recommendations(self):
        """분석 결과를 기반으로 실행 가능한 권고사항을 생성합니다."""
        recommendations = []

        nan_analysis = self.report_data.get('nan_analysis', {})
        inf_analysis = self.report_data.get('infinite_analysis', {})
        dist_analysis = self.report_data.get('distribution', {})

        # NaN 관련 권고
        very_high_nan = nan_analysis.get('columns_by_nan_ratio', {}).get('very_high_nan (>90%)', 0)
        if very_high_nan > 0:
            recommendations.append({
                'severity': 'HIGH',
                'category': 'NaN',
                'message': f'{very_high_nan} columns have >90% NaN - consider dropping',
                'action': 'Review DROP_SPARSE_COLS threshold'
            })

        rows_with_nan_pct = nan_analysis.get('rows_with_nan_pct', 0)
        if rows_with_nan_pct > 50:
            recommendations.append({
                'severity': 'HIGH',
                'category': 'NaN',
                'message': f'{rows_with_nan_pct:.1f}% rows have NaN - significant data loss risk',
                'action': 'Consider imputation instead of removal'
            })

        # Infinite 관련 권고
        rows_with_inf = inf_analysis.get('rows_with_inf', 0)
        if rows_with_inf > 0:
            recommendations.append({
                'severity': 'MEDIUM',
                'category': 'Infinite',
                'message': f'{rows_with_inf} rows have infinite values',
                'action': 'Review ratio calculations (division by zero)'
            })

        # 분포 관련 권고
        extreme_count = dist_analysis.get('statistics', {}).get('extreme_value_columns_count', 0)
        if extreme_count > 10:
            recommendations.append({
                'severity': 'MEDIUM',
                'category': 'Distribution',
                'message': f'{extreme_count} columns have extreme values (>1e10)',
                'action': 'Verify log transformation is applied'
            })

        same_value_count = dist_analysis.get('statistics', {}).get('same_value_columns_count', 0)
        if same_value_count > 10:
            recommendations.append({
                'severity': 'LOW',
                'category': 'Distribution',
                'message': f'{same_value_count} columns have >95% same value',
                'action': 'Consider dropping uninformative columns'
            })

        if not recommendations:
            recommendations.append({
                'severity': 'INFO',
                'category': 'General',
                'message': 'No major data quality issues detected',
                'action': 'Continue with current preprocessing'
            })

        self.report_data['recommendations'] = recommendations

    def _log_summary(self):
        """요약 결과를 콘솔/파일에 로깅합니다."""
        summary = self.report_data['summary']
        nan = self.report_data['nan_analysis']
        inf = self.report_data['infinite_analysis']

        self.logger.info("")
        self.logger.info(f"📊 Dataset: {summary['total_rows']:,} rows × {summary['total_columns']:,} columns")
        self.logger.info(f"   Memory usage: {summary['memory_usage_mb']:.1f} MB")
        self.logger.info("")

        self.logger.info(f"🔍 NaN Analysis:")
        self.logger.info(f"   Total NaN cells: {nan['total_nan_cells']:,} ({nan['nan_cell_pct']:.2f}%)")
        self.logger.info(f"   Rows with NaN: {nan['rows_with_nan']:,} ({nan['rows_with_nan_pct']:.1f}%)")
        self.logger.info(f"   Columns with NaN: {nan['columns_with_nan']:,}")

        for category, count in nan['columns_by_nan_ratio'].items():
            if count > 0:
                self.logger.info(f"      - {category}: {count}")
        self.logger.info("")

        if not inf.get('no_numeric_columns', False):
            self.logger.info(f"♾️  Infinite Analysis:")
            self.logger.info(f"   Total Inf cells: {inf['total_inf_cells']:,} ({inf['inf_cell_pct']:.4f}%)")
            self.logger.info(f"   Rows with Inf: {inf['rows_with_inf']:,} ({inf['rows_with_inf_pct']:.2f}%)")

            if inf['inf_columns_detail']:
                self.logger.info(f"   Affected columns:")
                for col, detail in list(inf['inf_columns_detail'].items())[:5]:
                    self.logger.info(f"      - {col}: {detail['total_inf']} ({detail['probable_cause']})")
        self.logger.info("")

        self.logger.info("📝 Recommendations:")
        for rec in self.report_data['recommendations']:
            severity_icon = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢', 'INFO': 'ℹ️'}.get(rec['severity'], '•')
            self.logger.info(f"   {severity_icon} [{rec['category']}] {rec['message']}")
            self.logger.info(f"      → {rec['action']}")

        self.logger.info("=" * 60)

    def save(self, filepath: str):
        """보고서를 Excel 파일로 저장합니다."""
        if not self.generated:
            self.generate()

        # 디렉토리 존재 확인
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # 요약 시트
            summary_df = pd.DataFrame([self.report_data['summary']]).T
            summary_df.columns = ['Value']
            summary_df.to_excel(writer, sheet_name='Summary')

            # NaN 분석 시트
            nan_df = pd.DataFrame([
                {'metric': k, 'value': v}
                for k, v in self.report_data['nan_analysis'].items()
                if not isinstance(v, dict)
            ])
            nan_df.to_excel(writer, sheet_name='NaN_Analysis', index=False)

            # NaN 상위 컬럼
            top_nan = self.report_data['nan_analysis'].get('top_nan_columns', {})
            if top_nan:
                top_nan_df = pd.DataFrame([
                    {'column': k, **v} for k, v in top_nan.items()
                ])
                top_nan_df.to_excel(writer, sheet_name='Top_NaN_Columns', index=False)

            # Infinite 분석 시트
            if not self.report_data['infinite_analysis'].get('no_numeric_columns', False):
                inf_cols = self.report_data['infinite_analysis'].get('inf_columns_detail', {})
                if inf_cols:
                    inf_df = pd.DataFrame([
                        {'column': k, **v} for k, v in inf_cols.items()
                    ])
                    inf_df.to_excel(writer, sheet_name='Infinite_Columns', index=False)

            # 권고사항 시트
            rec_df = pd.DataFrame(self.report_data['recommendations'])
            rec_df.to_excel(writer, sheet_name='Recommendations', index=False)

        self.logger.info(f"📄 Report saved to: {filepath}")

    def get_cleaning_stats(self) -> Dict[str, Any]:
        """데이터 클리닝 영향 추정을 위한 통계를 반환합니다."""
        if not self.generated:
            self.generate()

        nan = self.report_data['nan_analysis']
        inf = self.report_data['infinite_analysis']

        return {
            'rows_with_nan': nan.get('rows_with_nan', 0),
            'rows_with_nan_pct': nan.get('rows_with_nan_pct', 0),
            'rows_with_inf': inf.get('rows_with_inf', 0) if not inf.get('no_numeric_columns', False) else 0,
            'rows_with_inf_pct': inf.get('rows_with_inf_pct', 0) if not inf.get('no_numeric_columns', False) else 0,
            'estimated_removal_pct': min(
                nan.get('rows_with_nan_pct', 0) +
                (inf.get('rows_with_inf_pct', 0) if not inf.get('no_numeric_columns', False) else 0),
                100
            )
        }


class DataCleaningValidator:
    """
    A/B 테스트를 통해 데이터 클리닝 효과를 검증하는 클래스입니다.

    모델 성능을 비교합니다:
    - Case A: 비적극적 클리닝 (NaN/Inf를 중앙값/0으로 대체)
    - Case B: 현재 방식의 클리닝 (NaN/Inf가 있는 행 제거)
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        config: Optional[Dict] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.X_original = X.copy()
        self.y_original = y.copy()
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)

        self.validation_results = None

    def validate_cleaning_effect(self) -> Dict[str, Any]:
        """
        클리닝 전략을 비교하는 A/B 테스트를 실행합니다.

        Returns:
            다음 키를 포함하는 딕셔너리:
            - case_a_stats: Imputation 기반 클리닝 통계
            - case_b_stats: 제거 기반 클리닝 통계
            - comparison: 성능 비교
            - recommendation: 더 나은 접근 방식 권고
        """
        self.logger.info("=" * 60)
        self.logger.info("🧪 DATA CLEANING A/B TEST")
        self.logger.info("=" * 60)

        # Case A: Imputation 기반 (중앙값/0으로 대체)
        self.logger.info("\n📊 Case A: Imputation-based cleaning")
        X_a, y_a, stats_a = self._clean_with_imputation()

        # Case B: 제거 기반 (현재 방식)
        self.logger.info("\n📊 Case B: Removal-based cleaning")
        X_b, y_b, stats_b = self._clean_with_removal()

        # 비교
        comparison = self._compare_approaches(stats_a, stats_b)

        self.validation_results = {
            'case_a_imputation': {
                'stats': stats_a,
                'rows_after': len(X_a),
                'cols_after': len(X_a.columns)
            },
            'case_b_removal': {
                'stats': stats_b,
                'rows_after': len(X_b),
                'cols_after': len(X_b.columns)
            },
            'comparison': comparison,
            'recommendation': self._generate_recommendation(stats_a, stats_b, comparison)
        }

        self._log_results()

        return self.validation_results

    def _clean_with_imputation(self) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        """Imputation을 사용하여 데이터를 정제합니다 (NaN/Inf 대체)."""
        X = self.X_original.copy()
        y = self.y_original.copy()

        stats = {
            'method': 'imputation',
            'rows_before': len(X),
            'cols_before': len(X.columns)
        }

        # 수치형 컬럼만 추출
        numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]

        # 먼저 Infinite를 NaN으로 대체
        for col in numeric_cols:
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)

        # Imputation 전 NaN 수 카운트
        nan_before = X[numeric_cols].isna().sum().sum()
        stats['nan_cells_imputed'] = int(nan_before)

        # 중앙값으로 Imputation
        for col in numeric_cols:
            median_val = X[col].median()
            if pd.isna(median_val):
                median_val = 0
            X[col] = X[col].fillna(median_val)

        # 타겟(y)의 NaN 처리 (제거 필수, 타겟은 imputation 불가)
        y_nan_mask = y.isna()
        if y_nan_mask.sum() > 0:
            X = X[~y_nan_mask]
            y = y[~y_nan_mask]
            stats['y_nan_removed'] = int(y_nan_mask.sum())

        stats['rows_after'] = len(X)
        stats['cols_after'] = len(X.columns)
        stats['rows_kept_pct'] = float(len(X) / stats['rows_before'] * 100)

        self.logger.info(f"   Rows: {stats['rows_before']:,} → {stats['rows_after']:,} ({stats['rows_kept_pct']:.1f}% kept)")
        self.logger.info(f"   NaN cells imputed: {stats['nan_cells_imputed']:,}")

        return X, y, stats

    def _clean_with_removal(self) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        """제거 방식으로 데이터를 정제합니다 (현재 사용 중인 방식)."""
        X = self.X_original.copy()
        y = self.y_original.copy()

        stats = {
            'method': 'removal',
            'rows_before': len(X),
            'cols_before': len(X.columns)
        }

        # 수치형 컬럼만 추출
        numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]

        # Infinite가 있는 행 제거
        inf_mask = np.isinf(X[numeric_cols]).any(axis=1)
        stats['inf_rows_removed'] = int(inf_mask.sum())
        X = X[~inf_mask]
        y = y[~inf_mask]

        # 타겟(y)에 NaN이 있는 행 제거
        y_nan_mask = y.isna()
        if y_nan_mask.sum() > 0:
            X = X[~y_nan_mask]
            y = y[~y_nan_mask]
            stats['y_nan_removed'] = int(y_nan_mask.sum())

        # NaN이 너무 많은 행 제거 (컬럼의 >50%)
        nan_ratio_per_row = X[numeric_cols].isna().mean(axis=1)
        high_nan_mask = nan_ratio_per_row > 0.5
        stats['high_nan_rows_removed'] = int(high_nan_mask.sum())
        X = X[~high_nan_mask]
        y = y.loc[X.index]

        stats['rows_after'] = len(X)
        stats['cols_after'] = len(X.columns)
        stats['rows_kept_pct'] = float(len(X) / stats['rows_before'] * 100)
        stats['total_rows_removed'] = stats['rows_before'] - stats['rows_after']

        self.logger.info(f"   Rows: {stats['rows_before']:,} → {stats['rows_after']:,} ({stats['rows_kept_pct']:.1f}% kept)")
        self.logger.info(f"   Infinite rows removed: {stats.get('inf_rows_removed', 0):,}")
        self.logger.info(f"   High-NaN rows removed: {stats.get('high_nan_rows_removed', 0):,}")

        return X, y, stats

    def _compare_approaches(self, stats_a: Dict, stats_b: Dict) -> Dict:
        """두 가지 클리닝 접근 방식을 비교합니다."""
        comparison = {
            'data_retention': {
                'imputation_pct': stats_a['rows_kept_pct'],
                'removal_pct': stats_b['rows_kept_pct'],
                'difference_pct': stats_a['rows_kept_pct'] - stats_b['rows_kept_pct']
            },
            'rows_difference': stats_a['rows_after'] - stats_b['rows_after'],
            'imputation_keeps_more': stats_a['rows_after'] > stats_b['rows_after']
        }

        return comparison

    def _generate_recommendation(self, stats_a: Dict, stats_b: Dict, comparison: Dict) -> Dict:
        """비교 결과를 기반으로 권고사항을 생성합니다."""
        data_diff = comparison['data_retention']['difference_pct']

        if data_diff > 20:
            return {
                'preferred': 'imputation',
                'reason': f'Imputation keeps {data_diff:.1f}% more data',
                'confidence': 'HIGH',
                'action': 'Consider using imputation for sparse data'
            }
        elif data_diff > 10:
            return {
                'preferred': 'imputation',
                'reason': f'Imputation keeps {data_diff:.1f}% more data',
                'confidence': 'MEDIUM',
                'action': 'Test both approaches with actual model training'
            }
        elif data_diff < -5:
            return {
                'preferred': 'removal',
                'reason': 'Removal-based cleaning maintains data quality',
                'confidence': 'HIGH',
                'action': 'Current removal approach is appropriate'
            }
        else:
            return {
                'preferred': 'either',
                'reason': f'Both approaches retain similar data ({data_diff:.1f}% difference)',
                'confidence': 'MEDIUM',
                'action': 'Current approach is fine, no change needed'
            }

    def _log_results(self):
        """검증 결과를 로깅합니다."""
        results = self.validation_results
        comparison = results['comparison']
        rec = results['recommendation']

        self.logger.info("\n" + "=" * 60)
        self.logger.info("📋 A/B TEST RESULTS")
        self.logger.info("=" * 60)

        self.logger.info(f"\n📊 Data Retention Comparison:")
        self.logger.info(f"   Case A (Imputation): {comparison['data_retention']['imputation_pct']:.1f}% kept")
        self.logger.info(f"   Case B (Removal):    {comparison['data_retention']['removal_pct']:.1f}% kept")
        self.logger.info(f"   Difference:          {comparison['data_retention']['difference_pct']:+.1f}%")

        self.logger.info(f"\n💡 Recommendation:")
        confidence_icon = {'HIGH': '🟢', 'MEDIUM': '🟡', 'LOW': '🔴'}.get(rec['confidence'], '•')
        self.logger.info(f"   Preferred: {rec['preferred'].upper()}")
        self.logger.info(f"   Confidence: {confidence_icon} {rec['confidence']}")
        self.logger.info(f"   Reason: {rec['reason']}")
        self.logger.info(f"   Action: {rec['action']}")

        self.logger.info("=" * 60)


def generate_data_quality_report(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    config: Optional[Dict] = None,
    logger: Optional[logging.Logger] = None,
    save_path: Optional[str] = None
) -> DataQualityReport:
    """
    데이터 품질 보고서를 생성하는 편의 함수입니다.

    사용 예시:
        report = generate_data_quality_report(X_train, y_train, config, logger)
    """
    report = DataQualityReport(X, y, config, logger)
    report.generate()

    if save_path:
        report.save(save_path)

    return report


def validate_data_cleaning(
    X: pd.DataFrame,
    y: pd.Series,
    config: Optional[Dict] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    데이터 클리닝 효과를 검증하는 편의 함수입니다.

    사용 예시:
        results = validate_data_cleaning(X_train, y_train, config, logger)
    """
    validator = DataCleaningValidator(X, y, config, logger)
    return validator.validate_cleaning_effect()
