"""
Performance Monitoring System

모델 성능 모니터링 및 성능 저하 감지
"""

import numpy as np
import pandas as pd
import logging
from collections import deque
from scipy import stats
from typing import Dict, List, Optional, Any
import warnings


class PerformanceMonitor:
    """
    모델 성능 모니터링 및 성능 저하 감지

    모니터링 항목:
    1. Rolling window 성능 추적
    2. 성능 저하 알림 (degradation alert)
    3. 데이터 drift 감지
    4. 재학습 필요 여부 판단
    """

    def __init__(self,
                 window_size: int = 10,
                 alert_threshold: float = 0.1,
                 drift_threshold: float = 0.05):
        """
        Args:
            window_size: 성능 추적 윈도우 크기
            alert_threshold: 성능 저하 알림 임계값 (10% = 0.1)
            drift_threshold: 데이터 드리프트 p-value 임계값
        """
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self.drift_threshold = drift_threshold

        # 성능 이력
        self.performance_history = deque(maxlen=window_size)
        self.baseline_performance = None

        # 피처 분포 이력 (drift 감지용)
        self.feature_distributions = {}

        # 알림 이력
        self.alerts = []

    def update_performance(self, metrics: Dict[str, float], period_label: Optional[str] = None):
        """
        성능 업데이트 및 모니터링

        Args:
            metrics: 성능 지표 딕셔너리 (예: {'accuracy': 0.85, 'f1': 0.82})
            period_label: 기간 라벨 (예: '2023-01')
        """
        performance_record = {
            'period': period_label,
            **metrics
        }

        self.performance_history.append(performance_record)

        # Baseline 설정 (최초 3개 기간의 평균)
        if self.baseline_performance is None and len(self.performance_history) >= 3:
            self.baseline_performance = self._calculate_baseline()
            logging.info(f"📊 Baseline 성능 설정됨:")
            for metric, value in self.baseline_performance.items():
                logging.info(f"   {metric}: {value:.4f}")

        # 성능 저하 체크
        if self.baseline_performance is not None:
            alerts = self._check_performance_degradation(metrics, period_label)
            if alerts:
                self.alerts.extend(alerts)

    def _calculate_baseline(self) -> Dict[str, float]:
        """초기 성능 기준선 계산 (최초 3개 기간의 평균)"""
        history = list(self.performance_history)[:3]
        baseline = {}

        # 모든 metric의 평균 계산
        for metric in history[0].keys():
            if metric != 'period':
                values = [h[metric] for h in history if metric in h]
                if values:
                    baseline[metric] = np.mean(values)

        return baseline

    def _check_performance_degradation(self,
                                      current_metrics: Dict[str, float],
                                      period_label: Optional[str] = None) -> List[Dict]:
        """
        성능 저하 감지

        Args:
            current_metrics: 현재 성능 지표
            period_label: 기간 라벨

        Returns:
            알림 리스트
        """
        alerts = []

        for metric, baseline_value in self.baseline_performance.items():
            if metric == 'period':
                continue

            current_value = current_metrics.get(metric, 0)

            # 성능 하락률 계산
            if baseline_value != 0:
                degradation = (baseline_value - current_value) / baseline_value
            else:
                degradation = 0

            if degradation > self.alert_threshold:
                alert_msg = (
                    f"⚠️ 성능 저하 감지!\n"
                    f"   Period: {period_label}\n"
                    f"   Metric: {metric}\n"
                    f"   Baseline: {baseline_value:.4f}\n"
                    f"   Current:  {current_value:.4f}\n"
                    f"   저하율: {degradation*100:.2f}%"
                )
                logging.warning(alert_msg)

                alert = {
                    'period': period_label,
                    'metric': metric,
                    'baseline': baseline_value,
                    'current': current_value,
                    'degradation': degradation
                }
                alerts.append(alert)

        return alerts

    def check_feature_drift(self,
                           X_new: np.ndarray,
                           feature_names: List[str],
                           reference_X: Optional[np.ndarray] = None) -> List[Dict]:
        """
        피처 분포 변화 감지 (Data Drift)

        Kolmogorov-Smirnov test 사용하여 분포 변화 감지

        Args:
            X_new: 새로운 피처 데이터
            feature_names: 피처 이름 리스트
            reference_X: 기준 피처 데이터 (None이면 X_new를 기준으로 저장)

        Returns:
            드리프트가 감지된 피처 정보 리스트
        """
        if reference_X is None and not self.feature_distributions:
            # 첫 실행: 기준 분포 저장
            for i, feat in enumerate(feature_names):
                if i < X_new.shape[1]:
                    self.feature_distributions[feat] = X_new[:, i]
            logging.info(f"📊 기준 피처 분포 저장 완료 ({len(feature_names)}개 피처)")
            return []

        drift_features = []

        for i, feat in enumerate(feature_names):
            if i >= X_new.shape[1]:
                continue

            if feat not in self.feature_distributions:
                # 새로운 피처는 분포 저장
                self.feature_distributions[feat] = X_new[:, i]
                continue

            try:
                # KS test (두 분포 비교)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    statistic, p_value = stats.ks_2samp(
                        self.feature_distributions[feat],
                        X_new[:, i]
                    )

                if p_value < self.drift_threshold:
                    logging.warning(
                        f"⚠️ 피처 드리프트 감지: {feat}\n"
                        f"   KS statistic: {statistic:.4f}, p-value: {p_value:.4f}"
                    )
                    drift_features.append({
                        'feature': feat,
                        'ks_statistic': statistic,
                        'p_value': p_value
                    })

            except Exception as e:
                logging.debug(f"KS test failed for {feat}: {str(e)}")
                continue

        if drift_features:
            logging.warning(f"총 {len(drift_features)}개 피처에서 드리프트 감지됨")

        return drift_features

    def should_retrain(self, consecutive_periods: int = 3) -> bool:
        """
        재학습 필요 여부 판단

        최근 N개 기간 모두 baseline보다 성능이 낮으면 재학습 권장

        Args:
            consecutive_periods: 연속으로 체크할 기간 수

        Returns:
            재학습 필요 여부
        """
        if len(self.performance_history) < consecutive_periods:
            return False

        if self.baseline_performance is None:
            return False

        recent = list(self.performance_history)[-consecutive_periods:]

        # 주요 지표 선택 (accuracy or f1 or rmse)
        metric = None
        if 'accuracy' in self.baseline_performance:
            metric = 'accuracy'
        elif 'f1' in self.baseline_performance:
            metric = 'f1'
        elif 'rmse' in self.baseline_performance:
            metric = 'rmse'
        else:
            # 첫 번째 metric 사용
            metric = list(self.baseline_performance.keys())[0]

        if metric not in self.baseline_performance:
            return False

        baseline_value = self.baseline_performance[metric]

        # 최근 N개 기간의 metric 값 추출
        recent_values = []
        for r in recent:
            if metric in r:
                recent_values.append(r[metric])

        if len(recent_values) < consecutive_periods:
            return False

        # rmse는 낮을수록 좋음
        if metric == 'rmse':
            all_degraded = all(v > baseline_value * (1 + self.alert_threshold)
                              for v in recent_values)
        else:
            # accuracy, f1 등은 높을수록 좋음
            all_degraded = all(v < baseline_value * (1 - self.alert_threshold)
                              for v in recent_values)

        if all_degraded:
            logging.warning(
                f"🔄 재학습 권장!\n"
                f"   최근 {consecutive_periods}개 기간 모두 baseline 대비 "
                f"{self.alert_threshold*100}% 이상 변화\n"
                f"   Metric: {metric}\n"
                f"   Baseline: {baseline_value:.4f}\n"
                f"   Recent: {[f'{v:.4f}' for v in recent_values]}"
            )
            return True

        return False

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        성능 요약 리포트

        Returns:
            성능 요약 딕셔너리
        """
        if not self.performance_history:
            return {"status": "성능 이력 없음"}

        df = pd.DataFrame(list(self.performance_history))

        # 'period' 컬럼 제외하고 통계 계산
        numeric_cols = [col for col in df.columns if col != 'period']

        summary = {
            'baseline': self.baseline_performance,
            'current': dict(df.iloc[-1]),
            'window_size': len(self.performance_history),
            'statistics': {}
        }

        for col in numeric_cols:
            summary['statistics'][col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'current': df[col].iloc[-1]
            }

        # 알림 요약
        summary['total_alerts'] = len(self.alerts)
        summary['recent_alerts'] = len([a for a in self.alerts
                                       if a.get('period') == df.iloc[-1].get('period')])

        return summary

    def print_summary(self):
        """성능 요약 출력"""
        summary = self.get_performance_summary()

        if summary.get('status'):
            logging.info(summary['status'])
            return

        logging.info(f"\n{'='*60}")
        logging.info("성능 모니터링 요약")
        logging.info(f"{'='*60}")

        if summary.get('baseline'):
            logging.info("\n📊 Baseline 성능:")
            for metric, value in summary['baseline'].items():
                logging.info(f"   {metric}: {value:.4f}")

        logging.info(f"\n📈 통계 (최근 {summary['window_size']}개 기간):")
        for metric, stats in summary['statistics'].items():
            logging.info(f"   {metric}:")
            logging.info(f"      Mean: {stats['mean']:.4f} (±{stats['std']:.4f})")
            logging.info(f"      Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            logging.info(f"      Current: {stats['current']:.4f}")

        logging.info(f"\n⚠️ 알림:")
        logging.info(f"   총 알림: {summary['total_alerts']}")
        logging.info(f"   최근 알림: {summary['recent_alerts']}")

        logging.info(f"\n{'='*60}\n")

    def reset(self):
        """모니터링 상태 초기화"""
        self.performance_history.clear()
        self.baseline_performance = None
        self.feature_distributions.clear()
        self.alerts.clear()
        logging.info("✅ 성능 모니터 초기화 완료")

    def get_alerts_dataframe(self) -> pd.DataFrame:
        """알림 이력을 데이터프레임으로 반환"""
        if not self.alerts:
            return pd.DataFrame()
        return pd.DataFrame(self.alerts)
