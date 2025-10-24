"""
Rebalancing Period Optimizer

과거 데이터를 기반으로 최적의 리밸런싱 기간을 찾는 시스템
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from pathlib import Path


class RebalancingOptimizer:
    """
    리밸런싱 기간 최적화

    여러 리밸런싱 기간(1개월, 2개월, 3개월, 6개월 등)을 테스트하여
    수익률이 가장 좋은 기간을 찾습니다.
    """

    def __init__(self,
                 periods_to_test: List[int] = None,
                 optimization_metric: str = 'total_return',
                 min_trades: int = 10):
        """
        Args:
            periods_to_test: 테스트할 리밸런싱 기간 리스트 (개월 단위)
            optimization_metric: 최적화 기준 ('total_return', 'sharpe_ratio', 'win_rate')
            min_trades: 최소 거래 횟수 (이보다 적으면 제외)
        """
        if periods_to_test is None:
            periods_to_test = [1, 2, 3, 4, 6, 12]

        self.periods_to_test = periods_to_test
        self.optimization_metric = optimization_metric
        self.min_trades = min_trades
        self.results = []
        self.optimal_period = None

    def optimize(self,
                backtest_func: Any,
                data: pd.DataFrame,
                date_col: str,
                start_date: datetime,
                end_date: datetime,
                top_k: int = 5,
                **backtest_kwargs) -> Dict[str, Any]:
        """
        리밸런싱 기간 최적화 수행

        Args:
            backtest_func: 백테스팅 함수 (rebalance_months 파라미터 받아야 함)
            data: 전체 데이터
            date_col: 날짜 컬럼명
            start_date: 시작 날짜
            end_date: 종료 날짜
            top_k: 매번 선택할 주식 개수
            **backtest_kwargs: 백테스팅 함수에 전달할 추가 인자

        Returns:
            최적화 결과 딕셔너리
        """
        logging.info(f"\n{'='*60}")
        logging.info("리밸런싱 기간 최적화 시작")
        logging.info(f"{'='*60}")
        logging.info(f"테스트할 기간: {self.periods_to_test}")
        logging.info(f"최적화 기준: {self.optimization_metric}")
        logging.info(f"{'='*60}\n")

        self.results = []

        for period_months in self.periods_to_test:
            logging.info(f"\n{'='*60}")
            logging.info(f"리밸런싱 기간: {period_months}개월 테스트")
            logging.info(f"{'='*60}")

            try:
                # 백테스팅 수행
                result = backtest_func(
                    data=data,
                    date_col=date_col,
                    start_date=start_date,
                    end_date=end_date,
                    rebalance_months=period_months,
                    top_k=top_k,
                    **backtest_kwargs
                )

                # 결과 계산
                metrics = self._calculate_metrics(result, period_months)

                # 최소 거래 횟수 체크
                if metrics['num_trades'] < self.min_trades:
                    logging.warning(
                        f"⚠️ {period_months}개월 기간은 거래 횟수({metrics['num_trades']})가 "
                        f"최소 거래 횟수({self.min_trades})보다 적어 제외됩니다."
                    )
                    continue

                self.results.append(metrics)

                logging.info(f"\n📊 {period_months}개월 기간 결과:")
                logging.info(f"  총 수익률: {metrics['total_return']:.2f}%")
                logging.info(f"  연평균 수익률: {metrics['annualized_return']:.2f}%")
                logging.info(f"  샤프 비율: {metrics['sharpe_ratio']:.3f}")
                logging.info(f"  승률: {metrics['win_rate']:.2f}%")
                logging.info(f"  거래 횟수: {metrics['num_trades']}")
                logging.info(f"  최대 낙폭(MDD): {metrics['max_drawdown']:.2f}%")

            except Exception as e:
                logging.error(f"❌ {period_months}개월 기간 테스트 실패: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        if not self.results:
            logging.error("모든 테스트 실패!")
            return {}

        # 최적 기간 선택
        self.optimal_period = self._select_optimal_period()

        logging.info(f"\n{'='*60}")
        logging.info("최적화 완료")
        logging.info(f"{'='*60}")
        logging.info(f"🎯 최적 리밸런싱 기간: {self.optimal_period['period_months']}개월")
        logging.info(f"   {self.optimization_metric}: {self.optimal_period[self.optimization_metric]:.2f}")
        logging.info(f"{'='*60}\n")

        return {
            'optimal_period': self.optimal_period,
            'all_results': self.results
        }

    def _calculate_metrics(self, backtest_result: pd.DataFrame, period_months: int) -> Dict:
        """
        백테스팅 결과로부터 메트릭 계산

        Args:
            backtest_result: 백테스팅 결과 데이터프레임
            period_months: 리밸런싱 기간

        Returns:
            메트릭 딕셔너리
        """
        if backtest_result.empty or 'return' not in backtest_result.columns:
            return {
                'period_months': period_months,
                'total_return': 0,
                'annualized_return': 0,
                'sharpe_ratio': 0,
                'win_rate': 0,
                'num_trades': 0,
                'max_drawdown': 0,
                'avg_return': 0,
                'std_return': 0
            }

        returns = backtest_result['return'].values
        num_trades = len(returns)

        # 총 수익률
        cumulative_returns = (1 + returns / 100).cumprod()
        total_return = (cumulative_returns.iloc[-1] - 1) * 100 if len(cumulative_returns) > 0 else 0

        # 연평균 수익률
        years = num_trades * period_months / 12
        annualized_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100 if years > 0 else 0

        # 샤프 비율 (무위험 수익률 0% 가정)
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe_ratio = (avg_return / std_return) if std_return > 0 else 0

        # 승률
        win_rate = (returns > 0).sum() / len(returns) * 100 if len(returns) > 0 else 0

        # MDD (Maximum Drawdown)
        max_drawdown = self._calculate_mdd(cumulative_returns)

        return {
            'period_months': period_months,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'num_trades': num_trades,
            'max_drawdown': max_drawdown,
            'avg_return': avg_return,
            'std_return': std_return
        }

    def _calculate_mdd(self, cumulative_returns: pd.Series) -> float:
        """최대 낙폭(MDD) 계산"""
        if len(cumulative_returns) == 0:
            return 0

        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max * 100
        return abs(drawdown.min())

    def _select_optimal_period(self) -> Dict:
        """최적 기간 선택"""
        if not self.results:
            return {}

        # 메트릭 기준으로 정렬
        if self.optimization_metric == 'sharpe_ratio':
            # 샤프 비율은 높을수록 좋음
            sorted_results = sorted(self.results, key=lambda x: x['sharpe_ratio'], reverse=True)
        elif self.optimization_metric == 'win_rate':
            # 승률은 높을수록 좋음
            sorted_results = sorted(self.results, key=lambda x: x['win_rate'], reverse=True)
        else:  # total_return or annualized_return
            # 수익률은 높을수록 좋음
            metric_key = self.optimization_metric
            sorted_results = sorted(self.results, key=lambda x: x[metric_key], reverse=True)

        return sorted_results[0]

    def plot_results(self, save_path: Optional[str] = None):
        """결과 시각화"""
        if not self.results:
            logging.warning("결과가 없어 시각화할 수 없습니다.")
            return

        df = pd.DataFrame(self.results)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('리밸런싱 기간별 성능 비교', fontsize=16, fontweight='bold')

        # 1. 총 수익률
        axes[0, 0].bar(df['period_months'], df['total_return'])
        axes[0, 0].set_title('총 수익률')
        axes[0, 0].set_xlabel('리밸런싱 기간 (개월)')
        axes[0, 0].set_ylabel('수익률 (%)')
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.3)

        # 2. 연평균 수익률
        axes[0, 1].bar(df['period_months'], df['annualized_return'])
        axes[0, 1].set_title('연평균 수익률')
        axes[0, 1].set_xlabel('리밸런싱 기간 (개월)')
        axes[0, 1].set_ylabel('수익률 (%)')
        axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.3)

        # 3. 샤프 비율
        axes[0, 2].bar(df['period_months'], df['sharpe_ratio'])
        axes[0, 2].set_title('샤프 비율')
        axes[0, 2].set_xlabel('리밸런싱 기간 (개월)')
        axes[0, 2].set_ylabel('샤프 비율')
        axes[0, 2].axhline(y=0, color='r', linestyle='--', alpha=0.3)

        # 4. 승률
        axes[1, 0].bar(df['period_months'], df['win_rate'])
        axes[1, 0].set_title('승률')
        axes[1, 0].set_xlabel('리밸런싱 기간 (개월)')
        axes[1, 0].set_ylabel('승률 (%)')
        axes[1, 0].axhline(y=50, color='r', linestyle='--', alpha=0.3)

        # 5. 최대 낙폭
        axes[1, 1].bar(df['period_months'], df['max_drawdown'])
        axes[1, 1].set_title('최대 낙폭 (MDD)')
        axes[1, 1].set_xlabel('리밸런싱 기간 (개월)')
        axes[1, 1].set_ylabel('MDD (%)')

        # 6. 거래 횟수
        axes[1, 2].bar(df['period_months'], df['num_trades'])
        axes[1, 2].set_title('거래 횟수')
        axes[1, 2].set_xlabel('리밸런싱 기간 (개월)')
        axes[1, 2].set_ylabel('횟수')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"📊 차트 저장: {save_path}")

        plt.show()

    def get_results_dataframe(self) -> pd.DataFrame:
        """결과를 데이터프레임으로 반환"""
        if not self.results:
            return pd.DataFrame()
        return pd.DataFrame(self.results)

    def save_results(self, path: str):
        """결과 저장"""
        if not self.results:
            logging.warning("저장할 결과가 없습니다.")
            return

        df = self.get_results_dataframe()

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)
        logging.info(f"💾 결과 저장: {output_path}")

        # 최적 기간 정보도 함께 저장
        if self.optimal_period:
            summary_path = output_path.parent / f"{output_path.stem}_summary.txt"
            with open(summary_path, 'w') as f:
                f.write(f"최적 리밸런싱 기간: {self.optimal_period['period_months']}개월\n")
                f.write(f"최적화 기준: {self.optimization_metric}\n\n")
                f.write("성능 지표:\n")
                for key, value in self.optimal_period.items():
                    f.write(f"  {key}: {value}\n")

            logging.info(f"💾 요약 저장: {summary_path}")
