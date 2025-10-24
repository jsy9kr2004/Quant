"""
Model Comparator

여러 모델 버전을 비교하여 성능 개선 여부를 평가하는 시스템
"""

import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
from scipy import stats


class ModelComparator:
    """
    모델 버전 비교 시스템

    - 동일한 테스트 셋에서 여러 모델 평가
    - 통계적 유의성 검정
    - 성능 개선 여부 판단
    - 결과 시각화 및 저장
    """

    def __init__(self, experiment_name: str = "model_comparison"):
        """
        Args:
            experiment_name: 실험 이름
        """
        self.experiment_name = experiment_name
        self.models = {}
        self.results = []
        self.comparison_data = None

    def add_model(self,
                 model_name: str,
                 model_instance: Any,
                 description: str = "",
                 hyperparameters: Optional[Dict] = None):
        """
        비교할 모델 추가

        Args:
            model_name: 모델 이름 (예: "XGBoost_v1", "XGBoost_v2")
            model_instance: 모델 인스턴스
            description: 모델 설명
            hyperparameters: 하이퍼파라미터
        """
        self.models[model_name] = {
            'instance': model_instance,
            'description': description,
            'hyperparameters': hyperparameters or {},
            'added_time': datetime.now()
        }

        logging.info(f"✅ 모델 추가: {model_name}")
        if description:
            logging.info(f"   설명: {description}")

    def compare_models(self,
                      X_train: pd.DataFrame,
                      y_train: pd.Series,
                      X_test: pd.DataFrame,
                      y_test: pd.Series,
                      cv_splits: int = 5) -> pd.DataFrame:
        """
        모든 모델 비교

        Args:
            X_train: 학습 데이터
            y_train: 학습 레이블
            X_test: 테스트 데이터
            y_test: 테스트 레이블
            cv_splits: Cross-validation fold 수

        Returns:
            비교 결과 데이터프레임
        """
        if not self.models:
            logging.error("비교할 모델이 없습니다!")
            return pd.DataFrame()

        logging.info(f"\n{'='*60}")
        logging.info(f"모델 비교 시작: {len(self.models)}개 모델")
        logging.info(f"{'='*60}\n")

        self.results = []

        for model_name, model_info in self.models.items():
            logging.info(f"\n{'='*60}")
            logging.info(f"평가 중: {model_name}")
            logging.info(f"{'='*60}")

            try:
                model = model_info['instance']

                # 1. Cross-Validation 성능
                logging.info("1️⃣ Cross-Validation...")
                cv_scores, cv_fold_scores = model.cross_validate(
                    X_train, y_train, cv_splits=cv_splits, verbose=False
                )

                # 2. 학습
                logging.info("2️⃣ 전체 데이터로 학습...")
                model.fit(X_train, y_train, verbose=0)

                # 3. 테스트 평가
                logging.info("3️⃣ 테스트 데이터 평가...")
                test_metrics = model.evaluate(X_test, y_test)

                # 4. 예측 값 저장 (통계 검정용)
                y_pred = model.predict(X_test)

                result = {
                    'model_name': model_name,
                    'description': model_info['description'],
                    'train_samples': len(X_train),
                    'test_samples': len(X_test),
                    'y_pred': y_pred,  # 통계 검정용
                    **test_metrics,
                    **{f'cv_{k}': v for k, v in cv_scores.items()}
                }

                self.results.append(result)

                logging.info(f"\n📊 {model_name} 결과:")
                logging.info(f"  Test Accuracy: {test_metrics.get('accuracy', 0):.4f}")
                logging.info(f"  CV Accuracy: {cv_scores.get('accuracy_mean', 0):.4f} ± {cv_scores.get('accuracy_std', 0):.4f}")

            except Exception as e:
                logging.error(f"❌ {model_name} 평가 실패: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        # 결과 데이터프레임 생성
        self.comparison_data = self._create_comparison_df()

        # 통계적 유의성 검정
        self._perform_statistical_tests(y_test)

        # 요약
        self._print_summary()

        return self.comparison_data

    def _create_comparison_df(self) -> pd.DataFrame:
        """비교 데이터프레임 생성"""
        if not self.results:
            return pd.DataFrame()

        # y_pred는 제외하고 데이터프레임 생성
        df_data = []
        for result in self.results:
            row = {k: v for k, v in result.items() if k != 'y_pred'}
            df_data.append(row)

        df = pd.DataFrame(df_data)

        # 정렬 (테스트 accuracy 기준)
        if 'accuracy' in df.columns:
            df = df.sort_values('accuracy', ascending=False)

        return df

    def _perform_statistical_tests(self, y_test: pd.Series):
        """통계적 유의성 검정"""
        if len(self.results) < 2:
            return

        logging.info(f"\n{'='*60}")
        logging.info("통계적 유의성 검정")
        logging.info(f"{'='*60}\n")

        # 모든 모델 쌍에 대해 McNemar's test 수행 (분류 문제)
        for i in range(len(self.results)):
            for j in range(i + 1, len(self.results)):
                model1 = self.results[i]
                model2 = self.results[j]

                try:
                    # McNemar's test
                    y_pred1 = model1['y_pred']
                    y_pred2 = model2['y_pred']

                    # Confusion matrix
                    both_correct = ((y_pred1 == y_test) & (y_pred2 == y_test)).sum()
                    both_wrong = ((y_pred1 != y_test) & (y_pred2 != y_test)).sum()
                    model1_correct = ((y_pred1 == y_test) & (y_pred2 != y_test)).sum()
                    model2_correct = ((y_pred1 != y_test) & (y_pred2 == y_test)).sum()

                    # McNemar's test (continuity correction 적용)
                    if model1_correct + model2_correct > 0:
                        statistic = (abs(model1_correct - model2_correct) - 1) ** 2 / (model1_correct + model2_correct)
                        p_value = 1 - stats.chi2.cdf(statistic, 1)

                        logging.info(f"\n{model1['model_name']} vs {model2['model_name']}:")
                        logging.info(f"  McNemar's test statistic: {statistic:.4f}")
                        logging.info(f"  p-value: {p_value:.4f}")

                        if p_value < 0.05:
                            better_model = model1['model_name'] if model1_correct > model2_correct else model2['model_name']
                            logging.info(f"  ✅ 유의미한 차이 있음 (p < 0.05)")
                            logging.info(f"  🏆 {better_model}이(가) 통계적으로 더 우수")
                        else:
                            logging.info(f"  ❌ 유의미한 차이 없음 (p >= 0.05)")

                except Exception as e:
                    logging.debug(f"통계 검정 실패: {str(e)}")
                    continue

    def _print_summary(self):
        """요약 출력"""
        if self.comparison_data is None or self.comparison_data.empty:
            return

        logging.info(f"\n{'='*60}")
        logging.info("모델 비교 요약")
        logging.info(f"{'='*60}\n")

        # 주요 메트릭만 출력
        display_cols = ['model_name', 'description']

        if 'accuracy' in self.comparison_data.columns:
            display_cols.extend(['accuracy', 'cv_accuracy_mean', 'cv_accuracy_std'])
        if 'f1' in self.comparison_data.columns:
            display_cols.extend(['f1'])
        if 'precision' in self.comparison_data.columns:
            display_cols.extend(['precision', 'recall'])

        display_cols = [col for col in display_cols if col in self.comparison_data.columns]

        logging.info(self.comparison_data[display_cols].to_string())

        # 최고 성능 모델
        if 'accuracy' in self.comparison_data.columns:
            best_model = self.comparison_data.iloc[0]
            logging.info(f"\n🏆 최고 성능 모델: {best_model['model_name']}")
            logging.info(f"   Test Accuracy: {best_model['accuracy']:.4f}")

    def plot_comparison(self, save_path: Optional[str] = None):
        """모델 비교 시각화"""
        if self.comparison_data is None or self.comparison_data.empty:
            logging.warning("비교 데이터가 없어 시각화할 수 없습니다.")
            return

        df = self.comparison_data

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'모델 성능 비교: {self.experiment_name}', fontsize=16, fontweight='bold')

        model_names = df['model_name'].values
        x_pos = np.arange(len(model_names))

        # 1. Test Accuracy
        if 'accuracy' in df.columns:
            axes[0, 0].bar(x_pos, df['accuracy'])
            axes[0, 0].set_title('Test Accuracy')
            axes[0, 0].set_xticks(x_pos)
            axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].grid(axis='y', alpha=0.3)

        # 2. CV Accuracy (with error bars)
        if 'cv_accuracy_mean' in df.columns and 'cv_accuracy_std' in df.columns:
            axes[0, 1].bar(x_pos, df['cv_accuracy_mean'], yerr=df['cv_accuracy_std'], capsize=5)
            axes[0, 1].set_title('CV Accuracy (mean ± std)')
            axes[0, 1].set_xticks(x_pos)
            axes[0, 1].set_xticklabels(model_names, rotation=45, ha='right')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].grid(axis='y', alpha=0.3)

        # 3. Precision & Recall
        if 'precision' in df.columns and 'recall' in df.columns:
            width = 0.35
            axes[1, 0].bar(x_pos - width/2, df['precision'], width, label='Precision')
            axes[1, 0].bar(x_pos + width/2, df['recall'], width, label='Recall')
            axes[1, 0].set_title('Precision & Recall')
            axes[1, 0].set_xticks(x_pos)
            axes[1, 0].set_xticklabels(model_names, rotation=45, ha='right')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].legend()
            axes[1, 0].grid(axis='y', alpha=0.3)

        # 4. F1 Score
        if 'f1' in df.columns:
            axes[1, 1].bar(x_pos, df['f1'])
            axes[1, 1].set_title('F1 Score')
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].set_xticklabels(model_names, rotation=45, ha='right')
            axes[1, 1].set_ylabel('F1')
            axes[1, 1].grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"📊 차트 저장: {save_path}")

        plt.show()

    def save_results(self, output_dir: str = './results/model_comparison'):
        """결과 저장"""
        if self.comparison_data is None or self.comparison_data.empty:
            logging.warning("저장할 결과가 없습니다.")
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 1. CSV 저장
        csv_file = output_path / f'{self.experiment_name}_{timestamp}.csv'
        # y_pred 제외하고 저장
        df_to_save = self.comparison_data.copy()
        if 'y_pred' in df_to_save.columns:
            df_to_save = df_to_save.drop('y_pred', axis=1)
        df_to_save.to_csv(csv_file, index=False)
        logging.info(f"💾 결과 저장: {csv_file}")

        # 2. 메타데이터 저장 (JSON)
        metadata = {
            'experiment_name': self.experiment_name,
            'timestamp': timestamp,
            'num_models': len(self.models),
            'models': {}
        }

        for model_name, model_info in self.models.items():
            metadata['models'][model_name] = {
                'description': model_info['description'],
                'hyperparameters': model_info['hyperparameters'],
                'added_time': model_info['added_time'].strftime('%Y-%m-%d %H:%M:%S')
            }

        json_file = output_path / f'{self.experiment_name}_{timestamp}_metadata.json'
        with open(json_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logging.info(f"💾 메타데이터 저장: {json_file}")

        # 3. 요약 텍스트 저장
        summary_file = output_path / f'{self.experiment_name}_{timestamp}_summary.txt'
        with open(summary_file, 'w') as f:
            f.write(f"실험명: {self.experiment_name}\n")
            f.write(f"날짜: {timestamp}\n")
            f.write(f"비교 모델 수: {len(self.models)}\n\n")

            f.write("="*60 + "\n")
            f.write("성능 비교\n")
            f.write("="*60 + "\n")
            f.write(df_to_save.to_string())
            f.write("\n\n")

            if 'accuracy' in df_to_save.columns:
                best_model = df_to_save.iloc[0]
                f.write("="*60 + "\n")
                f.write("최고 성능 모델\n")
                f.write("="*60 + "\n")
                f.write(f"모델명: {best_model['model_name']}\n")
                f.write(f"설명: {best_model['description']}\n")
                f.write(f"Test Accuracy: {best_model['accuracy']:.4f}\n")

        logging.info(f"💾 요약 저장: {summary_file}")

    def get_best_model(self) -> Optional[Dict]:
        """최고 성능 모델 정보 반환"""
        if self.comparison_data is None or self.comparison_data.empty:
            return None

        best_row = self.comparison_data.iloc[0]
        return {
            'model_name': best_row['model_name'],
            'description': best_row.get('description', ''),
            'metrics': best_row.to_dict()
        }

    def is_improved(self,
                   new_model_name: str,
                   baseline_model_name: str,
                   metric: str = 'accuracy',
                   threshold: float = 0.01) -> bool:
        """
        새 모델이 baseline 대비 개선되었는지 확인

        Args:
            new_model_name: 새 모델 이름
            baseline_model_name: 기준 모델 이름
            metric: 비교 메트릭
            threshold: 개선 임계값 (1% = 0.01)

        Returns:
            개선 여부
        """
        if self.comparison_data is None or self.comparison_data.empty:
            return False

        new_model = self.comparison_data[self.comparison_data['model_name'] == new_model_name]
        baseline = self.comparison_data[self.comparison_data['model_name'] == baseline_model_name]

        if new_model.empty or baseline.empty:
            logging.warning(f"모델을 찾을 수 없습니다: {new_model_name} or {baseline_model_name}")
            return False

        new_score = new_model[metric].values[0]
        baseline_score = baseline[metric].values[0]

        improvement = new_score - baseline_score
        improvement_pct = improvement / baseline_score * 100 if baseline_score > 0 else 0

        logging.info(f"\n{'='*60}")
        logging.info(f"모델 개선 여부 확인")
        logging.info(f"{'='*60}")
        logging.info(f"Baseline: {baseline_model_name} ({metric}={baseline_score:.4f})")
        logging.info(f"New: {new_model_name} ({metric}={new_score:.4f})")
        logging.info(f"개선: {improvement:+.4f} ({improvement_pct:+.2f}%)")
        logging.info(f"임계값: {threshold:.4f} ({threshold*100:.2f}%)")

        if improvement > threshold:
            logging.info(f"✅ 개선되었습니다!")
            return True
        else:
            logging.info(f"❌ 개선되지 않았습니다.")
            return False
