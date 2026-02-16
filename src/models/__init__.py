"""퀀트 트레이딩을 위한 머신러닝 모델 패키지.

이 패키지는 주식 시장 예측 및 트레이딩 전략 개발을 위한 머신러닝 모델의
학습, 평가, 배포를 위한 통합 인터페이스를 제공합니다.

패키지 구성:
1. 기본 모델 아키텍처 (BaseModel)
2. Gradient boosting 구현체:
   - XGBoost: 빠르고 효율적인 gradient boosting
   - LightGBM: Microsoft의 저메모리 gradient boosting
   - CatBoost: Yandex의 범주형 feature 지원 gradient boosting
3. 앙상블 기법:
   - StackingEnsemble: 메타러닝 기반 앙상블
   - VotingEnsemble: 투표/평균 기반 앙상블
4. 금융 데이터에 최적화된 사전 설정

주요 기능:
- 모든 모델 유형에 걸친 일관된 API
- 모든 모델에 대한 GPU 가속 지원
- 내장 교차 검증 및 평가
- Feature 중요도 분석
- 모델 영속성 (저장/로드)
- 과적합 방지를 위한 조기 종료
- 시계열 인식 교차 검증

아키텍처:
    BaseModel (추상 클래스)
    ├── XGBoostModel
    ├── LightGBMModel
    └── CatBoostModel

    앙상블
    ├── StackingEnsemble
    └── VotingEnsemble

사용 예시:
    기본 모델 학습:
        from models import XGBoostModel

        # 모델 생성 및 학습
        model = XGBoostModel(task='classification', config_name='default')
        model.build_model()
        model.fit(X_train, y_train, X_val, y_val)

        # 예측 수행
        predictions = model.predict(X_test)
        metrics = model.evaluate(X_test, y_test)

        # 모델 저장
        model.save('trained_model.pkl')

    앙상블 학습:
        from models import StackingEnsemble, XGBoostModel, LightGBMModel, CatBoostModel

        # 기본 모델 학습
        xgb = XGBoostModel(task='classification').build_model()
        lgb = LightGBMModel(task='classification').build_model()
        cat = CatBoostModel(task='classification').build_model()

        xgb.fit(X_train, y_train, X_val, y_val)
        lgb.fit(X_train, y_train, X_val, y_val)
        cat.fit(X_train, y_train, X_val, y_val)

        # 앙상블 생성
        base_models = [
            ('xgboost', xgb.model),
            ('lightgbm', lgb.model),
            ('catboost', cat.model)
        ]

        ensemble = StackingEnsemble(
            base_models=base_models,
            task='classification',
            meta_learner='ridge'
        )
        ensemble.build_ensemble()
        ensemble.fit(X_train, y_train)

        # 예측 수행
        predictions = ensemble.predict(X_test)

    교차 검증:
        from models import CatBoostModel

        model = CatBoostModel(task='classification').build_model()

        # 교차 검증 및 학습
        avg_scores, fold_scores = model.fit_with_cv(
            X, y,
            dates=date_series,  # 시계열 교차 검증용
            cv_splits=5
        )

        print(f"CV Accuracy: {avg_scores['accuracy']:.4f}")

사용 가능한 모델:
    - BaseModel: 모든 모델의 추상 기본 클래스
    - XGBoostModel: XGBoost 분류기/회귀기
    - LightGBMModel: LightGBM 분류기/회귀기
    - CatBoostModel: CatBoost 분류기/회귀기
    - StackingEnsemble: 스태킹 앙상블 결합기
    - VotingEnsemble: 투표 앙상블 결합기 (기본 export 대상 아님)

설정:
    모든 모델은 config.py의 사전 설정을 사용하며, 금융 시계열 데이터에
    최적화되어 있습니다. 설정은 다음 방법으로 커스터마이즈할 수 있습니다:
    1. 다른 config_name 프리셋 사용
    2. build_model()에 커스텀 파라미터 전달
    3. config.py 직접 수정

모델 선택 가이드:
    - XGBoost: 빠른 학습, 우수한 범용 성능, 성숙한 라이브러리
    - LightGBM: 가장 빠른 학습, 최저 메모리, 대규모 데이터셋에 적합
    - CatBoost: 과적합 방지에 최적, 범주형 feature 처리,
                뛰어난 기본 성능 (트레이딩에 권장)
    - 앙상블: 최고의 종합 성능, 여러 모델의 장점 결합

성능 팁:
    - GPU 가속 사용 (설정에서 기본 활성화)
    - 과적합 방지를 위한 조기 종료 활성화
    - 일반화 성능 평가를 위한 교차 검증 사용
    - 견고한 기준선으로 CatBoost부터 시작
    - 프로덕션 시스템에서는 앙상블 사용

Note:
    - 모든 모델의 GPU 가속은 CUDA를 지원하는 GPU가 필요합니다
    - 금융 데이터에는 시계열 교차 검증을 권장합니다
    - Feature 엔지니어링이 모델 성능에 핵심적입니다
    - 항상 표본 외 데이터로 검증해야 합니다

작성자: Quantitative Trading Team
라이선스: Proprietary
"""

from .base_model import BaseModel
from .xgboost_model import XGBoostModel
from .lightgbm_model import LightGBMModel
from .catboost_model import CatBoostModel
from .ensemble import StackingEnsemble
from .model_factory import ModelFactory, create_models_for_regressor, create_models_for_backtest

__all__ = [
    'BaseModel',
    'XGBoostModel',
    'LightGBMModel',
    'CatBoostModel',
    'StackingEnsemble',
    'ModelFactory',
    'create_models_for_regressor',
    'create_models_for_backtest'
]

__version__ = '1.0.0'

# 패키지 메타데이터
__author__ = 'Quantitative Trading Team'
__description__ = '퀀트 트레이딩을 위한 머신러닝 모델'
