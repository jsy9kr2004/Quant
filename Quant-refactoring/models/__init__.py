"""
Machine learning models for quantitative trading
"""

from .base_model import BaseModel
from .xgboost_model import XGBoostModel
from .lightgbm_model import LightGBMModel
from .catboost_model import CatBoostModel
from .ensemble import StackingEnsemble

__all__ = [
    'BaseModel',
    'XGBoostModel',
    'LightGBMModel',
    'CatBoostModel',
    'StackingEnsemble'
]
