"""
Toollab - 工具库
包含因子分析、特征工程、模型研究等工具
"""

from .factor_ic_analyzer import FactorICAnalyzer
from .feature_preprocessor import FeaturePreprocessor
from .feature_engineer import FeatureEngineer
from .model_study import (
    ModelStudyConfig,
    ConvergenceAnalyzer,
    FeatureImportanceAnalyzer,
    TemporalStabilityAnalyzer,
    ModelStudyPipeline,
    run_full_analysis
)
from .metrics import calculate_score_metric
from .utils import (
    describe_dataset,
    missing_duplicates_analysis,
    detect_outliers,
    timer,
    now_str,
    safe_spearman,
    time_series_cv_splits,
    sliding_window_cv_splits,
    true_online_cv_splits,
    plot_feature_importance_lgbm,
    plot_feature_importance_catboost,
    analyze_factor_importance_from_model,
    analyze_lgbm_tree_complexity,
    apply_preprocessing_rules
)

# Model tuner (requires optuna)
from .model_tuner import ModelTuner

# Neural network models
from .nn_models import create_tabular_mlp

__all__ = [
    'FactorICAnalyzer',
    'FeaturePreprocessor',
    'FeatureEngineer',
    'ModelTuner',
    'calculate_score_metric',
    'ModelStudyConfig',
    'ConvergenceAnalyzer',
    'FeatureImportanceAnalyzer',
    'TemporalStabilityAnalyzer',
    'ModelStudyPipeline',
    'run_full_analysis',
    'describe_dataset',
    'missing_duplicates_analysis',
    'detect_outliers',
    'timer',
    'now_str',
    'safe_spearman',
    'time_series_cv_splits',
    'sliding_window_cv_splits',
    'true_online_cv_splits',
    'plot_feature_importance_lgbm',
    'plot_feature_importance_catboost',
    'analyze_factor_importance_from_model',
    'analyze_lgbm_tree_complexity',
    'create_tabular_mlp',
    'apply_preprocessing_rules'
]
