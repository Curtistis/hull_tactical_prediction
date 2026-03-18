"""
【实验3】动态因子池策略对比（三种策略）
评估指标：竞赛官方Score Metric（调整后夏普比率）
"""
import numpy as np
import pandas as pd
import sys
import os
import json
from datetime import datetime

# 配置
TARGET = 'market_forward_excess_returns'
TUNER_SEED = 42
RESULT_DIR = '/Users/curtis/Desktop/hull_tactical_prediction/v7_reuse/experiment3_results'

# 导入模块
from toollab.model_tuner import ModelTuner, calculate_score_metric
from toollab.factor_ic_analyzer import FactorICAnalyzer
from toollab.feature_preprocessor import FeaturePreprocessor
from toollab.utils import true_online_cv_splits
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit

print("="*80)
print("【实验3】固定因子池 vs 真·动态因子池对比（Score Metric版本）")
print("="*80)

# 创建结果目录
os.makedirs(RESULT_DIR, exist_ok=True)

# 加载数据
print("\n加载训练数据...")
train = pd.read_csv('train.csv')
print(f"✅ 训练数据加载完成: {train.shape}")

# 准备完整训练数据（包含所有原始因子）
exclude_meta = ['date_id', TARGET, 'forward_returns', 'risk_free_rate',
                'lagged_forward_returns', 'lagged_risk_free_rate',
                'lagged_market_forward_excess_returns', 'is_scored']

all_factor_cols = [c for c in train.columns if c not in exclude_meta]
train_factors = train[all_factor_cols + [TARGET]].copy()

print(f"\n原始因子数: {len(all_factor_cols)}")
print(f"训练样本数: {len(train_factors)}")

# 初始化
tuner = ModelTuner(
    seed=TUNER_SEED,
    use_sliding_window=True,
    train_window_size=90,  # 每个窗口90天
    cv_step=10             # 每10天滑动一次
)

analyzer = FactorICAnalyzer(window_size=20)
preprocessor = FeaturePreprocessor(analyzer, target_col=TARGET, verbose=False)

# 简单模型参数
simple_params = {
    'num_leaves': 31,
    'max_depth': 5,
    'learning_rate': 0.05,
    'n_estimators': 100,
    'objective': 'regression',
    'metric': 'rmse',
    'random_state': TUNER_SEED,
    'verbosity': -1
}

# 获取CV切分
y = train_factors[TARGET]
cv_splits = list(true_online_cv_splits(train_factors, train_size=90, step=10))
print(f"\nCV窗口数: {len(cv_splits)}")

# ============================================================
# 策略1: 固定因子池（全历史IC选Top30）
# ============================================================
print("\n" + "="*80)
print("【策略1】固定因子池 - 全历史IC选Top30")
print("="*80)

factor_df_all = analyzer.analyze_dataset(train_factors, verbose=False)
_, _, fixed_features = preprocessor.select_features_by_ic(
    train_factors, train_factors, factor_df_all, top_n=30
)
print(f"选中因子: {fixed_features[:5]}... (共{len(fixed_features)}个)")

window_scores_fixed = []
for window_id, (train_idx, val_idx) in enumerate(cv_splits):
    X_train = train_factors.iloc[train_idx][fixed_features]
    X_val = train_factors.iloc[val_idx][fixed_features]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    sample_weights = tuner.calculate_time_weights(len(y_train))

    model = LGBMRegressor(**simple_params)
    model.fit(X_train, y_train, sample_weight=sample_weights)

    preds = model.predict(X_val)
    score = calculate_score_metric(preds, y_val.values)
    window_scores_fixed.append(float(score))

    if window_id % 100 == 0:
        print(f"  窗口{window_id}: Score={score:.6f}")

strategy1_result = {
    "strategy": "fixed_all_history_ic",
    "n_windows": len(window_scores_fixed),
    "mean_score": float(np.mean(window_scores_fixed)),
    "std_score": float(np.std(window_scores_fixed)),
    "min_score": float(np.min(window_scores_fixed)),
    "max_score": float(np.max(window_scores_fixed)),
    "stability_score": float(np.mean(window_scores_fixed) - 2 * np.std(window_scores_fixed)),
}

print(f"\n策略1结果:")
print(f"  mean_score: {strategy1_result['mean_score']:.6f}")
print(f"  std_score: {strategy1_result['std_score']:.6f}")
print(f"  stability_score: {strategy1_result['stability_score']:.6f}")

# ============================================================
# 策略2: 动态因子池（每窗口用最近60天IC选Top30）
# ============================================================
print("\n" + "="*80)
print("【策略2】真·动态因子池 - 每窗口用最近60天IC选Top30")
print("="*80)

window_scores_dynamic_ic = []
factor_changes_ic = []

for window_id, (train_idx, val_idx) in enumerate(cv_splits):
    # 每个窗口都重新选因子
    train_window_data = train_factors.iloc[train_idx]

    # 用该窗口的最近60天计算IC
    recent_data = train_window_data.tail(60)
    factor_df_recent = analyzer.analyze_dataset(recent_data, verbose=False)
    _, _, current_features = preprocessor.select_features_by_ic(
        train_factors, train_factors, factor_df_recent, top_n=30
    )

    # 记录因子变化
    if window_id > 0:
        previous_features = factor_changes_ic[-1]['features']
        n_changed = len(set(current_features) - set(previous_features))
        factor_changes_ic.append({'window': window_id, 'features': current_features, 'n_changed': n_changed})
    else:
        factor_changes_ic.append({'window': window_id, 'features': current_features, 'n_changed': 0})

    # 使用当前因子训练
    X_train = train_factors.iloc[train_idx][current_features]
    X_val = train_factors.iloc[val_idx][current_features]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    sample_weights = tuner.calculate_time_weights(len(y_train))

    model = LGBMRegressor(**simple_params)
    model.fit(X_train, y_train, sample_weight=sample_weights)

    preds = model.predict(X_val)
    score = calculate_score_metric(preds, y_val.values)
    window_scores_dynamic_ic.append(float(score))

    if window_id % 100 == 0:
        print(f"  窗口{window_id}: Score={score:.6f}, 选中因子={current_features[:3]}...")

avg_change_ic = np.mean([x['n_changed'] for x in factor_changes_ic[1:]])
print(f"\n  因子变化统计: 平均每个窗口更换 {avg_change_ic:.1f} 个因子（共30个）")

strategy2_result = {
    "strategy": "dynamic_every_window_ic_60d",
    "n_windows": len(window_scores_dynamic_ic),
    "mean_score": float(np.mean(window_scores_dynamic_ic)),
    "std_score": float(np.std(window_scores_dynamic_ic)),
    "min_score": float(np.min(window_scores_dynamic_ic)),
    "max_score": float(np.max(window_scores_dynamic_ic)),
    "stability_score": float(np.mean(window_scores_dynamic_ic) - 2 * np.std(window_scores_dynamic_ic)),
    "avg_factor_change": float(avg_change_ic),
}

print(f"\n策略2结果:")
print(f"  mean_score: {strategy2_result['mean_score']:.6f}")
print(f"  std_score: {strategy2_result['std_score']:.6f}")
print(f"  stability_score: {strategy2_result['stability_score']:.6f}")

# ============================================================
# 策略3: 动态因子池（每窗口用最近60天Fitness选Top30）
# ============================================================
print("\n" + "="*80)
print("【策略3】真·动态因子池 - 每窗口用最近60天Fitness选Top30")
print("="*80)

window_scores_dynamic_fitness = []
factor_changes_fitness = []

for window_id, (train_idx, val_idx) in enumerate(cv_splits):
    # 每个窗口都重新选因子
    train_window_data = train_factors.iloc[train_idx]

    # 用该窗口的最近60天计算Fitness
    recent_data = train_window_data.tail(60)
    factor_df_recent = analyzer.analyze_dataset(recent_data, verbose=False)

    # 按Fitness（绝对值）排序选Top30
    factor_df_sorted = factor_df_recent.copy()
    factor_df_sorted['Fitness_abs'] = factor_df_sorted['Fitness'].abs()
    factor_df_sorted = factor_df_sorted.sort_values('Fitness_abs', ascending=False)
    current_features = factor_df_sorted.head(30)['特征'].tolist()

    # 记录因子变化
    if window_id > 0:
        previous_features = factor_changes_fitness[-1]['features']
        n_changed = len(set(current_features) - set(previous_features))
        factor_changes_fitness.append({'window': window_id, 'features': current_features, 'n_changed': n_changed})
    else:
        factor_changes_fitness.append({'window': window_id, 'features': current_features, 'n_changed': 0})

    # 使用当前因子训练
    X_train = train_factors.iloc[train_idx][current_features]
    X_val = train_factors.iloc[val_idx][current_features]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    sample_weights = tuner.calculate_time_weights(len(y_train))

    model = LGBMRegressor(**simple_params)
    model.fit(X_train, y_train, sample_weight=sample_weights)

    preds = model.predict(X_val)
    score = calculate_score_metric(preds, y_val.values)
    window_scores_dynamic_fitness.append(float(score))

    if window_id % 100 == 0:
        print(f"  窗口{window_id}: Score={score:.6f}, 选中因子={current_features[:3]}...")

avg_change_fitness = np.mean([x['n_changed'] for x in factor_changes_fitness[1:]])
print(f"\n  因子变化统计: 平均每个窗口更换 {avg_change_fitness:.1f} 个因子（共30个）")

strategy3_result = {
    "strategy": "dynamic_every_window_fitness_60d",
    "n_windows": len(window_scores_dynamic_fitness),
    "mean_score": float(np.mean(window_scores_dynamic_fitness)),
    "std_score": float(np.std(window_scores_dynamic_fitness)),
    "min_score": float(np.min(window_scores_dynamic_fitness)),
    "max_score": float(np.max(window_scores_dynamic_fitness)),
    "stability_score": float(np.mean(window_scores_dynamic_fitness) - 2 * np.std(window_scores_dynamic_fitness)),
    "avg_factor_change": float(avg_change_fitness),
}

print(f"\n策略3结果:")
print(f"  mean_score: {strategy3_result['mean_score']:.6f}")
print(f"  std_score: {strategy3_result['std_score']:.6f}")
print(f"  stability_score: {strategy3_result['stability_score']:.6f}")

# ============================================================
# 汇总结果
# ============================================================
print("\n" + "="*80)
print("【汇总结果】三种策略对比")
print("="*80)

results_df = pd.DataFrame([strategy1_result, strategy2_result, strategy3_result])
results_df = results_df.sort_values("mean_score", ascending=False).reset_index(drop=True)

print("\n" + results_df.to_string(index=False))

# 找出最佳策略
best_strategy = results_df.iloc[0]['strategy']
best_score = results_df.iloc[0]['mean_score']
print(f"\n✅ 最佳策略: {best_strategy} (mean_score={best_score:.6f})")

# ============================================================
# 保存结果
# ============================================================
print("\n" + "="*80)
print("【保存结果】")
print("="*80)

# 保存CSV
csv_path = os.path.join(RESULT_DIR, 'strategy_comparison.csv')
results_df.to_csv(csv_path, index=False)
print(f"✅ 策略对比表已保存: {csv_path}")

# 保存详细分数
detailed_scores = {
    'strategy1_fixed_ic': {
        'scores': window_scores_fixed,
        'config': strategy1_result
    },
    'strategy2_dynamic_ic': {
        'scores': window_scores_dynamic_ic,
        'factor_changes': [{'window': x['window'], 'n_changed': x['n_changed']} for x in factor_changes_ic],
        'config': strategy2_result
    },
    'strategy3_dynamic_fitness': {
        'scores': window_scores_dynamic_fitness,
        'factor_changes': [{'window': x['window'], 'n_changed': x['n_changed']} for x in factor_changes_fitness],
        'config': strategy3_result
    },
    'metadata': {
        'timestamp': datetime.now().isoformat(),
        'n_windows': len(cv_splits),
        'train_window_size': 90,
        'cv_step': 10,
        'top_n': 30,
        'ic_window_size': 60,
        'model_params': simple_params
    }
}

json_path = os.path.join(RESULT_DIR, 'detailed_scores.json')
with open(json_path, 'w') as f:
    json.dump(detailed_scores, f, indent=2)
print(f"✅ 详细分数已保存: {json_path}")

# 保存分数分布图数据
scores_df = pd.DataFrame({
    'window_id': range(len(cv_splits)),
    'strategy1_fixed_ic': window_scores_fixed,
    'strategy2_dynamic_ic': window_scores_dynamic_ic,
    'strategy3_dynamic_fitness': window_scores_dynamic_fitness,
})

scores_csv_path = os.path.join(RESULT_DIR, 'window_scores.csv')
scores_df.to_csv(scores_csv_path, index=False)
print(f"✅ 窗口分数已保存: {scores_csv_path}")

print("\n" + "="*80)
print("【实验完成】")
print("="*80)
print(f"\n结果保存目录: {RESULT_DIR}")
print(f"\n文件清单:")
print(f"  1. strategy_comparison.csv - 策略对比汇总表")
print(f"  2. detailed_scores.json - 详细分数和配置")
print(f"  3. window_scores.csv - 每个窗口的分数")

print("\n💡 关键发现:")
print(f"   - 最佳策略: {best_strategy}")
print(f"   - 平均Score: {best_score:.6f}")
print(f"   - 策略2平均因子变化: {avg_change_ic:.1f}/30")
print(f"   - 策略3平均因子变化: {avg_change_fitness:.1f}/30")
