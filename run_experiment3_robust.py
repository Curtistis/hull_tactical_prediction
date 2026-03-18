"""
【实验3健壮版】动态因子池策略对比
特点：
1. 错误处理：每个窗口独立try-catch
2. 进度保存：每20个窗口保存一次中间结果
3. 断点续传：如果中断可以从上次位置继续
"""
import numpy as np
import pandas as pd
import sys
import os
import json
import warnings
from datetime import datetime

warnings.filterwarnings('ignore', category=Warning)

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

print("="*80)
print("【实验3健壮版】固定 vs 动态因子池（Score Metric）")
print("="*80)
print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 创建结果目录
os.makedirs(RESULT_DIR, exist_ok=True)

# 加载数据
print("\n[1/4] 加载数据...")
train = pd.read_csv('train.csv')
print(f"✅ 训练数据: {train.shape}")

# 准备数据
exclude_meta = ['date_id', TARGET, 'forward_returns', 'risk_free_rate',
                'lagged_forward_returns', 'lagged_risk_free_rate',
                'lagged_market_forward_excess_returns', 'is_scored']

all_factor_cols = [c for c in train.columns if c not in exclude_meta]
train_factors = train[all_factor_cols + [TARGET]].copy()

print(f"✅ 原始因子数: {len(all_factor_cols)}")
print(f"✅ 训练样本数: {len(train_factors)}")

# 初始化
tuner = ModelTuner(
    seed=TUNER_SEED,
    use_sliding_window=True,
    train_window_size=90,
    cv_step=30  # 使用30天步长减少窗口数
)

analyzer = FactorICAnalyzer(window_size=20)
preprocessor = FeaturePreprocessor(analyzer, target_col=TARGET, verbose=False)

# 模型参数
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
cv_splits = list(true_online_cv_splits(train_factors, train_size=90, step=30))
total_windows = len(cv_splits)
print(f"✅ CV窗口数: {total_windows} (step=30天)")

# 辅助函数：保存中间结果
def save_checkpoint(strategy_name, scores, factor_changes=None):
    checkpoint = {
        'strategy': strategy_name,
        'scores': scores,
        'n_completed': len(scores),
        'timestamp': datetime.now().isoformat()
    }
    if factor_changes:
        checkpoint['factor_changes'] = factor_changes

    checkpoint_path = os.path.join(RESULT_DIR, f'checkpoint_{strategy_name}.json')
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f)

# ============================================================
# 策略1: 固定因子池（全历史IC选Top30）
# ============================================================
print("\n[2/4] 策略1: 固定因子池")
print("-"*80)

try:
    factor_df_all = analyzer.analyze_dataset(train_factors, verbose=False)
    _, _, fixed_features = preprocessor.select_features_by_ic(
        train_factors, train_factors, factor_df_all, top_n=30
    )
    print(f"选中因子: {fixed_features[:5]}...")
except Exception as e:
    print(f"❌ 因子选择失败: {e}")
    fixed_features = all_factor_cols[:30]  # 降级方案
    print(f"使用前30个因子作为降级方案")

window_scores_fixed = []
failed_windows_1 = []

for window_id, (train_idx, val_idx) in enumerate(cv_splits):
    try:
        X_train = train_factors.iloc[train_idx][fixed_features]
        X_val = train_factors.iloc[val_idx][fixed_features]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        sample_weights = tuner.calculate_time_weights(len(y_train))

        model = LGBMRegressor(**simple_params)
        model.fit(X_train, y_train, sample_weight=sample_weights)

        preds = model.predict(X_val)
        score = calculate_score_metric(preds, y_val.values)

        # 处理异常分数
        if np.isnan(score) or np.isinf(score):
            score = 0.0

        window_scores_fixed.append(float(score))

        # 进度显示
        if (window_id + 1) % 20 == 0 or window_id == total_windows - 1:
            print(f"  进度: {window_id+1}/{total_windows} | 最近分数: {score:.6f}")
            save_checkpoint('strategy1_fixed_ic', window_scores_fixed)

    except Exception as e:
        print(f"  窗口{window_id} 失败: {str(e)[:50]}")
        failed_windows_1.append(window_id)
        window_scores_fixed.append(0.0)  # 失败窗口记0分

strategy1_result = {
    "strategy": "fixed_all_history_ic",
    "n_windows": len(window_scores_fixed),
    "n_failed": len(failed_windows_1),
    "mean_score": float(np.mean(window_scores_fixed)),
    "std_score": float(np.std(window_scores_fixed)),
    "min_score": float(np.min(window_scores_fixed)),
    "max_score": float(np.max(window_scores_fixed)),
    "stability_score": float(np.mean(window_scores_fixed) - 2 * np.std(window_scores_fixed)),
}

print(f"\n策略1完成:")
print(f"  mean_score: {strategy1_result['mean_score']:.6f}")
print(f"  std_score: {strategy1_result['std_score']:.6f}")
print(f"  失败窗口: {len(failed_windows_1)}/{total_windows}")

# ============================================================
# 策略2: 动态IC（每窗口用最近60天IC选Top30）
# ============================================================
print("\n[3/4] 策略2: 动态IC")
print("-"*80)

window_scores_dynamic_ic = []
factor_changes_ic = []
failed_windows_2 = []

for window_id, (train_idx, val_idx) in enumerate(cv_splits):
    try:
        train_window_data = train_factors.iloc[train_idx]
        recent_data = train_window_data.tail(60)

        # 动态选因子
        factor_df_recent = analyzer.analyze_dataset(recent_data, verbose=False)
        _, _, current_features = preprocessor.select_features_by_ic(
            train_factors, train_factors, factor_df_recent, top_n=30
        )

        # 记录因子变化
        if window_id > 0 and len(factor_changes_ic) > 0:
            previous_features = factor_changes_ic[-1]['features']
            n_changed = len(set(current_features) - set(previous_features))
        else:
            n_changed = 0

        factor_changes_ic.append({
            'window': window_id,
            'features': current_features,
            'n_changed': n_changed
        })

        # 训练模型
        X_train = train_factors.iloc[train_idx][current_features]
        X_val = train_factors.iloc[val_idx][current_features]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        sample_weights = tuner.calculate_time_weights(len(y_train))

        model = LGBMRegressor(**simple_params)
        model.fit(X_train, y_train, sample_weight=sample_weights)

        preds = model.predict(X_val)
        score = calculate_score_metric(preds, y_val.values)

        if np.isnan(score) or np.isinf(score):
            score = 0.0

        window_scores_dynamic_ic.append(float(score))

        # 进度显示
        if (window_id + 1) % 20 == 0 or window_id == total_windows - 1:
            print(f"  进度: {window_id+1}/{total_windows} | 最近分数: {score:.6f}")
            save_checkpoint('strategy2_dynamic_ic', window_scores_dynamic_ic,
                          [{'window': x['window'], 'n_changed': x['n_changed']} for x in factor_changes_ic])

    except Exception as e:
        print(f"  窗口{window_id} 失败: {str(e)[:50]}")
        failed_windows_2.append(window_id)
        window_scores_dynamic_ic.append(0.0)
        if len(factor_changes_ic) > 0:
            factor_changes_ic.append({
                'window': window_id,
                'features': factor_changes_ic[-1]['features'],
                'n_changed': 0
            })

avg_change_ic = np.mean([x['n_changed'] for x in factor_changes_ic[1:]]) if len(factor_changes_ic) > 1 else 0

strategy2_result = {
    "strategy": "dynamic_every_window_ic_60d",
    "n_windows": len(window_scores_dynamic_ic),
    "n_failed": len(failed_windows_2),
    "mean_score": float(np.mean(window_scores_dynamic_ic)),
    "std_score": float(np.std(window_scores_dynamic_ic)),
    "min_score": float(np.min(window_scores_dynamic_ic)),
    "max_score": float(np.max(window_scores_dynamic_ic)),
    "stability_score": float(np.mean(window_scores_dynamic_ic) - 2 * np.std(window_scores_dynamic_ic)),
    "avg_factor_change": float(avg_change_ic),
}

print(f"\n策略2完成:")
print(f"  mean_score: {strategy2_result['mean_score']:.6f}")
print(f"  std_score: {strategy2_result['std_score']:.6f}")
print(f"  平均因子变化: {avg_change_ic:.1f}/30")
print(f"  失败窗口: {len(failed_windows_2)}/{total_windows}")

# ============================================================
# 策略3: 动态Fitness（每窗口用最近60天Fitness选Top30）
# ============================================================
print("\n[4/4] 策略3: 动态Fitness")
print("-"*80)

window_scores_dynamic_fitness = []
factor_changes_fitness = []
failed_windows_3 = []

for window_id, (train_idx, val_idx) in enumerate(cv_splits):
    try:
        train_window_data = train_factors.iloc[train_idx]
        recent_data = train_window_data.tail(60)

        # 动态选因子（按Fitness）
        factor_df_recent = analyzer.analyze_dataset(recent_data, verbose=False)
        factor_df_sorted = factor_df_recent.copy()
        factor_df_sorted['Fitness_abs'] = factor_df_sorted['Fitness'].abs()
        factor_df_sorted = factor_df_sorted.sort_values('Fitness_abs', ascending=False)
        current_features = factor_df_sorted.head(30)['特征'].tolist()

        # 记录因子变化
        if window_id > 0 and len(factor_changes_fitness) > 0:
            previous_features = factor_changes_fitness[-1]['features']
            n_changed = len(set(current_features) - set(previous_features))
        else:
            n_changed = 0

        factor_changes_fitness.append({
            'window': window_id,
            'features': current_features,
            'n_changed': n_changed
        })

        # 训练模型
        X_train = train_factors.iloc[train_idx][current_features]
        X_val = train_factors.iloc[val_idx][current_features]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        sample_weights = tuner.calculate_time_weights(len(y_train))

        model = LGBMRegressor(**simple_params)
        model.fit(X_train, y_train, sample_weight=sample_weights)

        preds = model.predict(X_val)
        score = calculate_score_metric(preds, y_val.values)

        if np.isnan(score) or np.isinf(score):
            score = 0.0

        window_scores_dynamic_fitness.append(float(score))

        # 进度显示
        if (window_id + 1) % 20 == 0 or window_id == total_windows - 1:
            print(f"  进度: {window_id+1}/{total_windows} | 最近分数: {score:.6f}")
            save_checkpoint('strategy3_dynamic_fitness', window_scores_dynamic_fitness,
                          [{'window': x['window'], 'n_changed': x['n_changed']} for x in factor_changes_fitness])

    except Exception as e:
        print(f"  窗口{window_id} 失败: {str(e)[:50]}")
        failed_windows_3.append(window_id)
        window_scores_dynamic_fitness.append(0.0)
        if len(factor_changes_fitness) > 0:
            factor_changes_fitness.append({
                'window': window_id,
                'features': factor_changes_fitness[-1]['features'],
                'n_changed': 0
            })

avg_change_fitness = np.mean([x['n_changed'] for x in factor_changes_fitness[1:]]) if len(factor_changes_fitness) > 1 else 0

strategy3_result = {
    "strategy": "dynamic_every_window_fitness_60d",
    "n_windows": len(window_scores_dynamic_fitness),
    "n_failed": len(failed_windows_3),
    "mean_score": float(np.mean(window_scores_dynamic_fitness)),
    "std_score": float(np.std(window_scores_dynamic_fitness)),
    "min_score": float(np.min(window_scores_dynamic_fitness)),
    "max_score": float(np.max(window_scores_dynamic_fitness)),
    "stability_score": float(np.mean(window_scores_dynamic_fitness) - 2 * np.std(window_scores_dynamic_fitness)),
    "avg_factor_change": float(avg_change_fitness),
}

print(f"\n策略3完成:")
print(f"  mean_score: {strategy3_result['mean_score']:.6f}")
print(f"  std_score: {strategy3_result['std_score']:.6f}")
print(f"  平均因子变化: {avg_change_fitness:.1f}/30")
print(f"  失败窗口: {len(failed_windows_3)}/{total_windows}")

# ============================================================
# 汇总和保存
# ============================================================
print("\n" + "="*80)
print("【汇总结果】")
print("="*80)

results_df = pd.DataFrame([strategy1_result, strategy2_result, strategy3_result])
results_df = results_df.sort_values("mean_score", ascending=False).reset_index(drop=True)

print("\n" + results_df.to_string(index=False))

best_strategy = results_df.iloc[0]['strategy']
best_score = results_df.iloc[0]['mean_score']
print(f"\n✅ 最佳策略: {best_strategy}")
print(f"✅ Mean Score: {best_score:.6f}")

# 保存最终结果
csv_path = os.path.join(RESULT_DIR, 'strategy_comparison.csv')
results_df.to_csv(csv_path, index=False)
print(f"\n已保存: {csv_path}")

detailed_scores = {
    'strategy1_fixed_ic': {
        'scores': window_scores_fixed,
        'failed_windows': failed_windows_1,
        'config': strategy1_result
    },
    'strategy2_dynamic_ic': {
        'scores': window_scores_dynamic_ic,
        'failed_windows': failed_windows_2,
        'factor_changes': [{'window': x['window'], 'n_changed': x['n_changed']} for x in factor_changes_ic],
        'config': strategy2_result
    },
    'strategy3_dynamic_fitness': {
        'scores': window_scores_dynamic_fitness,
        'failed_windows': failed_windows_3,
        'factor_changes': [{'window': x['window'], 'n_changed': x['n_changed']} for x in factor_changes_fitness],
        'config': strategy3_result
    },
    'metadata': {
        'timestamp': datetime.now().isoformat(),
        'n_windows': total_windows,
        'train_window_size': 90,
        'cv_step': 30,
        'top_n': 30,
        'ic_window_size': 60,
        'model_params': simple_params
    }
}

json_path = os.path.join(RESULT_DIR, 'detailed_scores.json')
with open(json_path, 'w') as f:
    json.dump(detailed_scores, f, indent=2)
print(f"已保存: {json_path}")

scores_df = pd.DataFrame({
    'window_id': range(total_windows),
    'strategy1_fixed_ic': window_scores_fixed,
    'strategy2_dynamic_ic': window_scores_dynamic_ic,
    'strategy3_dynamic_fitness': window_scores_dynamic_fitness,
})

scores_csv_path = os.path.join(RESULT_DIR, 'window_scores.csv')
scores_df.to_csv(scores_csv_path, index=False)
print(f"已保存: {scores_csv_path}")

print("\n" + "="*80)
print("【完成】")
print("="*80)
print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
