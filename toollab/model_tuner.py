"""
【模型调参器】Model Tuner
使用Optuna进行自动超参数搜索
支持LightGBM和CatBoost
支持滑动窗口CV和时间加权训练
"""

import time
import numpy as np
import pandas as pd
import optuna
import joblib
import itertools
from typing import Dict, Any, List, Optional
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from .utils import timer, safe_spearman, sliding_window_cv_splits, analyze_lgbm_tree_complexity
from .nn_models import create_tabular_mlp
import torch


def calculate_score_metric(y_pred, y_true, risk_free_rate=0.0):
    """
    使用竞赛官方评分函数计算调整后的夏普比率

    Args:
        y_pred: 预测的超额收益（连续值）
        y_true: 真实的超额收益
        risk_free_rate: 无风险利率（默认0，假设y_true已经是超额收益）

    Returns:
        adjusted_sharpe: 调整后的夏普比率（越高越好）

    Notes:
        - 仓位转换规则：pred <= 0 → 0仓, 0 < pred <= 0.001 → 50%仓, pred > 0.001 → 满仓
        - 包含波动率惩罚（超过市场1.2倍）和收益惩罚（低于市场）
        - 用于替代Spearman作为模型训练的优化目标
    """
    # 输入验证
    if len(y_pred) == 0 or len(y_true) == 0:
        return 0.0

    # 1. 仓位转换：基于预测值的简单规则
    # pred <= 0 → position 0 (空仓，全无风险资产)
    # 0 < pred <= 0.001 → position 1 (中性，50%股票)
    # pred > 0.001 → position 2 (满仓，200%杠杆)
    position = np.where(y_pred > 0.001, 2.0,
                       np.where(y_pred > 0, 1.0, 0.0))

    # 2. 反推forward_returns (因为y_true是超额收益)
    forward_returns = y_true + risk_free_rate

    # 3. 计算策略收益
    # 策略收益 = 无风险收益×(1-仓位) + 市场收益×仓位
    strategy_returns = (
        risk_free_rate * (1 - position) +
        position * forward_returns
    )

    # 4. 计算策略超额收益
    strategy_excess_returns = strategy_returns - risk_free_rate
    strategy_excess_cumulative = (1 + strategy_excess_returns).prod()

    if strategy_excess_cumulative <= 0:
        return 0.0

    strategy_mean_excess_return = (strategy_excess_cumulative) ** (1 / len(y_pred)) - 1

    # 5. 计算策略标准差
    strategy_std = strategy_returns.std()

    if strategy_std == 0 or strategy_std < 1e-8:
        return 0.0

    # 6. 计算年化夏普比率
    trading_days_per_yr = 252
    sharpe = strategy_mean_excess_return / strategy_std * np.sqrt(trading_days_per_yr)

    # 7. 计算策略年化波动率
    strategy_volatility = float(strategy_std * np.sqrt(trading_days_per_yr) * 100)

    # 8. 计算市场基准指标
    market_excess_returns = forward_returns - risk_free_rate
    market_excess_cumulative = (1 + market_excess_returns).prod()

    if market_excess_cumulative <= 0:
        return sharpe  # 如果市场收益为负，直接返回sharpe

    market_mean_excess_return = (market_excess_cumulative) ** (1 / len(y_pred)) - 1
    market_std = forward_returns.std()

    if market_std == 0:
        return sharpe

    market_volatility = float(market_std * np.sqrt(trading_days_per_yr) * 100)

    # 9. 计算波动率惩罚（如果策略波动率超过市场1.2倍）
    excess_vol = max(0, strategy_volatility / market_volatility - 1.2) if market_volatility > 0 else 0
    vol_penalty = 1 + excess_vol

    # 10. 计算收益差距惩罚（如果策略收益低于市场）
    return_gap = max(0, (market_mean_excess_return - strategy_mean_excess_return) * 100 * trading_days_per_yr)
    return_penalty = 1 + (return_gap**2) / 100

    # 11. 计算最终的调整后夏普比率
    adjusted_sharpe = sharpe / (vol_penalty * return_penalty)

    return min(float(adjusted_sharpe), 1_000_000)


class ModelTuner:
    """
    【核心类】ModelTuner - Optuna自动调参器

    这个类使用Optuna框架自动搜索最优超参数
    Optuna vs 传统调参:
    - 网格搜索: 盲目尝试所有组合 (慢且低效)
    - Optuna: 贝叶斯优化，根据历史结果智能选择
    """

    def __init__(self, seed: int = 42, n_splits: int = 4,
                 use_sliding_window: bool = True,
                 train_window_size: int = 90,
                 time_decay: float = 0.95,
                 cv_step: int = 10,
                 use_reverse_cv: bool = True,
                 use_pruning: bool = True,
                 pruning_steps: list = None,
                 pruning_threshold: float = 0.005,
                 cv_time_decay: float = 0.995):
        """
        【功能】初始化ModelTuner对象

        【参数】
            seed (int): 随机种子，保证结果可复现
            n_splits (int): 交叉验证折数（仅用于传统CV）
            use_sliding_window (bool): 是否使用滑动窗口CV
            train_window_size (int): 滑动窗口大小
            time_decay (float): 时间衰减因子（用于样本加权）
            cv_step (int): CV滑动步长（每多少天一个测试窗口）
            use_reverse_cv (bool): 是否倒序CV（从新到老）
            use_pruning (bool): 是否启用early stopping
            pruning_steps (list): 在哪些步骤检查是否pruning，如[50, 200]
            pruning_threshold (float): Pruning阈值，低于此值淘汰
            cv_time_decay (float): CV窗口时间衰减（用于窗口加权）
        """
        self.seed = seed
        self.n_splits = n_splits
        self.use_sliding_window = use_sliding_window
        self.train_window_size = train_window_size
        self.time_decay = time_decay
        self.cv_step = cv_step
        self.use_reverse_cv = use_reverse_cv
        self.use_pruning = use_pruning
        self.pruning_steps = pruning_steps if pruning_steps else [50, 200]
        self.pruning_threshold = pruning_threshold
        self.cv_time_decay = cv_time_decay

    def calculate_time_weights(self, n_samples):
        """
        计算时间加权 - 新样本权重高，旧样本权重低

        Args:
            n_samples: 样本数量

        Returns:
            weights: 时间权重数组 (归一化后总和=n_samples)

        Example:
            如果 time_decay=0.95, n_samples=100:
            weights[0] (最老) = 0.95^99
            weights[99] (最新) = 0.95^0 = 1.0
        """
        weights = np.array([self.time_decay ** (n_samples - i - 1) for i in range(n_samples)])
        # 归一化: 让sum(weights) = n_samples (方便与sklearn的sample_weight配合)
        return weights / weights.sum() * n_samples

    @timer
    def tune_lightgbm(self, X: pd.DataFrame, y: pd.Series, n_trials: int = 30, n_jobs: int = 1) -> optuna.study.Study:
        """
        【功能】使用Optuna自动搜索LightGBM的最优超参数

        【参数】
            X (pd.DataFrame): 特征矩阵
            y (pd.Series): 目标变量
            n_trials (int): 尝试多少组不同的参数 (10=快速测试, 50-100=更好效果)
            n_jobs (int): 并行任务数 (1=串行, -1=用所有CPU核心)

        【返回】
            optuna.study.Study: 包含最优参数和所有试验历史

        【工作原理】
            1. Optuna随机选择一组参数
            2. 用这组参数训练模型，做交叉验证
            3. 计算平均Spearman分数
            4. Optuna根据这个分数，智能选择下一组参数
            5. 重复n_trials次
            6. 返回得分最高的参数
        """
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def objective(trial):
            # 针对小窗口优化的参数空间（90样本 × 50特征）
            params = {
                "objective": "regression",
                "metric": "rmse",
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),  # 增加树数量
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),  # 适中学习率
                "max_depth": trial.suggest_int("max_depth", 3, 6),  # 限制深度防止过拟合
                "num_leaves": trial.suggest_int("num_leaves", 8, 64),  # 适中叶子数
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0001, 0.5, log=True),  # 降低L2正则化
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0001, 0.5, log=True),  # 降低L1正则化
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),  # 提高列采样
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),  # 提高行采样
                "subsample_freq": 1,
                "min_child_samples": trial.suggest_int("min_child_samples", 1, 5),  # 关键：大幅降低（90/5=18）
                "min_child_weight": trial.suggest_float("min_child_weight", 0.001, 0.1, log=True),  # 降低权重阈值
                "random_state": self.seed,
                "verbosity": -1
            }

            window_scores = []
            fold_metrics = {
                'ic': [],            # Pearson IC（线性相关）
                'ric': [],           # Rank IC（Spearman）
                'avg_depth': [],     # 每个窗口 LightGBM 平均树深度
                'avg_leaves': [],    # 每个窗口 LightGBM 平均叶子数
            }

            # 选择CV策略
            if self.use_sliding_window:
                # 真正的在线学习CV（滑动窗口）
                from .utils import true_online_cv_splits
                cv_splits = list(true_online_cv_splits(X, train_size=self.train_window_size, step=self.cv_step))

                # 如果启用倒序，从新到老遍历
                if self.use_reverse_cv:
                    cv_splits = list(reversed(cv_splits))
            else:
                # 传统时间序列CV (扩展窗口)
                tscv = TimeSeriesSplit(n_splits=self.n_splits)
                cv_splits = list(tscv.split(X))

            evaluated_count = 0

            for train_idx, val_idx in cv_splits:
                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]
                X_val = X.iloc[val_idx]
                y_val = y.iloc[val_idx]

                # 计算时间权重（新样本权重高）
                sample_weights = self.calculate_time_weights(len(y_train))

                model = LGBMRegressor(**params)
                model.fit(
                    X_train, y_train,
                    sample_weight=sample_weights
                    # 注释掉 early_stopping：对于10样本验证集不适用
                    # eval_set=[(X_val, y_val)],
                    # callbacks=[early_stopping(stopping_rounds=50, verbose=False)]
                )

                # ========= 新增：树结构诊断 =========
                tree_stats = analyze_lgbm_tree_complexity(model)
                fold_metrics['avg_depth'].append(tree_stats['avg_depth'])
                fold_metrics['avg_leaves'].append(tree_stats['avg_leaves'])

                # 如果平均深度 / 叶子数明显偏低，给出提示（"树无法再分裂"的直接信号）
                if tree_stats['avg_depth'] < 2.0 or tree_stats['avg_leaves'] < 4.0:
                    print(
                        f"    [警告] 当前窗口树结构过于简单："
                        f"avg_depth={tree_stats['avg_depth']:.2f}, avg_leaves={tree_stats['avg_leaves']:.2f} "
                        f"(max_depth={params['max_depth']}, num_leaves={params['num_leaves']})"
                    )

                preds = model.predict(X_val)

                # 诊断：检查预测和真实值的分布（前5个窗口）
                if evaluated_count < 5:
                    print(f"\n  [诊断] Window {evaluated_count + 1}:")
                    print(f"    训练集: {len(X_train)}样本 × {X_train.shape[1]}特征")
                    print(f"    验证集: {len(X_val)}样本")
                    print(f"    y_val分布: min={y_val.min():.6f}, max={y_val.max():.6f}, std={y_val.std():.6f}")
                    print(f"    预测分布: min={preds.min():.6f}, max={preds.max():.6f}, std={preds.std():.6f}")
                    print(f"    唯一预测值数: {len(np.unique(preds))}")

                    # 检查训练特征是否有问题
                    n_const_cols = sum(1 for col in X_train.columns if X_train[col].nunique() <= 1)
                    print(f"    常数特征数: {n_const_cols}/{X_train.shape[1]}")
                    print(
                        f"    树复杂度: n_trees={tree_stats['n_trees']}, "
                        f"avg_depth={tree_stats['avg_depth']:.2f} (max={tree_stats['max_depth']}), "
                        f"avg_leaves={tree_stats['avg_leaves']:.2f} (max={tree_stats['max_leaves']})"
                    )

                # 主指标：竞赛官方ScoreMetric（调整后夏普比率）
                spearman_score = calculate_score_metric(preds, y_val.values)
                window_scores.append(spearman_score)

                # 辅助指标：诊断用
                from scipy.stats import pearsonr
                ic, _ = pearsonr(y_val.values, preds)
                fold_metrics['ic'].append(ic if not np.isnan(ic) else 0.0)
                fold_metrics['ric'].append(spearman_score)

                evaluated_count += 1

                # Early Stopping / Pruning检查
                if self.use_pruning and evaluated_count in self.pruning_steps:
                    # 计算到目前为止的平均Spearman
                    current_avg = np.mean(window_scores)

                    # 调试：打印分数分布（首次检查时）
                    if evaluated_count == self.pruning_steps[0]:
                        print(f"\n  [Trial {trial.number}] 第{evaluated_count}个窗口检查点:")
                        print(f"    平均Spearman: {current_avg:.6f}")
                        print(f"    最小: {np.min(window_scores):.6f}, 最大: {np.max(window_scores):.6f}")
                        print(f"    零/负数窗口: {sum(1 for s in window_scores if s <= 0)}/{len(window_scores)}")
                        print(f"    阈值: {self.pruning_threshold:.6f}")

                    # 如果表现很差，提前淘汰这组参数
                    if current_avg < self.pruning_threshold:
                        trial.set_user_attr(f'pruned_at_step', evaluated_count)
                        trial.set_user_attr(f'pruned_score', current_avg)
                        print(f"    → 剪枝！({current_avg:.6f} < {self.pruning_threshold:.6f})")
                        raise optuna.TrialPruned()

                    # 报告中间结果给Optuna（用于MedianPruner）
                    trial.report(current_avg, evaluated_count)

            # 计算时间加权得分
            # 如果倒序：window_scores[0]=最新窗口（最重要），window_scores[-1]=最老窗口
            # 如果正序：window_scores[0]=最老窗口，window_scores[-1]=最新窗口（最重要）
            n_windows = len(window_scores)

            if self.use_reverse_cv:
                # 倒序：第i个窗口的权重 = decay^i（i=0最新，权重最高）
                time_weights = np.array([self.cv_time_decay ** i for i in range(n_windows)])
            else:
                # 正序：第i个窗口的权重 = decay^(n-i-1)（i=n-1最新，权重最高）
                time_weights = np.array([self.cv_time_decay ** (n_windows - i - 1) for i in range(n_windows)])

            time_weights = time_weights / time_weights.sum()  # 归一化

            # 加权平均（主指标）
            weighted_spearman = float(np.average(window_scores, weights=time_weights))

            # 简单平均（辅助指标）
            mean_spearman = float(np.mean(window_scores))
            std_spearman = float(np.std(window_scores))
            min_spearman = float(np.min(window_scores))
            mean_ic = float(np.mean(fold_metrics['ic']))

            # 计算IC Information Ratio（类似Sharpe Ratio）
            ic_ir = mean_ic / (std_spearman + 1e-8)

            # ========= 新增：模型复杂度惩罚 =========
            # 计算平均树深度和叶子数
            avg_tree_depth = float(np.mean(fold_metrics['avg_depth']))
            avg_tree_leaves = float(np.mean(fold_metrics['avg_leaves']))

            # 计算复杂度利用率（实际复杂度 / 最大复杂度）
            max_depth = params['max_depth']
            max_leaves = params['num_leaves']
            depth_ratio = avg_tree_depth / max_depth if max_depth > 0 else 0.0
            leaf_ratio = avg_tree_leaves / max_leaves if max_leaves > 0 else 0.0
            complexity_ratio = 0.5 * (depth_ratio + leaf_ratio)

            # 复杂度惩罚：如果树结构过于简单（利用率<50%），降低分数
            # final_score = weighted_spearman * (0.5 + 0.5 * complexity_ratio)
            # 这样：complexity_ratio=1 → final_score=weighted_spearman（无惩罚）
            #       complexity_ratio=0.5 → final_score=0.75*weighted_spearman（轻微惩罚）
            #       complexity_ratio=0 → final_score=0.5*weighted_spearman（重度惩罚）
            final_score = weighted_spearman * (0.5 + 0.5 * complexity_ratio)

            # 保存辅助指标到trial（用于后续分析）
            trial.set_user_attr('weighted_spearman', weighted_spearman)
            trial.set_user_attr('mean_spearman', mean_spearman)
            trial.set_user_attr('std_spearman', std_spearman)
            trial.set_user_attr('min_spearman', min_spearman)
            trial.set_user_attr('mean_ic', mean_ic)
            trial.set_user_attr('ic_ir', ic_ir)
            trial.set_user_attr('stability_score', mean_spearman - 2 * std_spearman)
            trial.set_user_attr('n_windows_evaluated', n_windows)
            trial.set_user_attr('avg_tree_depth', avg_tree_depth)
            trial.set_user_attr('avg_tree_leaves', avg_tree_leaves)
            trial.set_user_attr('complexity_ratio', complexity_ratio)
            trial.set_user_attr('final_score', final_score)

            # 返回复杂度惩罚后的最终分数（Optuna优化目标）
            return final_score

        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)
        return study

    @timer
    def tune_catboost(self, X: pd.DataFrame, y: pd.Series, n_trials: int = 30,
                     use_pool: bool = False, task_type: str = "GPU") -> optuna.study.Study:
        """
        【功能】使用Optuna自动搜索CatBoost的最优超参数

        【参数】
            X (pd.DataFrame): 特征矩阵
            y (pd.Series): 目标变量
            n_trials (int): 尝试次数
            use_pool (bool): 是否使用CatBoost的Pool格式
            task_type (str): "CPU" 或 "GPU"

        【返回】
            optuna.study.Study: 包含最优参数和试验历史
        """
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def objective(trial):
            # 针对小窗口优化的参数空间
            params = {
                "loss_function": "RMSE",
                "iterations": trial.suggest_int("iterations", 50, 300),  # 增加迭代次数
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "depth": trial.suggest_int("depth", 3, 8),  # 允许更深的树
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.01, 5.0, log=True),  # 降低L2正则化
                "random_strength": trial.suggest_float("random_strength", 0.01, 2.0, log=True),  # 降低随机强度
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),  # 降低bagging温度
                "random_seed": self.seed,
                "verbose": False,
                "bootstrap_type": "MVS",
            }

            if task_type.upper() == "GPU":
                params["task_type"] = "GPU"
                params["devices"] = "0"

            window_scores = []
            fold_metrics = {
                'ic': [],           # Pearson IC（线性相关）
                'ric': [],          # Rank IC（Spearman）
            }

            # 选择CV策略
            if self.use_sliding_window:
                # 真正的在线学习CV（滑动窗口）
                from .utils import true_online_cv_splits
                cv_splits = list(true_online_cv_splits(X, train_size=self.train_window_size, step=self.cv_step))

                # 如果启用倒序，从新到老遍历
                if self.use_reverse_cv:
                    cv_splits = list(reversed(cv_splits))
            else:
                # 传统时间序列CV (扩展窗口)
                tscv = TimeSeriesSplit(n_splits=self.n_splits)
                cv_splits = list(tscv.split(X))

            evaluated_count = 0

            for train_idx, val_idx in cv_splits:
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # 计算时间权重
                sample_weights = self.calculate_time_weights(len(y_train))

                if use_pool:
                    pool_train = Pool(X_train, label=y_train, weight=sample_weights)
                    model = CatBoostRegressor(**params)
                    model.fit(pool_train, verbose=False)
                    # 禁用 early_stopping：对于10样本验证集不适用
                else:
                    model = CatBoostRegressor(**params)
                    model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)
                    # 禁用 early_stopping：对于10样本验证集不适用

                preds = model.predict(X_val)

                # 诊断：检查预测和真实值的分布（前5个窗口）
                if evaluated_count < 5:
                    print(f"\n  [诊断] Window {evaluated_count + 1}:")
                    print(f"    训练集: {len(X_train)}样本 × {X_train.shape[1]}特征")
                    print(f"    验证集: {len(X_val)}样本")
                    print(f"    y_val分布: min={y_val.min():.6f}, max={y_val.max():.6f}, std={y_val.std():.6f}")
                    print(f"    预测分布: min={preds.min():.6f}, max={preds.max():.6f}, std={preds.std():.6f}")
                    print(f"    唯一预测值数: {len(np.unique(preds))}")

                    # 检查训练特征是否有问题
                    n_const_cols = sum(1 for col in X_train.columns if X_train[col].nunique() <= 1)
                    print(f"    常数特征数: {n_const_cols}/{X_train.shape[1]}")

                # 主指标：竞赛官方ScoreMetric（调整后夏普比率）
                spearman_score = calculate_score_metric(preds, y_val.values)
                window_scores.append(spearman_score)

                # 辅助指标
                from scipy.stats import pearsonr
                ic, _ = pearsonr(y_val.values, preds)
                fold_metrics['ic'].append(ic if not np.isnan(ic) else 0.0)
                fold_metrics['ric'].append(spearman_score)

                evaluated_count += 1

                # Early Stopping / Pruning检查
                if self.use_pruning and evaluated_count in self.pruning_steps:
                    current_avg = np.mean(window_scores)

                    # 调试：打印分数分布（首次检查时）
                    if evaluated_count == self.pruning_steps[0]:
                        print(f"\n  [Trial {trial.number}] 第{evaluated_count}个窗口检查点:")
                        print(f"    平均Spearman: {current_avg:.6f}")
                        print(f"    最小: {np.min(window_scores):.6f}, 最大: {np.max(window_scores):.6f}")
                        print(f"    零/负数窗口: {sum(1 for s in window_scores if s <= 0)}/{len(window_scores)}")
                        print(f"    阈值: {self.pruning_threshold:.6f}")

                    if current_avg < self.pruning_threshold:
                        trial.set_user_attr(f'pruned_at_step', evaluated_count)
                        trial.set_user_attr(f'pruned_score', current_avg)
                        print(f"    → 剪枝！({current_avg:.6f} < {self.pruning_threshold:.6f})")
                        raise optuna.TrialPruned()

                    trial.report(current_avg, evaluated_count)

            # 计算时间加权得分
            n_windows = len(window_scores)

            if self.use_reverse_cv:
                time_weights = np.array([self.cv_time_decay ** i for i in range(n_windows)])
            else:
                time_weights = np.array([self.cv_time_decay ** (n_windows - i - 1) for i in range(n_windows)])

            time_weights = time_weights / time_weights.sum()

            # 加权平均（主指标）
            weighted_spearman = float(np.average(window_scores, weights=time_weights))

            # 简单平均（辅助指标）
            mean_spearman = float(np.mean(window_scores))
            std_spearman = float(np.std(window_scores))
            min_spearman = float(np.min(window_scores))
            mean_ic = float(np.mean(fold_metrics['ic']))
            ic_ir = mean_ic / (std_spearman + 1e-8)

            # 保存辅助指标
            trial.set_user_attr('weighted_spearman', weighted_spearman)
            trial.set_user_attr('mean_spearman', mean_spearman)
            trial.set_user_attr('std_spearman', std_spearman)
            trial.set_user_attr('min_spearman', min_spearman)
            trial.set_user_attr('mean_ic', mean_ic)
            trial.set_user_attr('ic_ir', ic_ir)
            trial.set_user_attr('stability_score', mean_spearman - 2 * std_spearman)
            trial.set_user_attr('n_windows_evaluated', n_windows)

            # 返回加权Spearman
            return weighted_spearman

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        return study

    @timer
    def train_final_lgbm(self, X: pd.DataFrame, y: pd.Series, best_params: Dict[str, Any], model_path: str):
        """
        【功能】用Optuna找到的最优参数训练最终LightGBM模型

        【参数】
            X (pd.DataFrame): 全部训练数据的特征
            y (pd.Series): 全部训练数据的目标
            best_params (dict): Optuna找到的最优参数
            model_path (str): 模型保存路径

        【返回】
            训练好的LightGBM模型
        """
        p = dict(best_params)
        p.setdefault("random_state", self.seed)
        p.setdefault("verbosity", -1)

        # 计算时间权重
        sample_weights = self.calculate_time_weights(len(y))

        model = LGBMRegressor(**p)
        model.fit(X, y, sample_weight=sample_weights)
        joblib.dump(model, model_path)
        return model

    @timer
    def train_final_catboost(self, X: pd.DataFrame, y: pd.Series, best_params: Dict[str, Any], model_path: str):
        """
        【功能】用Optuna找到的最优参数训练最终CatBoost模型

        【参数】
            X (pd.DataFrame): 全部训练数据的特征
            y (pd.Series): 全部训练数据的目标
            best_params (dict): Optuna找到的最优参数
            model_path (str): 模型保存路径 (.cbm格式)

        【返回】
            训练好的CatBoost模型
        """
        p = dict(best_params)
        p.setdefault("random_seed", self.seed)
        p.setdefault("verbose", False)

        # 计算时间权重
        sample_weights = self.calculate_time_weights(len(y))

        model = CatBoostRegressor(**p)
        model.fit(X, y, sample_weight=sample_weights, use_best_model=False, verbose=False)
        model.save_model(model_path)
        return model

    @timer
    def train_final_mlp_tabular(self, X: pd.DataFrame, y: pd.Series, best_params: Dict[str, Any], model_path: str):
        """
        【功能】训练最终的表格MLP模型（用于集成学习）

        【参数】
            X (pd.DataFrame): 全部训练数据的特征
            y (pd.Series): 全部训练数据的目标
            best_params (dict): MLP超参数（如hidden_dims, dropout, lr等）
            model_path (str): 模型保存路径

        【返回】
            训练好的MLP模型
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        # 计算时间权重
        sample_weights = self.calculate_time_weights(len(y))

        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X.values)
        y_tensor = torch.FloatTensor(y.values).reshape(-1, 1)
        weights_tensor = torch.FloatTensor(sample_weights)

        # 创建数据集和加载器
        dataset = TensorDataset(X_tensor, y_tensor, weights_tensor)
        batch_size = best_params.get('batch_size', 32)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # 创建模型
        input_dim = X.shape[1]
        hidden_dims = best_params.get('hidden_dims', [128, 64, 32])
        dropout = best_params.get('dropout', 0.2)
        model = create_tabular_mlp(input_dim, hidden_dims, dropout)

        # 训练配置
        lr = best_params.get('lr', 0.001)
        epochs = best_params.get('epochs', 100)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss(reduction='none')

        # 训练循环
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for X_batch, y_batch, w_batch in dataloader:
                optimizer.zero_grad()
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                weighted_loss = (loss * w_batch.reshape(-1, 1)).mean()
                weighted_loss.backward()
                optimizer.step()
                epoch_loss += weighted_loss.item()

            if (epoch + 1) % 20 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.6f}")

        # 保存模型
        torch.save(model.state_dict(), model_path)
        return model

    @staticmethod
    def compute_ensemble_weights_from_scores(scores: List[float]) -> np.ndarray:
        """
        【功能】根据模型的验证集得分计算集成权重

        【参数】
            scores (list): 各模型的验证集Spearman分数

        【返回】
            weights (np.ndarray): 归一化的集成权重

        【示例】
            scores = [0.15, 0.12, 0.18]  # LGBM, CatBoost, MLP
            weights = [0.3, 0.2, 0.5]  # MLP表现最好，权重最高
        """
        scores = np.array(scores)
        # 处理负分数：shift to positive
        if scores.min() < 0:
            scores = scores - scores.min() + 1e-6

        # Softmax权重（温度参数=1）
        exp_scores = np.exp(scores)
        weights = exp_scores / exp_scores.sum()
        return weights

    @staticmethod
    def blend_predictions(pred_list: List[np.ndarray], weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        【功能】融合多个模型的预测结果

        【参数】
            pred_list (list): 各模型的预测数组
            weights (np.ndarray): 各模型的权重（如果None，则均权重）

        【返回】
            blended_pred (np.ndarray): 加权平均后的预测

        【示例】
            pred_lgbm = [0.01, 0.02, -0.01]
            pred_cat = [0.015, 0.018, -0.008]
            pred_mlp = [0.012, 0.022, -0.012]
            blend = blend_predictions([pred_lgbm, pred_cat, pred_mlp], weights=[0.3, 0.3, 0.4])
        """
        if weights is None:
            weights = np.ones(len(pred_list)) / len(pred_list)

        blended = np.zeros_like(pred_list[0])
        for pred, w in zip(pred_list, weights):
            blended += w * pred

        return blended

    @staticmethod
    def _analyze_lgbm_tree_complexity(model: LGBMRegressor) -> Dict[str, float]:
        """
        【功能】分析LightGBM模型的树结构复杂度（内部方法）

        【参数】
            model (LGBMRegressor): 训练好的LightGBM模型

        【返回】
            stats (dict): 包含avg_depth, avg_leaves等统计信息

        【注意】
            这是一个内部方法，外部调用应使用utils.analyze_lgbm_tree_complexity
        """
        return analyze_lgbm_tree_complexity(model)

    @timer
    def grid_search_lgbm_tree_structure(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        base_params: Dict[str, Any],
        max_depth_list: List[int],
        num_leaves_list: List[int],
        min_child_samples_list: List[int],
        min_split_gain_list: List[float],
    ) -> pd.DataFrame:
        """
        【实验1】在固定数据窗口和特征下，系统扫描树结构超参数，
        同时记录树复杂度指标和验证集表现（Spearman / 官方score可二选一）

        【参数】
            X_train, y_train: 已经选好窗口的训练集（比如某个在线窗口的历史数据）
            X_val, y_val: 对应的验证集
            base_params: 其他已固定的 LightGBM 参数（通常是 Optuna 最优参数）
            max_depth_list: 要扫描的 max_depth 值列表
            num_leaves_list: 要扫描的 num_leaves 值列表
            min_child_samples_list: 要扫描的 min_child_samples（类似 min_data_in_leaf）
            min_split_gain_list: 要扫描的 min_split_gain（即 min_gain_to_split）

        【返回】
            pd.DataFrame: 每一行是一组参数 + 树结构统计 + 验证集 Spearman

        【用途】
            诊断"树无法再分裂"问题的专项实验
        """
        results = []

        for md, nl, mcs, msg in itertools.product(
            max_depth_list, num_leaves_list, min_child_samples_list, min_split_gain_list
        ):
            params = dict(base_params)
            params.update(
                {
                    "max_depth": md,
                    "num_leaves": nl,
                    "min_child_samples": mcs,
                    "min_split_gain": msg,
                }
            )
            params.setdefault("objective", "regression")
            params.setdefault("metric", "rmse")
            params.setdefault("random_state", self.seed)
            params.setdefault("verbosity", -1)

            sample_weights = self.calculate_time_weights(len(y_train))

            model = LGBMRegressor(**params)
            model.fit(X_train, y_train, sample_weight=sample_weights)

            preds = model.predict(X_val)
            val_spearman = safe_spearman(y_val, preds)

            # 计算竞赛官方score（调整后夏普比率）
            val_score = calculate_score_metric(preds, y_val.values)

            tree_stats = self._analyze_lgbm_tree_complexity(model)

            results.append(
                {
                    "max_depth": md,
                    "num_leaves": nl,
                    "min_child_samples": mcs,
                    "min_split_gain": msg,
                    "spearman": float(val_spearman),
                    "score": float(val_score),
                    "n_trees": tree_stats["n_trees"],
                    "avg_depth": tree_stats["avg_depth"],
                    "max_depth_observed": tree_stats["max_depth"],
                    "avg_leaves": tree_stats["avg_leaves"],
                    "max_leaves_observed": tree_stats["max_leaves"],
                }
            )

        result_df = pd.DataFrame(results)
        # 按score从高到低排序（竞赛官方指标）
        result_df = result_df.sort_values("score", ascending=False).reset_index(drop=True)
        return result_df

    @timer
    def experiment_lgbm_window_sizes(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        best_params: Dict[str, Any],
        window_sizes: List[int],
        step: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        【实验2】在相同参数下，对比不同在线训练窗口长度下的表现与树复杂度

        说明：
            - 使用当前 ModelTuner 的 true_online_cv_splits 逻辑
            - 每个 window_size 下，跑一轮 Walk-Forward CV：
                * 每个窗口记录 Spearman
                * 每个窗口记录树结构统计（平均深度、叶子数）
            - 最后输出每个 window_size 的：
                * mean_spearman / std_spearman / stability_score
                * avg_depth / avg_leaves / n_windows

        【参数】
            X, y: 全量时间序列数据（已完成因子工程）
            best_params: 已经调好的 LightGBM 参数
            window_sizes: 需要比较的训练窗口长度列表，例如 [60, 90, 120, 180]
            step: 滑动步长（多少天重新训练一次），如果为 None，则使用 self.cv_step

        【返回】
            pd.DataFrame: 每个 window_size 一行的对比表
        """
        from .utils import true_online_cv_splits

        original_train_window_size = self.train_window_size
        original_cv_step = self.cv_step
        original_use_sliding = self.use_sliding_window

        if step is not None:
            self.cv_step = step

        results = []

        try:
            for ws in window_sizes:
                self.train_window_size = ws
                self.use_sliding_window = True

                cv_splits = list(true_online_cv_splits(X, train_size=ws, step=self.cv_step))
                if self.use_reverse_cv:
                    cv_splits = list(reversed(cv_splits))

                window_scores: List[float] = []
                window_depth: List[float] = []
                window_leaves: List[float] = []

                for fold_id, (train_idx, val_idx) in enumerate(cv_splits):
                    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

                    sample_weights = self.calculate_time_weights(len(y_train))

                    params = dict(best_params)
                    params.setdefault("objective", "regression")
                    params.setdefault("metric", "rmse")
                    params.setdefault("random_state", self.seed)
                    params.setdefault("verbosity", -1)

                    model = LGBMRegressor(**params)
                    model.fit(X_train, y_train, sample_weight=sample_weights)

                    preds = model.predict(X_val)
                    score = safe_spearman(y_val, preds)
                    window_scores.append(float(score))

                    tree_stats = self._analyze_lgbm_tree_complexity(model)
                    window_depth.append(tree_stats["avg_depth"])
                    window_leaves.append(tree_stats["avg_leaves"])

                if not window_scores:
                    continue

                mean_spearman = float(np.mean(window_scores))
                std_spearman = float(np.std(window_scores))
                stability_score = mean_spearman - 2 * std_spearman

                results.append(
                    {
                        "train_window_size": ws,
                        "n_windows": len(window_scores),
                        "mean_spearman": mean_spearman,
                        "std_spearman": std_spearman,
                        "stability_score": stability_score,
                        "avg_tree_depth": float(np.mean(window_depth)) if window_depth else 0.0,
                        "avg_tree_leaves": float(np.mean(window_leaves)) if window_leaves else 0.0,
                    }
                )
        finally:
            # 恢复原始超参数设置
            self.train_window_size = original_train_window_size
            self.cv_step = original_cv_step
            self.use_sliding_window = original_use_sliding

        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df = results_df.sort_values("mean_spearman", ascending=False).reset_index(drop=True)
        return results_df

    @timer
    def compare_factor_sets_lgbm(
        self,
        factor_strategies: Dict[str, pd.DataFrame],
        y: pd.Series,
        best_params: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        【实验3】在相同调参结果下，对比多种"因子管理策略"对应的模型表现

        使用方式：
            - factor_strategies: 一个字典，每个 key 是策略名，每个 value 是对应的特征矩阵 X_df
              例如:
                  {
                      "fixed_all_history": X_fixed,
                      "monthly_recent_ic": X_monthly_dynamic,
                      "short_window_weighted": X_weighted_dynamic,
                  }
            - y: 与 X 对齐的目标序列
            - best_params: LightGBM 的一套超参数（通常是 Optuna 最优）

        CV 设置：
            - 复用当前 ModelTuner 的 sliding_window / train_window_size / cv_step 配置
            - 每个策略跑一遍相同的 Walk-Forward CV，然后汇总 mean/std/stability
        """
        from .utils import true_online_cv_splits

        results = []

        # 统一使用同一套CV切分，保证可比较性
        any_X = next(iter(factor_strategies.values()))
        cv_splits = list(
            true_online_cv_splits(any_X, train_size=self.train_window_size, step=self.cv_step)
            if self.use_sliding_window
            else TimeSeriesSplit(n_splits=self.n_splits).split(any_X)
        )
        if self.use_reverse_cv:
            cv_splits = list(reversed(cv_splits))

        for strategy_name, X in factor_strategies.items():
            window_scores: List[float] = []

            for train_idx, val_idx in cv_splits:
                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

                sample_weights = self.calculate_time_weights(len(y_train))

                params = dict(best_params)
                params.setdefault("objective", "regression")
                params.setdefault("metric", "rmse")
                params.setdefault("random_state", self.seed)
                params.setdefault("verbosity", -1)

                model = LGBMRegressor(**params)
                model.fit(X_train, y_train, sample_weight=sample_weights)

                preds = model.predict(X_val)
                score = safe_spearman(y_val, preds)
                window_scores.append(float(score))

            if not window_scores:
                continue

            mean_spearman = float(np.mean(window_scores))
            std_spearman = float(np.std(window_scores))
            stability_score = mean_spearman - 2 * std_spearman

            results.append(
                {
                    "strategy": strategy_name,
                    "n_windows": len(window_scores),
                    "mean_spearman": mean_spearman,
                    "std_spearman": std_spearman,
                    "stability_score": stability_score,
                }
            )

        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df = results_df.sort_values("mean_spearman", ascending=False).reset_index(drop=True)
        return results_df

    def compare_dynamic_factor_selection_lgbm(
        self,
        train_data: pd.DataFrame,
        y: pd.Series,
        best_params: Dict[str, Any],
        top_n: int = 30,
        ic_window_size: int = 60,
    ) -> pd.DataFrame:
        """
        【实验3增强版】对比固定因子池 vs 真·动态因子池（每窗口重选）

        策略对比：
            1. fixed_all_history: 使用全历史IC选因子，整个CV期间固定不变
            2. dynamic_every_window: 每个窗口都用该窗口最近N天重新选因子

        Args:
            train_data: 完整训练数据（包含所有原始因子）
            y: 目标变量
            best_params: LightGBM参数
            top_n: 每次选择的因子数量（默认30）
            ic_window_size: 动态策略计算IC的回看天数（默认60）

        Returns:
            对比结果DataFrame
        """
        from .factor_ic_analyzer import FactorICAnalyzer
        from .feature_preprocessor import FeaturePreprocessor
        from .utils import true_online_cv_splits

        results = []

        # 初始化分析器
        analyzer = FactorICAnalyzer(window_size=20)
        preprocessor = FeaturePreprocessor(analyzer, target_col=y.name, verbose=False)

        # 获取CV切分
        cv_splits = list(
            true_online_cv_splits(train_data, train_size=self.train_window_size, step=self.cv_step)
            if self.use_sliding_window
            else TimeSeriesSplit(n_splits=self.n_splits).split(train_data)
        )
        if self.use_reverse_cv:
            cv_splits = list(reversed(cv_splits))

        # 策略1: 固定因子池（全历史IC）
        print("\n[策略1] 固定因子池 - 使用全历史IC选因子...")
        factor_df_all = analyzer.analyze_dataset(train_data, verbose=False)
        _, _, fixed_features = preprocessor.select_features_by_ic(
            train_data, train_data, factor_df_all, top_n=top_n
        )
        print(f"  选中因子: {fixed_features[:5]}... (共{len(fixed_features)}个)")

        window_scores_fixed = []
        for train_idx, val_idx in cv_splits:
            X_train = train_data.iloc[train_idx][fixed_features]
            X_val = train_data.iloc[val_idx][fixed_features]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            sample_weights = self.calculate_time_weights(len(y_train))

            params = dict(best_params)
            params.setdefault("objective", "regression")
            params.setdefault("metric", "rmse")
            params.setdefault("random_state", self.seed)
            params.setdefault("verbosity", -1)

            model = LGBMRegressor(**params)
            model.fit(X_train, y_train, sample_weight=sample_weights)

            preds = model.predict(X_val)
            score = safe_spearman(y_val, preds)
            window_scores_fixed.append(float(score))

        results.append({
            "strategy": "fixed_all_history",
            "n_windows": len(window_scores_fixed),
            "mean_spearman": float(np.mean(window_scores_fixed)),
            "std_spearman": float(np.std(window_scores_fixed)),
            "stability_score": float(np.mean(window_scores_fixed) - 2 * np.std(window_scores_fixed)),
        })

        # 策略2: 真·动态因子池（每个窗口都重选）
        print(f"\n[策略2] 真·动态因子池 - 每个窗口用最近{ic_window_size}天IC重选因子...")
        window_scores_dynamic = []
        factor_changes = []  # 记录因子变化

        for window_id, (train_idx, val_idx) in enumerate(cv_splits):
            # 每个窗口都重新选因子
            train_window_data = train_data.iloc[train_idx]

            # 用该窗口的最近N天计算IC
            recent_data = train_window_data.tail(ic_window_size)
            factor_df_recent = analyzer.analyze_dataset(recent_data, verbose=False)
            _, _, current_features = preprocessor.select_features_by_ic(
                train_data, train_data, factor_df_recent, top_n=top_n
            )

            # 每100个窗口打印一次
            if window_id % 100 == 0:
                print(f"  窗口{window_id}: 选中因子 -> {current_features[:3]}... (共{len(current_features)}个)")

            # 记录因子变化
            if window_id > 0:
                previous_features = factor_changes[-1]['features']
                n_changed = len(set(current_features) - set(previous_features))
                factor_changes.append({'window': window_id, 'features': current_features, 'n_changed': n_changed})
            else:
                factor_changes.append({'window': window_id, 'features': current_features, 'n_changed': 0})

            # 使用当前因子训练
            X_train = train_data.iloc[train_idx][current_features]
            X_val = train_data.iloc[val_idx][current_features]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            sample_weights = self.calculate_time_weights(len(y_train))

            params = dict(best_params)
            params.setdefault("objective", "regression")
            params.setdefault("metric", "rmse")
            params.setdefault("random_state", self.seed)
            params.setdefault("verbosity", -1)

            model = LGBMRegressor(**params)
            model.fit(X_train, y_train, sample_weight=sample_weights)

            preds = model.predict(X_val)
            score = safe_spearman(y_val, preds)
            window_scores_dynamic.append(float(score))

        # 统计因子变化情况
        avg_change = np.mean([x['n_changed'] for x in factor_changes[1:]])  # 跳过第一个窗口
        print(f"\n  因子变化统计: 平均每个窗口更换 {avg_change:.1f} 个因子（共{top_n}个）")

        results.append({
            "strategy": f"dynamic_every_window(ic_{ic_window_size}d)",
            "n_windows": len(window_scores_dynamic),
            "mean_spearman": float(np.mean(window_scores_dynamic)),
            "std_spearman": float(np.std(window_scores_dynamic)),
            "stability_score": float(np.mean(window_scores_dynamic) - 2 * np.std(window_scores_dynamic)),
        })

        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df = results_df.sort_values("mean_spearman", ascending=False).reset_index(drop=True)
        return results_df

    @timer
    def experiment_dynamic_factors_with_feature_engineering(
        self,
        train_data: pd.DataFrame,
        window_start_idx: int,
        window_size: int = 90,
        val_size: int = 10,
        ic_window_size: int = 60,
        top_n: int = 30,
        base_params: Dict[str, Any] = None,
        max_depth_list: List[int] = None,
        num_leaves_list: List[int] = None,
        min_child_samples_list: List[int] = None,
        min_split_gain_list: List[float] = None,
    ) -> Dict[str, Any]:
        """
        【实验1增强版】动态因子选择 + 特征工程 + 树结构参数网格搜索

        完整pipeline:
            1. 在指定窗口用最近N天IC动态选择Top M个因子
            2. 对这M个因子做特征工程分裂（Lag/Rolling/EWMA等）
            3. 用扩展后的特征做树结构参数网格搜索
            4. 返回最优参数 + 树复杂度统计

        Args:
            train_data: 完整训练数据（包含所有原始因子 + date_id + target）
            window_start_idx: 窗口起始位置（行索引）
            window_size: 训练窗口大小（天数）
            val_size: 验证集大小（天数）
            ic_window_size: IC计算窗口（用最近N天）
            top_n: 选择的因子数量
            base_params: 基础LightGBM参数（非树结构参数）
            max_depth_list: 要扫描的max_depth列表
            num_leaves_list: 要扫描的num_leaves列表
            min_child_samples_list: 要扫描的min_child_samples列表
            min_split_gain_list: 要扫描的min_split_gain列表

        Returns:
            dict: {
                'selected_factors': 选中的因子列表,
                'n_features_after_engineering': 特征工程后特征数,
                'best_tree_params': 最优参数字典,
                'grid_search_results': 完整网格搜索结果DataFrame
            }
        """
        from .factor_ic_analyzer import FactorICAnalyzer
        from .feature_preprocessor import FeaturePreprocessor
        from .feature_engineer import FeatureEngineer

        # 默认参数
        if base_params is None:
            base_params = {
                'n_estimators': 200,
                'learning_rate': 0.05,
                'reg_lambda': 0.01,
                'reg_alpha': 0.01,
                'colsample_bytree': 0.8,
                'subsample': 0.8,
            }

        if max_depth_list is None:
            max_depth_list = [3, 4, 5, 6]
        if num_leaves_list is None:
            num_leaves_list = [8, 16, 31, 64]
        if min_child_samples_list is None:
            min_child_samples_list = [1, 3, 5, 10]
        if min_split_gain_list is None:
            min_split_gain_list = [0.0, 0.001, 0.01, 0.1]

        print("\n" + "="*80)
        print("【实验1】动态因子选择 + 特征工程 + 树结构优化")
        print("="*80)

        # Step 1: 构造train/val窗口
        train_end_idx = window_start_idx + window_size
        val_end_idx = train_end_idx + val_size

        # 边界检查
        max_idx = len(train_data)
        if window_start_idx < 0:
            raise ValueError(f"window_start_idx={window_start_idx} 不能为负数")
        if train_end_idx > max_idx:
            raise ValueError(
                f"训练窗口结束索引 {train_end_idx} 超出数据范围 {max_idx}\n"
                f"  数据长度: {max_idx}\n"
                f"  window_start_idx: {window_start_idx}\n"
                f"  window_size: {window_size}\n"
                f"  建议设置 window_start_idx <= {max_idx - window_size - val_size}"
            )
        if val_end_idx > max_idx:
            raise ValueError(
                f"验证窗口结束索引 {val_end_idx} 超出数据范围 {max_idx}\n"
                f"  数据长度: {max_idx}\n"
                f"  train_end_idx: {train_end_idx}\n"
                f"  val_size: {val_size}\n"
                f"  建议设置 window_start_idx <= {max_idx - window_size - val_size}"
            )

        print(f"\n窗口划分:")
        print(f"  数据总长度: {max_idx}")
        print(f"  训练集: 样本 {window_start_idx}-{train_end_idx} ({window_size}天)")
        print(f"  验证集: 样本 {train_end_idx}-{val_end_idx} ({val_size}天)")

        # Step 2: 动态因子选择
        print(f"\n[Step 1/4] 动态因子选择 - 用最近{ic_window_size}天IC选Top{top_n}因子...")

        analyzer = FactorICAnalyzer(window_size=20)

        # 获取target列名
        target_col = None
        for col in train_data.columns:
            if 'excess' in col or 'target' in col or 'forward_returns' in col:
                target_col = col
                break

        if target_col is None:
            raise ValueError("无法自动识别target列，请确保数据包含target")

        preprocessor = FeaturePreprocessor(analyzer, target_col=target_col, verbose=False)

        # 用训练窗口的最近N天计算IC
        train_window_data = train_data.iloc[window_start_idx:train_end_idx]
        recent_data = train_window_data.tail(ic_window_size)

        factor_df_recent = analyzer.analyze_dataset(recent_data, verbose=False)
        _, _, selected_factors = preprocessor.select_features_by_ic(
            train_data, train_data, factor_df_recent, top_n=top_n
        )

        print(f"  选中因子: {selected_factors[:5]}... (共{len(selected_factors)}个)")

        # Step 3: 提取选中因子的数据
        print(f"\n[Step 2/4] 提取选中因子数据...")

        required_cols = selected_factors.copy()
        if 'date_id' in train_data.columns:
            required_cols.append('date_id')
        required_cols.append(target_col)

        train_factors = train_data.iloc[window_start_idx:train_end_idx][required_cols].copy()
        val_factors = train_data.iloc[train_end_idx:val_end_idx][required_cols].copy()

        print(f"  训练集: {train_factors.shape}")
        print(f"  验证集: {val_factors.shape}")

        # Step 4: 特征工程分裂
        print(f"\n[Step 3/4] 特征工程分裂 - {len(selected_factors)}因子 → 扩展特征...")

        engineer = FeatureEngineer(target_col=target_col, verbose=True)
        train_expanded = engineer.create_features_slim(train_factors)
        val_expanded = engineer.create_features_slim(val_factors)

        print(f"  训练集扩展后: {train_expanded.shape}")
        print(f"  验证集扩展后: {val_expanded.shape}")

        # Step 5: 提取特征列
        exclude_cols = ['date_id', target_col]
        feature_cols = [c for c in train_expanded.columns if c not in exclude_cols]

        X_train = train_expanded[feature_cols]
        y_train = train_expanded[target_col]
        X_val = val_expanded[feature_cols]
        y_val = val_expanded[target_col]

        print(f"\n  特征工程后特征数: {len(feature_cols)}")
        print(f"  X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"  X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

        # 数据验证
        if X_train.shape[0] == 0:
            raise ValueError(f"训练集为空！窗口: {window_start_idx}-{train_end_idx}")
        if X_val.shape[0] == 0:
            raise ValueError(f"验证集为空！窗口: {train_end_idx}-{val_end_idx}")
        if len(feature_cols) == 0:
            raise ValueError(f"特征工程后没有特征列！原始因子数: {len(selected_factors)}")

        # Step 6: 树结构参数网格搜索
        print(f"\n[Step 4/4] 树结构参数网格搜索...")
        print(f"  参数空间: {len(max_depth_list)}×{len(num_leaves_list)}×{len(min_child_samples_list)}×{len(min_split_gain_list)} = {len(max_depth_list)*len(num_leaves_list)*len(min_child_samples_list)*len(min_split_gain_list)}种组合")

        tree_results = self.grid_search_lgbm_tree_structure(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            base_params=base_params,
            max_depth_list=max_depth_list,
            num_leaves_list=num_leaves_list,
            min_child_samples_list=min_child_samples_list,
            min_split_gain_list=min_split_gain_list
        )

        # Step 7: 返回结果
        best_row = tree_results.iloc[0]

        return {
            'selected_factors': selected_factors,
            'n_features_after_engineering': len(feature_cols),
            'best_tree_params': best_row.to_dict(),
            'grid_search_results': tree_results,
            'train_window': (window_start_idx, train_end_idx),
            'val_window': (train_end_idx, val_end_idx)
        }

    @timer
    def experiment_lgbm_vs_mlp_ensemble(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        lgbm_params: Dict[str, Any],
        catboost_params: Optional[Dict[str, Any]] = None,
        mlp_hidden_layer_sizes: tuple = (64, 32),
        mlp_alpha: float = 1e-4,
        mlp_learning_rate_init: float = 1e-3,
    ) -> pd.DataFrame:
        """
        【实验4】在给定 train/val 切分下，对比 LGBM / CatBoost / MLP / 简单集成 / 线性 stacking 的表现

        说明：
            - 阶段1：训练 LGBM（树模型主力）、CatBoost（补充树模型）和简单 MLP（Tabular）
            - 阶段2：对 val 集做简单平均集成（三模型）
            - 阶段3：在 val 集上用 Ridge 做一个「线性 stacking」（实验版）

        注意：
            - 为了严格防止信息泄露，真实 stacking 应该在多折时间序列CV上构造 OOF 预测再训练 meta 模型。
              这里提供的是一个简单原型，帮助你快速验证「NN + Tree 是否值得引入」。

        【参数】
            X_train, y_train: 训练集
            X_val, y_val: 验证集
            lgbm_params: LightGBM参数
            catboost_params: CatBoost参数（可选，如果为None则使用默认参数）
            mlp_hidden_layer_sizes: MLP隐藏层结构
            mlp_alpha: MLP L2正则化系数
            mlp_learning_rate_init: MLP初始学习率

        【返回】
            pd.DataFrame: 各模型的性能对比表
        """
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import Ridge

        # 1) 训练 LGBM 基准模型
        params = dict(lgbm_params)
        params.setdefault("objective", "regression")
        params.setdefault("metric", "rmse")
        params.setdefault("random_state", self.seed)
        params.setdefault("verbosity", -1)

        sample_weights = self.calculate_time_weights(len(y_train))

        lgbm_model = LGBMRegressor(**params)
        lgbm_model.fit(X_train, y_train, sample_weight=sample_weights)

        lgbm_pred_val = lgbm_model.predict(X_val)
        lgbm_spearman = safe_spearman(y_val, lgbm_pred_val)

        # 2) 训练 CatBoost 补充模型
        if catboost_params is None:
            catboost_params = {
                "iterations": 200,
                "learning_rate": 0.05,
                "depth": 4,
                "l2_leaf_reg": 3.0,
                "random_seed": self.seed,
                "verbose": False,
            }

        cat_params = dict(catboost_params)
        cat_params.setdefault("random_seed", self.seed)
        cat_params.setdefault("verbose", False)

        cat_model = CatBoostRegressor(**cat_params)
        cat_model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)

        cat_pred_val = cat_model.predict(X_val)
        cat_spearman = safe_spearman(y_val, cat_pred_val)

        # 3) 训练 Tabular MLP（带标准化 + early_stopping）
        mlp = MLPRegressor(
            hidden_layer_sizes=mlp_hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=mlp_alpha,
            batch_size="auto",
            learning_rate="adaptive",
            learning_rate_init=mlp_learning_rate_init,
            max_iter=200,
            early_stopping=True,
            n_iter_no_change=10,
            validation_fraction=0.1,
            random_state=self.seed,
            verbose=False,
        )
        mlp_model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("mlp", mlp),
            ]
        )
        mlp_model.fit(X_train, y_train)

        mlp_pred_val = mlp_model.predict(X_val)
        mlp_spearman = safe_spearman(y_val, mlp_pred_val)

        # 4) 简单平均集成（阶段2）- 三模型均权
        ensemble_mean_pred = (lgbm_pred_val + cat_pred_val + mlp_pred_val) / 3.0
        ensemble_mean_spearman = safe_spearman(y_val, ensemble_mean_pred)

        # 5) 线性 stacking（阶段3 · 原型）
        stack_X_val = np.vstack([lgbm_pred_val, cat_pred_val, mlp_pred_val]).T  # shape=(n_samples, 3)
        stack_model = Ridge(alpha=1e-3)
        stack_model.fit(stack_X_val, y_val)
        stack_pred_val = stack_model.predict(stack_X_val)
        stack_spearman = safe_spearman(y_val, stack_pred_val)

        result = pd.DataFrame(
            [
                {
                    "model": "lgbm",
                    "spearman": float(lgbm_spearman),
                },
                {
                    "model": "catboost",
                    "spearman": float(cat_spearman),
                },
                {
                    "model": "mlp",
                    "spearman": float(mlp_spearman),
                },
                {
                    "model": "mean_ensemble(lgbm+cat+mlp)",
                    "spearman": float(ensemble_mean_spearman),
                },
                {
                    "model": "stacking_ridge(lgbm+cat+mlp)",
                    "spearman": float(stack_spearman),
                },
            ]
        ).sort_values("spearman", ascending=False)
        return result.reset_index(drop=True)

    @timer
    def tune_lgbm_dynamic_full(
        self,
        train_data: pd.DataFrame,
        target_col: str,
        n_trials: int = 30,
        top_n: int = 30,
        ic_window_size: int = 60,
        n_jobs: int = 1,
        fixed_params: Optional[Dict[str, Any]] = None,
    ) -> optuna.study.Study:
        """
        【完整版动态因子调参】每个窗口动态选因子+特征工程+LightGBM训练

        流程：
            对每个Optuna trial:
                对每个滑动窗口:
                    1. 用最近ic_window_size天IC动态选top_n个因子
                    2. 对这top_n个因子做特征工程分裂（30 → 1250特征）
                    3. 用这些特征训练LightGBM
                    4. 验证集评分
                返回所有窗口的平均分数

        Args:
            train_data: 完整训练数据（包含所有原始因子 + target列）
            target_col: 目标列名
            n_trials: Optuna试验次数
            top_n: 每次选择的因子数量
            ic_window_size: IC计算窗口（用最近N天）
            n_jobs: Optuna并行数
            fixed_params: 固定参数（例如从实验1得到的最优树参数）

        Returns:
            optuna.study.Study: 包含最优参数和试验历史

        ⚠️ 警告：
            - 这个方法会非常慢（每个窗口都要做IC分析+特征工程）
            - 建议先用少量trials测试（n_trials=5）
            - 预计耗时：802窗口 × 30trials ≈ 2-5小时
        """
        from .factor_ic_analyzer import FactorICAnalyzer
        from .feature_preprocessor import FeaturePreprocessor
        from .feature_engineer import FeatureEngineer

        print("\n" + "="*80)
        print("【完整版动态因子调参】LightGBM")
        print("="*80)
        print(f"  训练数据: {train_data.shape}")
        print(f"  窗口大小: {self.train_window_size}天")
        print(f"  滑动步长: {self.cv_step}天")
        print(f"  IC窗口: {ic_window_size}天")
        print(f"  选择因子数: {top_n}个")
        print(f"  Optuna trials: {n_trials}")

        # 初始化分析器
        analyzer = FactorICAnalyzer(window_size=20)
        preprocessor = FeaturePreprocessor(analyzer, target_col=target_col, verbose=False)
        engineer = FeatureEngineer(target_col=target_col, verbose=False)

        # 获取CV切分
        from .utils import true_online_cv_splits
        cv_splits = list(
            true_online_cv_splits(
                train_data,
                train_size=self.train_window_size,
                step=self.cv_step
            ) if self.use_sliding_window
            else TimeSeriesSplit(n_splits=self.n_splits).split(train_data)
        )

        if self.use_reverse_cv:
            cv_splits = list(reversed(cv_splits))

        n_windows = len(cv_splits)
        print(f"  预计窗口数: {n_windows}")
        print(f"  预计总特征工程次数: {n_windows * n_trials * 2} (训练+验证)")
        print(f"\n⚠️  这将会很慢，建议先用n_trials=5测试")

        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def objective(trial):
            # 参数空间
            params = {
                "objective": "regression",
                "metric": "rmse",
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 6),
                "num_leaves": trial.suggest_int("num_leaves", 8, 64),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0001, 0.5, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0001, 0.5, log=True),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "subsample_freq": 1,
                "min_child_samples": trial.suggest_int("min_child_samples", 1, 5),
                "min_child_weight": trial.suggest_float("min_child_weight", 0.001, 0.1, log=True),
                "random_state": self.seed,
                "verbosity": -1
            }

            # 应用固定参数
            if fixed_params:
                params.update(fixed_params)

            window_scores = []
            fold_metrics = {
                'ic': [],
                'ric': [],
                'avg_depth': [],
                'avg_leaves': [],
            }

            # 遍历每个窗口
            for window_id, (train_idx, val_idx) in enumerate(cv_splits):
                # ====== Step 1: 动态选因子 ======
                train_window_raw = train_data.iloc[train_idx]

                # 用最近N天计算IC
                recent_data = train_window_raw.tail(ic_window_size)
                factor_df_recent = analyzer.analyze_dataset(recent_data, target_col=target_col, verbose=False)
                _, _, selected_factors = preprocessor.select_features_by_ic(
                    train_data, train_data, factor_df_recent, top_n=top_n
                )

                # ====== Step 2: 提取选中因子数据 ======
                required_cols = selected_factors.copy()
                if 'date_id' in train_data.columns:
                    required_cols.append('date_id')
                required_cols.append(target_col)

                train_factors = train_data.iloc[train_idx][required_cols].copy()
                val_factors = train_data.iloc[val_idx][required_cols].copy()

                # ====== Step 3: 特征工程分裂 ======
                train_expanded = engineer.create_features_slim(train_factors)
                val_expanded = engineer.create_features_slim(val_factors)

                # 提取特征列
                exclude_cols = ['date_id', target_col]
                feature_cols = [c for c in train_expanded.columns if c not in exclude_cols]

                X_train = train_expanded[feature_cols]
                y_train = train_expanded[target_col]
                X_val = val_expanded[feature_cols]
                y_val = val_expanded[target_col]

                # ====== Step 4: 训练LightGBM ======
                sample_weights = self.calculate_time_weights(len(y_train))

                model = LGBMRegressor(**params)
                model.fit(X_train, y_train, sample_weight=sample_weights)

                # 树复杂度诊断
                tree_stats = analyze_lgbm_tree_complexity(model)
                fold_metrics['avg_depth'].append(tree_stats['avg_depth'])
                fold_metrics['avg_leaves'].append(tree_stats['avg_leaves'])

                # 预测和评分
                preds = model.predict(X_val)
                spearman_score = calculate_score_metric(preds, y_val.values)
                window_scores.append(spearman_score)

                # 辅助指标
                from scipy.stats import pearsonr
                ic, _ = pearsonr(y_val.values, preds)
                fold_metrics['ic'].append(ic if not np.isnan(ic) else 0.0)
                fold_metrics['ric'].append(spearman_score)

                # 每100个窗口打印进度
                if window_id % 100 == 0:
                    print(f"  Trial {trial.number} - Window {window_id}/{n_windows}: score={spearman_score:.4f}, factors={len(selected_factors)}")

            # 计算最终得分（带复杂度惩罚）
            mean_spearman = float(np.mean(window_scores))
            std_spearman = float(np.std(window_scores))

            if fold_metrics['avg_depth']:
                avg_tree_depth = float(np.mean(fold_metrics['avg_depth']))
                avg_tree_leaves = float(np.mean(fold_metrics['avg_leaves']))
            else:
                avg_tree_depth = 0.0
                avg_tree_leaves = 0.0

            # 复杂度比例
            depth_ratio = avg_tree_depth / max(params["max_depth"], 1)
            leaf_ratio = avg_tree_leaves / max(params["num_leaves"], 1)
            complexity_ratio = 0.5 * (depth_ratio + leaf_ratio)
            complexity_ratio = max(0.0, min(1.0, complexity_ratio))

            # 最终得分
            final_score = mean_spearman * (0.5 + 0.5 * complexity_ratio)

            # 保存指标
            trial.set_user_attr('mean_spearman', mean_spearman)
            trial.set_user_attr('std_spearman', std_spearman)
            trial.set_user_attr('avg_tree_depth', avg_tree_depth)
            trial.set_user_attr('avg_tree_leaves', avg_tree_leaves)
            trial.set_user_attr('complexity_ratio', complexity_ratio)

            return final_score

        # 运行Optuna优化
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

        print(f"\n✅ 优化完成！")
        print(f"  最优得分: {study.best_value:.6f}")
        print(f"  最优参数: {study.best_params}")

        return study

    @timer
    def tune_catboost_dynamic_full(
        self,
        train_data: pd.DataFrame,
        target_col: str,
        n_trials: int = 20,
        top_n: int = 30,
        ic_window_size: int = 60,
        use_pool: bool = False,
        task_type: str = "CPU",
        fixed_params: Optional[Dict[str, Any]] = None,
    ) -> optuna.study.Study:
        """
        【完整版动态因子调参】每个窗口动态选因子+特征工程+CatBoost训练

        功能同 tune_lgbm_dynamic_full，但使用CatBoost模型

        Args:
            train_data: 完整训练数据（包含所有原始因子 + target列）
            target_col: 目标列名
            n_trials: Optuna试验次数
            top_n: 每次选择的因子数量
            ic_window_size: IC计算窗口（用最近N天）
            use_pool: 是否使用CatBoost Pool
            task_type: "CPU" 或 "GPU"
            fixed_params: 固定参数

        Returns:
            optuna.study.Study: 包含最优参数和试验历史
        """
        from .factor_ic_analyzer import FactorICAnalyzer
        from .feature_preprocessor import FeaturePreprocessor
        from .feature_engineer import FeatureEngineer

        print("\n" + "="*80)
        print("【完整版动态因子调参】CatBoost")
        print("="*80)
        print(f"  训练数据: {train_data.shape}")
        print(f"  窗口大小: {self.train_window_size}天")
        print(f"  IC窗口: {ic_window_size}天")
        print(f"  选择因子数: {top_n}个")
        print(f"  Optuna trials: {n_trials}")

        # 初始化
        analyzer = FactorICAnalyzer(window_size=20)
        preprocessor = FeaturePreprocessor(analyzer, target_col=target_col, verbose=False)
        engineer = FeatureEngineer(target_col=target_col, verbose=False)

        # CV切分
        from .utils import true_online_cv_splits
        cv_splits = list(
            true_online_cv_splits(
                train_data,
                train_size=self.train_window_size,
                step=self.cv_step
            ) if self.use_sliding_window
            else TimeSeriesSplit(n_splits=self.n_splits).split(train_data)
        )

        if self.use_reverse_cv:
            cv_splits = list(reversed(cv_splits))

        n_windows = len(cv_splits)
        print(f"  预计窗口数: {n_windows}")

        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def objective(trial):
            params = {
                "loss_function": "RMSE",
                "iterations": trial.suggest_int("iterations", 50, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "depth": trial.suggest_int("depth", 3, 8),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.01, 5.0, log=True),
                "random_strength": trial.suggest_float("random_strength", 0.01, 2.0, log=True),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
                "random_seed": self.seed,
                "verbose": False,
                "bootstrap_type": "MVS",
            }

            if task_type.upper() == "GPU":
                params["task_type"] = "GPU"
                params["devices"] = "0"

            if fixed_params:
                params.update(fixed_params)

            window_scores = []

            for window_id, (train_idx, val_idx) in enumerate(cv_splits):
                # 动态选因子
                train_window_raw = train_data.iloc[train_idx]
                recent_data = train_window_raw.tail(ic_window_size)
                factor_df_recent = analyzer.analyze_dataset(recent_data, target_col=target_col, verbose=False)
                _, _, selected_factors = preprocessor.select_features_by_ic(
                    train_data, train_data, factor_df_recent, top_n=top_n
                )

                # 提取数据
                required_cols = selected_factors.copy()
                if 'date_id' in train_data.columns:
                    required_cols.append('date_id')
                required_cols.append(target_col)

                train_factors = train_data.iloc[train_idx][required_cols].copy()
                val_factors = train_data.iloc[val_idx][required_cols].copy()

                # 特征工程
                train_expanded = engineer.create_features_slim(train_factors)
                val_expanded = engineer.create_features_slim(val_factors)

                exclude_cols = ['date_id', target_col]
                feature_cols = [c for c in train_expanded.columns if c not in exclude_cols]

                X_train = train_expanded[feature_cols]
                y_train = train_expanded[target_col]
                X_val = val_expanded[feature_cols]
                y_val = val_expanded[target_col]

                # 训练CatBoost
                sample_weights = self.calculate_time_weights(len(y_train))

                if use_pool:
                    pool_train = Pool(X_train, label=y_train, weight=sample_weights)
                    model = CatBoostRegressor(**params)
                    model.fit(pool_train, verbose=False)
                else:
                    model = CatBoostRegressor(**params)
                    model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)

                # 评分
                preds = model.predict(X_val)
                spearman_score = calculate_score_metric(preds, y_val.values)
                window_scores.append(spearman_score)

                if window_id % 100 == 0:
                    print(f"  Trial {trial.number} - Window {window_id}/{n_windows}: score={spearman_score:.4f}")

            mean_spearman = float(np.mean(window_scores))
            trial.set_user_attr('mean_spearman', mean_spearman)
            trial.set_user_attr('std_spearman', float(np.std(window_scores)))

            return mean_spearman

        study.optimize(objective, n_trials=n_trials, n_jobs=1)

        print(f"\n✅ 优化完成！")
        print(f"  最优得分: {study.best_value:.6f}")
        print(f"  最优参数: {study.best_params}")

        return study

    @timer
    def tune_mlp_tabular(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 20,
        n_jobs: int = 1,
        fixed_params: Optional[Dict[str, Any]] = None,
    ) -> optuna.study.Study:
        """
        【MLP超参数搜索】用Optuna + 滑动窗口CV搜索MLPRegressor最优超参数

        Args:
            X: 特征矩阵（已经做过特征工程的1250维特征）
            y: 目标变量
            n_trials: Optuna试验次数（建议10-30）
            n_jobs: 并行数（1=串行）
            fixed_params: 固定参数（可选）

        Returns:
            optuna.study.Study: 包含最优参数和试验历史

        搜索空间：
            - hidden_layer_sizes: [(64,), (64,32), (128,64)]
            - learning_rate_init: [1e-4, 3e-4, 1e-3, 3e-3]
            - alpha: [1e-5, 1e-4, 1e-3, 1e-2]

        固定参数：
            - activation="relu", solver="adam"
            - learning_rate="adaptive"（验证集不涨分时自动降LR）
            - early_stopping=True + validation_fraction=0.1（防止过拟合）

        Notes:
            - MLP是纯前馈网络，输入是展平的特征向量
            - 你已经用lag/roll把时间信息编码进特征
            - MLP学习"这些特征组合 → 未来收益"的映射
            - 比LSTM简单、快速、不易过拟合
        """
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        print(f"\n{'='*60}")
        print(f"  🔥 开始 MLP Tabular 超参数搜索")
        print(f"{'='*60}")
        print(f"  数据维度: {X.shape}")
        print(f"  试验次数: {n_trials}")
        print(f"  CV策略: 滑动窗口 (窗口={self.train_window_size}天, 步长={self.cv_step}天)")
        print(f"  时间衰减: {self.time_decay}")

        # 1. 准备滑动窗口CV
        splits = sliding_window_cv_splits(
            data_length=len(X),
            train_window_size=self.train_window_size,
            test_window_size=self.cv_step,
            step=self.cv_step
        )
        n_windows = len(splits)
        print(f"  滑动窗口数: {n_windows}")

        # 2. 定义Optuna目标函数
        def objective(trial: optuna.trial.Trial) -> float:
            # 2.1 定义搜索空间
            hidden_layer_sizes_choices = [
                (64,),
                (64, 32),
                (128, 64),
            ]
            hidden_layer_sizes = trial.suggest_categorical(
                "hidden_layer_sizes",
                [str(h) for h in hidden_layer_sizes_choices]  # Optuna需要字符串
            )
            # 转回tuple
            hidden_layer_sizes = eval(hidden_layer_sizes)

            learning_rate_init = trial.suggest_categorical(
                "learning_rate_init",
                [1e-4, 3e-4, 1e-3, 3e-3]
            )

            alpha = trial.suggest_categorical(
                "alpha",
                [1e-5, 1e-4, 1e-3, 1e-2]
            )

            # 2.2 创建MLP Pipeline（StandardScaler + MLP）
            mlp = MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                activation="relu",
                solver="adam",
                alpha=alpha,
                batch_size="auto",
                learning_rate="adaptive",  # 验证集不涨时自动降LR
                learning_rate_init=learning_rate_init,
                max_iter=200,
                early_stopping=True,  # 防止过拟合
                n_iter_no_change=10,
                validation_fraction=0.1,
                random_state=self.seed,
                verbose=False,
            )

            model = Pipeline([
                ("scaler", StandardScaler()),
                ("mlp", mlp),
            ])

            # 应用fixed_params（如果有）
            if fixed_params:
                for k, v in fixed_params.items():
                    if hasattr(model.named_steps['mlp'], k):
                        setattr(model.named_steps['mlp'], k, v)

            # 2.3 滑动窗口CV
            window_scores = []
            for i, (train_idx, val_idx) in enumerate(splits):
                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]
                X_val = X.iloc[val_idx]
                y_val = y.iloc[val_idx]

                # 计算时间权重（新样本权重高）
                sample_weights = self.calculate_time_weights(len(train_idx))

                # 训练（MLP不支持sample_weight，所以忽略）
                try:
                    model.fit(X_train, y_train)

                    # 预测验证集
                    y_pred = model.predict(X_val)

                    # 计算Spearman相关系数
                    corr, _ = safe_spearman(y_pred, y_val)
                    window_scores.append(corr if not np.isnan(corr) else 0.0)

                except Exception as e:
                    # MLP训练失败（可能收敛问题）
                    window_scores.append(0.0)

                # 每100个窗口打印进度
                if (i + 1) % 100 == 0:
                    print(f"    Trial {trial.number}: 完成 {i+1}/{n_windows} 窗口, 当前均分={np.mean(window_scores):.6f}")

            # 2.4 计算加权平均分数（新窗口权重高）
            window_weights = self.calculate_time_weights(len(window_scores))
            weighted_score = np.average(window_scores, weights=window_weights)

            return weighted_score

        # 3. 创建Optuna Study并优化
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        )

        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)

        # 4. 输出最优结果
        print(f"\n{'='*60}")
        print(f"  ✅ MLP 超参数搜索完成！")
        print(f"{'='*60}")
        print(f"  最优得分: {study.best_value:.6f}")
        print(f"  最优参数: {study.best_params}")

        return study

    @timer
    def tune_mlp_dynamic_full(
        self,
        train_data: pd.DataFrame,
        target_col: str,
        n_trials: int = 20,
        top_n: int = 30,
        ic_window_size: int = 60,
        n_jobs: int = 1,
        fixed_params: Optional[Dict[str, Any]] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ) -> optuna.study.Study:
        """
        【完整版MLP动态因子调参】每个窗口动态选因子+特征工程+PyTorch MLP训练（支持GPU）

        Args:
            train_data: 训练数据（包含原始因子+target，未分裂）
            target_col: 目标列名
            n_trials: Optuna试验次数
            top_n: 每个窗口选择的因子数
            ic_window_size: IC计算窗口大小（天数）
            n_jobs: 并行数（PyTorch+GPU时建议=1）
            fixed_params: 固定参数
            device: 'cuda' 或 'cpu'

        Returns:
            optuna.study.Study

        流程：
            对每个Optuna trial:
                对每个滑动窗口:
                    1. 用最近ic_window_size天IC动态选top_n个因子
                    2. 对这top_n个因子做特征工程分裂（30 → 1250特征）
                    3. 用这些特征训练PyTorch MLP（GPU加速）
                    4. 验证集评分
                返回所有窗口的平均分数

        Notes:
            - 和树模型完全一样的动态因子逻辑
            - 使用PyTorch MLP，支持GPU加速
        """
        import torch
        from .nn_models import train_pytorch_mlp, predict_pytorch_mlp

        print(f"\n{'='*60}")
        print(f"  🔥 开始 PyTorch MLP 动态因子超参数搜索（GPU版）")
        print(f"{'='*60}")
        print(f"  数据维度: {train_data.shape}")
        print(f"  试验次数: {n_trials}")
        print(f"  每窗口选因子数: {top_n}")
        print(f"  IC窗口: {ic_window_size}天")
        print(f"  CV策略: 滑动窗口 (窗口={self.train_window_size}天, 步长={self.cv_step}天)")
        print(f"  设备: {device}")
        if device == 'cuda':
            print(f"  GPU可用: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"  GPU名称: {torch.cuda.get_device_name(0)}")

        # 初始化工具
        from .factor_ic_analyzer import FactorICAnalyzer
        from .feature_preprocessor import FeaturePreprocessor
        from .feature_engineer import FeatureEngineer

        analyzer = FactorICAnalyzer(window_size=ic_window_size)
        preprocessor = FeaturePreprocessor(analyzer, target_col=target_col)
        engineer = FeatureEngineer(target_col=target_col, verbose=False)

        # 准备滑动窗口
        splits = sliding_window_cv_splits(
            data_length=len(train_data),
            train_window_size=self.train_window_size,
            test_window_size=self.cv_step,
            step=self.cv_step
        )
        n_windows = len(splits)
        print(f"  滑动窗口数: {n_windows}")

        # 获取所有因子列
        meta_cols = ['date_id', target_col, 'forward_returns', 'risk_free_rate',
                     'lagged_forward_returns', 'lagged_risk_free_rate',
                     'lagged_market_forward_excess_returns', 'is_scored']
        all_factor_cols = [c for c in train_data.columns if c not in meta_cols]

        print(f"  原始因子数: {len(all_factor_cols)}")
        print(f"\n⚠️  注意: 每个窗口都会重新选因子+分裂，耗时较长")

        # 定义Optuna目标函数
        def objective(trial: optuna.trial.Trial) -> float:
            # 定义搜索空间
            hidden_dims_choices = [
                [64],
                [64, 32],
                [128, 64],
            ]
            hidden_dims = trial.suggest_categorical(
                "hidden_dims",
                [str(h) for h in hidden_dims_choices]
            )
            hidden_dims = eval(hidden_dims)

            lr = trial.suggest_categorical(
                "lr",
                [1e-4, 3e-4, 1e-3, 3e-3]
            )

            weight_decay = trial.suggest_categorical(
                "weight_decay",
                [1e-5, 1e-4, 1e-3, 1e-2]
            )

            dropout = trial.suggest_categorical(
                "dropout",
                [0.0, 0.1, 0.2, 0.3]
            )

            # 滑动窗口CV
            window_scores = []

            for i, (train_idx, val_idx) in enumerate(splits):
                # 提取训练和验证窗口
                train_window = train_data.iloc[train_idx].copy()
                val_window = train_data.iloc[val_idx].copy()

                # Step 1: 动态选因子（用训练窗口最近ic_window_size天）
                ic_data = train_window.tail(min(ic_window_size, len(train_window)))
                factor_df = analyzer.analyze_dataset(ic_data, verbose=False)

                # 按Fitness排序选Top N
                top_factors = factor_df.nlargest(top_n, 'Fitness')['特征'].tolist()

                # Step 2: 提取选中因子数据
                selected_cols = top_factors + [target_col]
                train_selected = train_window[[c for c in selected_cols if c in train_window.columns]].copy()
                val_selected = val_window[[c for c in selected_cols if c in val_window.columns]].copy()

                # Step 3: 特征工程分裂
                try:
                    train_expanded = engineer.create_features_slim(train_selected)
                    val_expanded = engineer.create_features_slim(val_selected)
                except Exception as e:
                    # 特征工程失败（可能窗口太小）
                    window_scores.append(0.0)
                    continue

                # 提取特征
                feature_cols = [c for c in train_expanded.columns if c not in ['date_id', target_col]]
                X_train = train_expanded[feature_cols].values
                y_train = train_expanded[target_col].values
                X_val = val_expanded[feature_cols].values
                y_val = val_expanded[target_col].values

                # 检查有效性
                if len(X_train) < 10 or len(X_val) < 1:
                    window_scores.append(0.0)
                    continue

                # Step 4: 训练PyTorch MLP
                try:
                    model, scaler = train_pytorch_mlp(
                        X_train=X_train,
                        y_train=y_train,
                        X_val=X_val,
                        y_val=y_val,
                        hidden_dims=hidden_dims,
                        dropout=dropout,
                        lr=lr,
                        weight_decay=weight_decay,
                        batch_size=32,
                        epochs=200,
                        early_stopping_patience=10,
                        device=device,
                        verbose=False
                    )

                    # 预测
                    y_pred = predict_pytorch_mlp(model, scaler, X_val, device=device)
                    corr, _ = safe_spearman(y_pred, y_val)
                    window_scores.append(corr if not np.isnan(corr) else 0.0)
                except Exception as e:
                    window_scores.append(0.0)

                # 打印进度
                if (i + 1) % 100 == 0:
                    print(f"    Trial {trial.number}: 完成 {i+1}/{n_windows} 窗口, 当前均分={np.mean(window_scores):.6f}")

            # 计算加权平均分数
            if len(window_scores) == 0:
                return 0.0

            window_weights = self.calculate_time_weights(len(window_scores))
            weighted_score = np.average(window_scores, weights=window_weights)

            return weighted_score

        # 创建Optuna Study并优化
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        )

        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)

        # 输出结果
        print(f"\n{'='*60}")
        print(f"  ✅ PyTorch MLP 动态因子超参数搜索完成！")
        print(f"{'='*60}")
        print(f"  最优得分: {study.best_value:.6f}")
        print(f"  最优参数: {study.best_params}")

        return study
