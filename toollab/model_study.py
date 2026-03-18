"""
模型研究分析模块
提供统一的模型评估接口，支持LightGBM、XGBoost、CatBoost等树模型
以及基于Permutation Importance的深度学习模型分析

核心功能:
1. ConvergenceAnalyzer - 训练收敛分析
2. FeatureImportanceAnalyzer - 特征重要性分析
3. TemporalStabilityAnalyzer - 时序稳定性分析
4. run_full_analysis - 一键完成所有分析

"""

import os
import gc
import json
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# 尝试导入LightGBM（可选）
try:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    warnings.warn("LightGBM not available. Convergence analysis will be limited.")

# 尝试导入PyTorch（可选）
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.preprocessing import StandardScaler
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. LSTM analysis will be skipped.")

# 导入本地工具
from .metrics import calculate_score_metric


# ============================================================
# LSTM模型定义（PyTorch）
# ============================================================

if PYTORCH_AVAILABLE:
    class LSTMPredictor(nn.Module):
        """
        CNN → FC → LSTM → Attention → FC 架构
        用于时间序列预测，支持序列长度为60的输入
        """

        def __init__(self, input_dim, hidden_dim=32, num_layers=1, dropout=0):
            super(LSTMPredictor, self).__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers

            # CNN: 输入维度 → 16通道
            self.cnn = nn.Sequential(
                nn.Conv1d(in_channels=input_dim, out_channels=16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(16)
            )

            # FC: CNN输出(16) → LSTM输入(hidden_dim)
            self.feature_fc = nn.Sequential(
                nn.Linear(16, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

            # LSTM
            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=0 if num_layers == 1 else dropout
            )

            self.norm = nn.LayerNorm(hidden_dim)

            # Attention
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.Tanh(),
                nn.Linear(hidden_dim // 4, 1)
            )

            # Output FC
            self.fc_out = nn.Sequential(
                nn.Linear(hidden_dim, 16),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(8, 1)
            )

            self._init_weights()

        def forward(self, x):
            # CNN处理
            x_cnn = x.transpose(1, 2)  # (batch, seq, features) → (batch, features, seq)
            cnn_out = self.cnn(x_cnn)

            # 转回序列格式
            cnn_out = cnn_out.transpose(1, 2)  # (batch, 16, seq) → (batch, seq, 16)
            cnn_out = self.feature_fc(cnn_out)

            # LSTM处理
            lstm_out, _ = self.lstm(cnn_out)
            lstm_out = self.norm(lstm_out)

            # Attention加权
            attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
            context = torch.sum(attention_weights * lstm_out, dim=1)

            # 最终输出
            output = self.fc_out(context)
            return output

        def _init_weights(self):
            # LSTM权重初始化
            for name, param in self.lstm.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    nn.init.constant_(param.data, 0.0)
                    n = param.size(0)
                    param.data[n // 4:n // 2] = 1.0

            # 其他层权重初始化
            for module in list(self.cnn) + list(self.feature_fc) + list(self.attention) + list(self.fc_out):
                if isinstance(module, (nn.Linear, nn.Conv1d)):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)


# ============================================================
# 配置类
# ============================================================

class ModelStudyConfig:
    """模型研究配置类"""

    DEFAULT_CONFIG = {
        'test_mode': True,
        'top_n_features': 50,
        'n_temporal_windows': 20,
        'save_plots': True,
        'plot_dir': './model_study_plots/',
        'figsize': (14, 8),
        'dpi': 100,
        'verbose': True,
    }

    def __init__(self, **kwargs):
        self.config = self.DEFAULT_CONFIG.copy()
        self.config.update(kwargs)

        # 创建输出目录
        if self.config['save_plots']:
            os.makedirs(self.config['plot_dir'], exist_ok=True)

    def __getitem__(self, key):
        return self.config[key]

    def get(self, key, default=None):
        return self.config.get(key, default)

    def copy(self):
        """返回配置字典的副本"""
        return self.config.copy()


# ============================================================
# 1. 训练收敛分析器
# ============================================================

class ConvergenceAnalyzer:
    """
    训练收敛分析器
    支持模型: LightGBM, XGBoost, CatBoost

    分析内容:
    - RMSE收敛曲线（训练集 vs 验证集）
    - Score Metric收敛曲线
    - 最佳迭代数诊断
    - 过拟合程度分析
    - 收敛速度评估
    """

    def __init__(self, model_type: str = 'lightgbm'):
        """
        Args:
            model_type: 模型类型 ('lightgbm', 'xgboost', 'catboost', 'lstm')
        """
        self.model_type = model_type.lower()
        self.supported_types = ['lightgbm', 'xgboost', 'catboost', 'lstm']

        if self.model_type not in self.supported_types:
            raise ValueError(f"Unsupported model_type: {model_type}. Must be one of {self.supported_types}")

    def analyze(
        self,
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        config: Union[Dict, ModelStudyConfig],
        hyperparams: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        执行收敛分析

        Args:
            model: 已训练的模型对象
            X_train: 训练集特征
            y_train: 训练集标签
            X_valid: 验证集特征
            y_valid: 验证集标签
            config: 配置字典或ModelStudyConfig对象
            hyperparams: 超参数字典（用于保存到文件）

        Returns:
            results: 分析结果字典
                - model: 训练好的模型
                - best_iteration: 最佳迭代数
                - best_score_iteration: Score Metric最佳迭代数
                - best_valid_score: 最佳验证分数
                - overfitting_gap: 过拟合程度 (%)
                - convergence_speed: 收敛速度 (%)
                - train_rmses: 训练RMSE历史
                - valid_rmses: 验证RMSE历史
                - train_scores: 训练Score历史
                - valid_scores: 验证Score历史
        """
        if isinstance(config, dict):
            config = ModelStudyConfig(**config)

        if config['verbose']:
            print("\n" + "="*80)
            print("【训练收敛分析】")
            print("="*80)

        # 提取训练历史
        train_rmses, valid_rmses, best_iteration = self._extract_training_history(model)

        # 计算Score Metric（每100轮采样）
        if config['verbose']:
            print("\n计算Score Metric（每100轮采样）...")

        train_scores, valid_scores, score_iterations = self._calculate_score_metrics(
            model, X_train, y_train, X_valid, y_valid, len(train_rmses)
        )

        # 找到最佳Score迭代
        best_score_idx = np.argmax(valid_scores)
        best_score_iteration = score_iterations[best_score_idx]
        best_valid_score = valid_scores[best_score_idx]

        # 计算诊断指标（处理valid_score=0的情况）
        if abs(best_valid_score) > 1e-8:
            overfitting_gap = (train_scores[best_score_idx] - best_valid_score) / best_valid_score * 100
        else:
            # Valid Score接近0，使用绝对差值
            overfitting_gap = train_scores[best_score_idx] - best_valid_score
        convergence_speed = (train_rmses[0] - train_rmses[min(500, len(train_rmses)-1)]) / train_rmses[0] * 100

        if config['verbose']:
            print(f"\n✅ 分析完成:")
            print(f"   - 最佳RMSE迭代: 第 {best_iteration} 轮")
            print(f"   - 最佳Score迭代: 第 {best_score_iteration} 轮")
            print(f"   - 最佳Valid Score: {best_valid_score:.6f}")
            print(f"   - 过拟合程度: {overfitting_gap:.1f}%")
            print(f"   - 收敛速度: {convergence_speed:.1f}%")

        # 可视化
        self._plot_convergence(
            train_rmses, valid_rmses, best_iteration,
            score_iterations, train_scores, valid_scores, best_score_iteration,
            config, hyperparams
        )

        # 保存超参数到JSON
        if config['save_plots'] and hyperparams is not None:
            self._save_hyperparams(hyperparams, overfitting_gap, convergence_speed,
                                   best_iteration, best_valid_score, config)

        # 诊断建议
        if config['verbose']:
            self._print_diagnostics(
                model, best_iteration, len(train_rmses),
                overfitting_gap, convergence_speed
            )

        return {
            'model': model,
            'best_iteration': best_iteration,
            'best_score_iteration': best_score_iteration,
            'best_valid_score': best_valid_score,
            'overfitting_gap': overfitting_gap,
            'convergence_speed': convergence_speed,
            'train_rmses': train_rmses,
            'valid_rmses': valid_rmses,
            'train_scores': train_scores,
            'valid_scores': valid_scores,
            'score_iterations': score_iterations,
        }

    def _extract_training_history(self, model) -> Tuple[List, List, int]:
        """提取训练历史（RMSE）"""
        if self.model_type == 'lstm':
            # LSTM模型的训练历史存储在model对象的属性中
            if not hasattr(model, 'train_losses'):
                raise ValueError("LSTM model must have train_losses attribute")

            # LSTM使用MSE loss，需要转换为RMSE
            train_rmses = [np.sqrt(loss) for loss in model.train_losses]
            valid_rmses = [np.sqrt(loss) for loss in model.valid_losses]
            best_iteration = model.best_epoch

        elif self.model_type == 'lightgbm':
            if not hasattr(model, 'evals_result_'):
                raise ValueError("Model must be trained with eval_set to extract training history")

            train_rmses = model.evals_result_['train']['l2']
            valid_rmses = model.evals_result_['valid']['l2']
            # 当没有early stopping时，best_iteration_可能是0，用实际训练轮数代替
            best_iteration = model.best_iteration_ if model.best_iteration_ > 0 else len(train_rmses)

        elif self.model_type == 'xgboost':
            if not hasattr(model, 'evals_result'):
                raise ValueError("Model must be trained with eval_set to extract training history")

            train_rmses = model.evals_result['validation_0']['rmse']
            valid_rmses = model.evals_result['validation_1']['rmse']
            best_iteration = model.best_iteration

        elif self.model_type == 'catboost':
            if not hasattr(model, 'evals_result_'):
                raise ValueError("Model must be trained with eval_set to extract training history")

            train_rmses = model.evals_result_['learn']['RMSE']
            valid_rmses = model.evals_result_['validation']['RMSE']
            best_iteration = model.get_best_iteration()

        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        return train_rmses, valid_rmses, best_iteration

    def _calculate_score_metrics(
        self,
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        total_iterations: int
    ) -> Tuple[List, List, List]:
        """计算Score Metric（每100轮或每10epoch采样）"""
        train_scores = []
        valid_scores = []
        score_iterations = []

        if self.model_type == 'lstm':
            # LSTM: 直接使用训练时计算的scores（每10个epoch采样）
            for i in range(0, len(model.train_scores), max(1, len(model.train_scores) // 20)):
                score_iterations.append(i)
                train_scores.append(model.train_scores[i])
                valid_scores.append(model.valid_scores[i])
            # 确保包含最后一个epoch
            if len(model.train_scores) - 1 not in score_iterations:
                score_iterations.append(len(model.train_scores) - 1)
                train_scores.append(model.train_scores[-1])
                valid_scores.append(model.valid_scores[-1])
            return train_scores, valid_scores, score_iterations

        for i in range(0, total_iterations, 100):
            iteration = i + 1

            # 预测
            if self.model_type == 'lightgbm':
                y_train_pred = model.predict(X_train, num_iteration=iteration)
                y_valid_pred = model.predict(X_valid, num_iteration=iteration)
            elif self.model_type == 'xgboost':
                y_train_pred = model.predict(X_train, iteration_range=(0, iteration))
                y_valid_pred = model.predict(X_valid, iteration_range=(0, iteration))
            elif self.model_type == 'catboost':
                y_train_pred = model.predict(X_train, ntree_end=iteration)
                y_valid_pred = model.predict(X_valid, ntree_end=iteration)

            # 计算Score（带详细诊断）
            # 训练集使用分位数方法
            train_score = calculate_score_metric(y_train_pred, y_train.values,
                                                use_percentile=True, low_threshold=0.3, high_threshold=0.7)
            # 验证集使用分位数方法，拼接训练集尾部
            valid_score, valid_diagnostics = self._calculate_score_with_diagnostics(
                y_valid_pred, y_valid.values, train_pred_history=y_train_pred, use_percentile=True)

            # 调试：打印所有采样点的详细信息
            # 训练集预测分布（详细百分位数）
            train_pred_mean = float(y_train_pred.mean())
            train_pred_std = float(y_train_pred.std())
            train_pred_min = float(y_train_pred.min())
            train_pred_max = float(y_train_pred.max())
            train_pred_p1 = float(np.percentile(y_train_pred, 1))
            train_pred_p5 = float(np.percentile(y_train_pred, 5))
            train_pred_p10 = float(np.percentile(y_train_pred, 10))
            train_pred_p25 = float(np.percentile(y_train_pred, 25))
            train_pred_p50 = float(np.percentile(y_train_pred, 50))
            train_pred_p75 = float(np.percentile(y_train_pred, 75))
            train_pred_p90 = float(np.percentile(y_train_pred, 90))
            train_pred_p95 = float(np.percentile(y_train_pred, 95))
            train_pred_p99 = float(np.percentile(y_train_pred, 99))

            # 统计仓位分布（训练集）- 固定阈值方法
            train_position_fixed = np.where(y_train_pred > 0.001, 2.0, np.where(y_train_pred > 0, 1.0, 0.0))
            train_pos_0_fixed = int((train_position_fixed == 0).sum())
            train_pos_1_fixed = int((train_position_fixed == 1).sum())
            train_pos_2_fixed = int((train_position_fixed == 2).sum())

            # 统计仓位分布（训练集）- 分位数方法
            train_position_pct = np.zeros(len(y_train_pred))
            for i in range(len(y_train_pred)):
                percentile = (y_train_pred < y_train_pred[i]).sum() / len(y_train_pred)
                if percentile >= 0.7:
                    train_position_pct[i] = 2.0
                elif percentile <= 0.3:
                    train_position_pct[i] = 0.0
                else:
                    train_position_pct[i] = 1.0
            train_pos_0_pct = int((train_position_pct == 0).sum())
            train_pos_1_pct = int((train_position_pct == 1).sum())
            train_pos_2_pct = int((train_position_pct == 2).sum())

            print(f"\n  [Iter {iteration}]")
            print(f"    === 训练集 (N={len(y_train_pred)}) ===")
            print(f"    Train Score: {train_score:.6f}")
            print(f"    预测统计: 均值={train_pred_mean:.6f}, std={train_pred_std:.6f}")
            print(f"    预测范围: [{train_pred_min:.6f}, {train_pred_max:.6f}]")
            print(f"    百分位数分布:")
            print(f"      P1={train_pred_p1:.6f}, P5={train_pred_p5:.6f}, P10={train_pred_p10:.6f}")
            print(f"      P25={train_pred_p25:.6f}, P50={train_pred_p50:.6f}, P75={train_pred_p75:.6f}")
            print(f"      P90={train_pred_p90:.6f}, P95={train_pred_p95:.6f}, P99={train_pred_p99:.6f}")
            print(f"    仓位分布-固定阈值(0.001): 0仓={train_pos_0_fixed}({train_pos_0_fixed/len(y_train_pred)*100:.1f}%), "
                  f"1仓={train_pos_1_fixed}({train_pos_1_fixed/len(y_train_pred)*100:.1f}%), "
                  f"2仓={train_pos_2_fixed}({train_pos_2_fixed/len(y_train_pred)*100:.1f}%)")
            print(f"    仓位分布-分位数方法(P30/P70): 0仓={train_pos_0_pct}({train_pos_0_pct/len(y_train_pred)*100:.1f}%), "
                  f"1仓={train_pos_1_pct}({train_pos_1_pct/len(y_train_pred)*100:.1f}%), "
                  f"2仓={train_pos_2_pct}({train_pos_2_pct/len(y_train_pred)*100:.1f}%)")

            print(f"\n    === 验证集 (N={len(y_valid_pred)}) ===")
            print(f"    Valid Score: {valid_score:.6f}")
            print(f"    【使用分位数方法：训练集尾部20个 + 验证集7个 = 窗口27个】")
            if valid_diagnostics:
                print(f"    预测范围: [{valid_diagnostics['pred_min']:.6f}, {valid_diagnostics['pred_max']:.6f}]")
                print(f"    预测std: {valid_diagnostics['pred_std']:.6f}")
                print(f"    预测值: {y_valid_pred}")
                print(f"    仓位分布(P30/P70): 0仓={valid_diagnostics['pos_0']}, 1仓={valid_diagnostics['pos_1']}, 2仓={valid_diagnostics['pos_2']}")
                print(f"    策略收益std: {valid_diagnostics['strategy_std']:.6f}")
                print(f"    策略累积收益: {valid_diagnostics['cumulative_return']:.6f}")
                print(f"\n    === Score分解诊断 ===")
                print(f"    策略年化收益: {valid_diagnostics.get('strategy_annual_return', 0):.2f}%")
                print(f"    市场年化收益: {valid_diagnostics.get('market_annual_return', 0):.2f}%")
                print(f"    策略年化波动率: {valid_diagnostics.get('strategy_volatility', 0):.2f}%")
                print(f"    市场年化波动率: {valid_diagnostics.get('market_volatility', 0):.2f}%")
                print(f"    原始Sharpe: {valid_diagnostics.get('sharpe', 0):.4f}")
                print(f"    波动率惩罚: {valid_diagnostics.get('vol_penalty', 1):.4f}x")
                print(f"    收益惩罚: {valid_diagnostics.get('return_penalty', 1):.4f}x")
                print(f"    最终Score = Sharpe / (vol_penalty × return_penalty) = {valid_score:.6f}")
                if valid_diagnostics['fail_reason'] != 'None':
                    print(f"    失败原因: {valid_diagnostics['fail_reason']}")

            train_scores.append(train_score)
            valid_scores.append(valid_score)
            score_iterations.append(iteration)

        return train_scores, valid_scores, score_iterations

    def _calculate_score_with_diagnostics(self, y_pred, y_true, risk_free_rate=0.0,
                                          train_pred_history=None, use_percentile=True):
        """
        计算Score并返回详细诊断信息

        Args:
            y_pred: 验证集预测值
            y_true: 验证集真实值
            risk_free_rate: 无风险利率
            train_pred_history: 训练集预测值（用于拼接窗口）
            use_percentile: 是否使用分位数方法
        """
        diagnostics = {
            'pred_min': float(y_pred.min()),
            'pred_max': float(y_pred.max()),
            'pred_std': float(y_pred.std()),
            'fail_reason': 'None'
        }

        # 仓位转换
        if use_percentile and train_pred_history is not None:
            # 方案1：拼接训练集尾部（最后20个）+ 验证集
            history_window = np.concatenate([train_pred_history[-20:], y_pred])

            position = np.zeros(len(y_pred))
            for i in range(len(y_pred)):
                # 使用拼接后的历史计算分位数
                current_pred = y_pred[i]
                percentile = (history_window < current_pred).sum() / len(history_window)

                if percentile >= 0.7:
                    position[i] = 2.0
                elif percentile <= 0.3:
                    position[i] = 0.0
                else:
                    position[i] = 1.0
        else:
            # 固定阈值方法（回退）
            position = np.where(y_pred > 0.001, 2.0, np.where(y_pred > 0, 1.0, 0.0))

        diagnostics['pos_0'] = int((position == 0).sum())
        diagnostics['pos_1'] = int((position == 1).sum())
        diagnostics['pos_2'] = int((position == 2).sum())

        # 策略收益
        forward_returns = y_true + risk_free_rate
        strategy_returns = risk_free_rate * (1 - position) + position * forward_returns
        strategy_excess_returns = strategy_returns - risk_free_rate

        diagnostics['strategy_std'] = float(strategy_returns.std())

        # 累积收益
        strategy_excess_cumulative = (1 + strategy_excess_returns).prod()
        diagnostics['cumulative_return'] = float(strategy_excess_cumulative)

        # 检查失败原因
        if len(y_pred) == 0 or len(y_true) == 0:
            diagnostics['fail_reason'] = 'Empty input'
            return 0.0, diagnostics

        if strategy_excess_cumulative <= 0:
            diagnostics['fail_reason'] = 'Cumulative return <= 0'
            return 0.0, diagnostics

        if diagnostics['strategy_std'] < 1e-8:
            diagnostics['fail_reason'] = 'Strategy std too small'
            return 0.0, diagnostics

        # 正常计算Score（带详细分解）
        score = calculate_score_metric(y_pred, y_true, risk_free_rate,
                                       use_percentile=use_percentile,
                                       low_threshold=0.3, high_threshold=0.7)

        # 添加Score分解诊断
        strategy_mean_excess_return = (diagnostics['cumulative_return']) ** (1 / len(y_pred)) - 1
        strategy_volatility = float(diagnostics['strategy_std'] * np.sqrt(252) * 100)

        # 市场指标
        forward_returns = y_true + risk_free_rate
        market_excess_cumulative = (1 + (forward_returns - risk_free_rate)).prod()
        if market_excess_cumulative > 0:
            market_mean_excess_return = (market_excess_cumulative) ** (1 / len(y_pred)) - 1
            market_std = forward_returns.std()
            market_volatility = float(market_std * np.sqrt(252) * 100)
        else:
            market_mean_excess_return = 0
            market_volatility = 0

        diagnostics['strategy_annual_return'] = float(strategy_mean_excess_return * 252 * 100)
        diagnostics['strategy_volatility'] = strategy_volatility
        diagnostics['market_annual_return'] = float(market_mean_excess_return * 252 * 100)
        diagnostics['market_volatility'] = market_volatility

        # Sharpe比率
        if diagnostics['strategy_std'] > 1e-8:
            sharpe = strategy_mean_excess_return / diagnostics['strategy_std'] * np.sqrt(252)
            diagnostics['sharpe'] = float(sharpe)
        else:
            diagnostics['sharpe'] = 0.0

        # 波动率惩罚
        if market_volatility > 0:
            excess_vol = max(0, strategy_volatility / market_volatility - 1.2)
            vol_penalty = 1 + excess_vol
        else:
            vol_penalty = 1.0
        diagnostics['vol_penalty'] = float(vol_penalty)

        # 收益惩罚
        return_gap = max(0, (market_mean_excess_return - strategy_mean_excess_return) * 100 * 252)
        return_penalty = 1 + (return_gap**2) / 100
        diagnostics['return_penalty'] = float(return_penalty)

        return score, diagnostics

    def _plot_convergence(
        self,
        train_rmses, valid_rmses, best_iteration,
        score_iterations, train_scores, valid_scores, best_score_iteration,
        config, hyperparams=None
    ):
        """绘制收敛曲线（优化版：双Y轴+Score诊断）"""
        # 创建2x2子图布局
        fig = plt.figure(figsize=(config['figsize'][0], config['figsize'][1] * 1.2))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])  # 左上：RMSE
        ax2 = fig.add_subplot(gs[0, 1])  # 右上：Score (双Y轴)
        ax3 = fig.add_subplot(gs[1, :])  # 下方：Score诊断条形图

        # 如果有超参数，在图表上方添加标题显示关键参数
        if hyperparams is not None:
            param_text = f"n_estimators={hyperparams.get('n_estimators', 'N/A')}, " \
                        f"lr={hyperparams.get('learning_rate', 'N/A'):.4f}, " \
                        f"max_depth={hyperparams.get('max_depth', 'N/A')}, " \
                        f"reg_lambda={hyperparams.get('reg_lambda', 'N/A'):.4f}"
            fig.suptitle(f'Training Convergence Analysis\n{param_text}',
                        fontsize=12, fontweight='bold', y=0.98)

        # ============ 左上图：RMSE收敛曲线 ============
        ax1.plot(train_rmses, label='Train RMSE', linewidth=2, alpha=0.8, color='#1f77b4')
        ax1.plot(valid_rmses, label='Valid RMSE', linewidth=2, alpha=0.8, color='#ff7f0e')
        ax1.axvline(best_iteration, color='red', linestyle='--', alpha=0.5,
                   label=f'Early Stop ({best_iteration})')
        ax1.set_xlabel('Iteration', fontsize=11)
        ax1.set_ylabel('RMSE', fontsize=11)
        ax1.set_title('RMSE Convergence', fontsize=13, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)

        # ============ 右上图：Score曲线（双Y轴） ============
        # 左Y轴：RMSE (方便对照)
        color_rmse = '#2ca02c'
        ax2.plot(valid_rmses, label='Valid RMSE', linewidth=1.5, alpha=0.6,
                color=color_rmse, linestyle='--')
        ax2.set_xlabel('Iteration', fontsize=11)
        ax2.set_ylabel('RMSE', fontsize=11, color=color_rmse)
        ax2.tick_params(axis='y', labelcolor=color_rmse)

        # 右Y轴：Score Metric (处理负值)
        ax2_right = ax2.twinx()
        color_score = '#d62728'

        # 标注Score=0的区域
        valid_scores_array = np.array(valid_scores)
        zero_mask = valid_scores_array <= 0
        if zero_mask.any():
            # 找到第一个变为0的位置
            first_zero_idx = np.where(zero_mask)[0][0] if zero_mask.any() else None
            if first_zero_idx is not None and first_zero_idx < len(score_iterations):
                ax2_right.axvline(score_iterations[first_zero_idx], color='orange',
                                 linestyle=':', alpha=0.7, linewidth=2,
                                 label=f'Score→0 (iter {score_iterations[first_zero_idx]})')

        ax2_right.plot(score_iterations, valid_scores, label='Valid Score',
                      linewidth=2.5, alpha=0.9, marker='s', markersize=5,
                      color=color_score, markerfacecolor='white', markeredgewidth=1.5)
        ax2_right.plot(score_iterations, train_scores, label='Train Score',
                      linewidth=1.5, alpha=0.5, marker='o', markersize=3,
                      color='#9467bd')

        # 标注最佳Score
        ax2_right.axvline(best_score_iteration, color='red', linestyle='--', alpha=0.5,
                         label=f'Best Score (iter {best_score_iteration})')

        ax2_right.set_ylabel('Score Metric', fontsize=11, color=color_score)
        ax2_right.tick_params(axis='y', labelcolor=color_score)

        # 允许显示负值
        score_min = min(min(valid_scores), min(train_scores))
        score_max = max(max(valid_scores), max(train_scores))
        if score_min < 0:
            ax2_right.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
            ax2_right.set_ylim(score_min * 1.1, score_max * 1.1)

        ax2.set_title('Score vs RMSE (Dual Y-axis)', fontsize=13, fontweight='bold')

        # 合并图例
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_right.get_legend_handles_labels()
        ax2_right.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)

        # ============ 下方：Score关键轮次对比条形图 ============
        # 选择关键轮次：第1轮、最佳Score轮、第一个Score=0轮、最后一轮
        key_iterations = [0]  # 第1轮（索引0）

        if best_score_iteration in score_iterations:
            key_iterations.append(score_iterations.index(best_score_iteration))

        # 找第一个Score=0的轮次
        first_zero_iter_idx = None
        for i, score in enumerate(valid_scores):
            if score <= 0:
                first_zero_iter_idx = i
                break
        if first_zero_iter_idx is not None:
            key_iterations.append(first_zero_iter_idx)

        if len(score_iterations) - 1 not in key_iterations:
            key_iterations.append(len(score_iterations) - 1)  # 最后一轮

        key_iterations = sorted(set(key_iterations))[:4]  # 最多4个关键点

        x_labels = []
        train_score_vals = []
        valid_score_vals = []

        for idx in key_iterations:
            if idx < len(score_iterations):
                iter_num = score_iterations[idx]
                x_labels.append(f'Iter {iter_num}')
                train_score_vals.append(train_scores[idx])
                valid_score_vals.append(valid_scores[idx])

        x_pos = np.arange(len(x_labels))
        width = 0.35

        bars1 = ax3.bar(x_pos - width/2, train_score_vals, width,
                       label='Train Score', alpha=0.8, color='#9467bd')
        bars2 = ax3.bar(x_pos + width/2, valid_score_vals, width,
                       label='Valid Score', alpha=0.8, color=color_score)

        # 在柱子上标注数值
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if abs(height) > 1e-3:
                    ax3.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.3f}',
                            ha='center', va='bottom' if height > 0 else 'top',
                            fontsize=8, fontweight='bold')

        ax3.set_xlabel('Key Iterations', fontsize=11)
        ax3.set_ylabel('Score Value', fontsize=11)
        ax3.set_title('Score Comparison at Key Iterations', fontsize=13, fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(x_labels)
        ax3.legend(loc='best', fontsize=9)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=1)

        if config['save_plots']:
            plot_path = os.path.join(config['plot_dir'], 'convergence_analysis.png')
            plt.savefig(plot_path, dpi=config['dpi'], bbox_inches='tight')
            if config['verbose']:
                print(f"✅ 图片已保存: {plot_path}")

        plt.show()

    def _save_hyperparams(
        self,
        hyperparams: Dict,
        overfitting_gap: float,
        convergence_speed: float,
        best_iteration: int,
        best_valid_score: float,
        config: ModelStudyConfig
    ):
        """保存超参数和诊断结果到JSON文件"""
        import datetime

        save_data = {
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'hyperparameters': hyperparams,
            'diagnostics': {
                'best_iteration': int(best_iteration),
                'best_valid_score': float(best_valid_score),
                'overfitting_gap_percent': float(overfitting_gap),
                'convergence_speed_percent': float(convergence_speed),
            },
            'training_info': {
                'model_type': self.model_type,
                'configured_n_estimators': hyperparams.get('n_estimators', None),
                'actual_best_iteration': int(best_iteration),
            }
        }

        json_path = os.path.join(config['plot_dir'], 'hyperparams_and_diagnostics.json')
        with open(json_path, 'w') as f:
            json.dump(save_data, f, indent=2)

        if config['verbose']:
            print(f"✅ 超参数已保存: {json_path}")

    def _print_diagnostics(
        self,
        model,
        best_iteration: int,
        total_iterations: int,
        overfitting_gap: float,
        convergence_speed: float
    ):
        """打印诊断建议"""
        print("\n" + "="*80)
        print("【诊断建议】")
        print("="*80)

        # 从模型中提取参数
        if self.model_type == 'lightgbm':
            n_estimators = model.n_estimators
            learning_rate = model.learning_rate
            max_depth = model.max_depth
            reg_lambda = model.reg_lambda
            reg_alpha = model.reg_alpha
        else:
            n_estimators = total_iterations
            learning_rate = None
            max_depth = None
            reg_lambda = None
            reg_alpha = None

        # 1. 迭代数诊断
        iteration_ratio = best_iteration / n_estimators
        if iteration_ratio < 0.5:
            print(f"⚠️ Early stopping在 {best_iteration} 轮（早于一半，{iteration_ratio*100:.1f}%）")
            print(f"   说明: 模型快速收敛，{n_estimators}轮设置过多")
            print(f"   建议: 可减少 n_estimators 到 {int(best_iteration * 1.5)} 左右")
        elif iteration_ratio > 0.9:
            print(f"⚠️ Early stopping在 {best_iteration} 轮（接近上限，{iteration_ratio*100:.1f}%）")
            print(f"   说明: 模型未充分训练，需要更多迭代")
            print(f"   建议: 增加 n_estimators 到 {int(best_iteration * 1.5)} 以上")
        else:
            print(f"✅ Early stopping在 {best_iteration} 轮（合理范围，{iteration_ratio*100:.1f}%）")

        # 2. 过拟合诊断
        if overfitting_gap > 30:
            print(f"\n⚠️ 过拟合严重（train-valid gap = {overfitting_gap:.1f}%）")
            print(f"   建议: 增强正则化")
            if reg_lambda is not None:
                print(f"         - 提高 reg_lambda (当前={reg_lambda:.3f})")
            if reg_alpha is not None:
                print(f"         - 提高 reg_alpha (当前={reg_alpha:.3f})")
            if max_depth is not None:
                print(f"         - 减少 max_depth (当前={max_depth})")
        elif overfitting_gap > 15:
            print(f"\n⚠️ 轻微过拟合（train-valid gap = {overfitting_gap:.1f}%）")
            print(f"   建议: 适当增强正则化")
        else:
            print(f"\n✅ 过拟合程度可接受（train-valid gap = {overfitting_gap:.1f}%）")

        # 3. 收敛速度诊断
        if convergence_speed < 30:
            print(f"\n⚠️ 收敛速度较慢（前500轮RMSE仅下降{convergence_speed:.1f}%）")
            if learning_rate is not None:
                print(f"   建议: 提高 learning_rate（当前={learning_rate:.4f}）")
        else:
            print(f"\n✅ 收敛速度正常（前500轮RMSE下降{convergence_speed:.1f}%）")

        print("="*80)


# ============================================================
# 2. 特征重要性分析器
# ============================================================

class FeatureImportanceAnalyzer:
    """
    特征重要性分析器
    支持模型: LightGBM, XGBoost, CatBoost（基于内置importance）
              以及任何模型（基于Permutation Importance）

    分析内容:
    - Top N重要特征可视化
    - 特征分类统计（原始因子、lagged、衍生特征等）
    - 维度灾难诊断（特征数/样本数比例）
    """

    def __init__(self, importance_type: str = 'gain'):
        """
        Args:
            importance_type: 重要性类型
                - 'gain': 基于split gain（LightGBM/XGBoost）
                - 'split': 基于split count
                - 'permutation': 基于Permutation Importance（模型无关）
        """
        self.importance_type = importance_type

    def analyze(
        self,
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        feature_names: List[str],
        lagged_features: Optional[List[str]] = None,
        original_factors: Optional[List[str]] = None,
        config: Union[Dict, ModelStudyConfig] = None
    ) -> Dict[str, Any]:
        """
        执行特征重要性分析

        Args:
            model: 已训练的模型
            X_train: 训练集特征（用于Permutation Importance）
            y_train: 训练集标签（用于Permutation Importance）
            feature_names: 特征名称列表
            lagged_features: lagged特征列表（用于分类统计）
            original_factors: 原始因子列表（用于分类统计）
            config: 配置字典或ModelStudyConfig对象

        Returns:
            results: 分析结果字典
                - importances: 特征重要性数组
                - feature_names: 特征名称
                - top_features: Top N特征
                - category_stats: 特征分类统计
        """
        if config is None:
            config = ModelStudyConfig()
        elif isinstance(config, dict):
            config = ModelStudyConfig(**config)

        if config['verbose']:
            print("\n" + "="*80)
            print("【特征重要性分析】")
            print("="*80)

        # 提取特征重要性
        if self.importance_type == 'permutation':
            importances = self._permutation_importance(model, X_train, y_train)
        else:
            importances = self._extract_importance(model)

        # Top N特征
        top_n = config['top_n_features']
        top_indices = np.argsort(importances)[-top_n:][::-1]
        top_features = [(feature_names[i], importances[i]) for i in top_indices]

        if config['verbose']:
            print(f"\n✅ Top {top_n} 重要特征:")
            for rank, (feat, imp) in enumerate(top_features[:10], 1):
                print(f"   {rank:2d}. {feat:40s} {imp:10.2f}")

        # 特征分类统计
        if lagged_features is not None and original_factors is not None:
            category_stats = self._categorize_features(
                feature_names, importances, lagged_features, original_factors
            )

            if config['verbose']:
                print(f"\n📊 特征分类统计:")
                for cat, info in category_stats.items():
                    print(f"   {cat:20s}: {info['count']:4d} 个, 总重要性={info['total_importance']:.2f}")
        else:
            category_stats = None

        # 维度灾难诊断
        n_features = len(feature_names)
        n_samples = len(X_train)
        feature_sample_ratio = n_features / n_samples

        if config['verbose']:
            print(f"\n🔍 维度诊断:")
            print(f"   特征数: {n_features}")
            print(f"   样本数: {n_samples}")
            print(f"   特征/样本比: {feature_sample_ratio:.4f}")

            if feature_sample_ratio > 0.5:
                print(f"   ⚠️ 维度灾难风险高！特征数接近样本数")
            elif feature_sample_ratio > 0.2:
                print(f"   ⚠️ 维度较高，建议特征选择")
            else:
                print(f"   ✅ 维度合理")

        # 可视化
        self._plot_importance(
            top_features, category_stats, config
        )

        return {
            'importances': importances,
            'feature_names': feature_names,
            'top_features': top_features,
            'category_stats': category_stats,
            'feature_sample_ratio': feature_sample_ratio,
        }

    def _extract_importance(self, model) -> np.ndarray:
        """提取模型内置importance"""
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        else:
            raise ValueError("Model does not have feature_importances_ attribute. "
                           "Use importance_type='permutation' instead.")

    def _permutation_importance(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        n_repeats: int = 5
    ) -> np.ndarray:
        """
        计算Permutation Importance（模型无关方法）
        注意：这个方法计算量大，仅在必要时使用
        """
        from sklearn.inspection import permutation_importance

        result = permutation_importance(
            model, X, y, n_repeats=n_repeats,
            random_state=42, n_jobs=-1
        )
        return result.importances_mean

    def _categorize_features(
        self,
        feature_names: List[str],
        importances: np.ndarray,
        lagged_features: List[str],
        original_factors: List[str]
    ) -> Dict[str, Dict]:
        """
        特征分类统计
        分类：lagged, original_factor, lag_derived, rolling_derived, pct_change, interaction
        """
        categories = {
            'lagged': {'count': 0, 'total_importance': 0.0},
            'original_factor': {'count': 0, 'total_importance': 0.0},
            'lag_derived': {'count': 0, 'total_importance': 0.0},
            'rolling_derived': {'count': 0, 'total_importance': 0.0},
            'pct_change': {'count': 0, 'total_importance': 0.0},
            'interaction': {'count': 0, 'total_importance': 0.0},
        }

        for feat, imp in zip(feature_names, importances):
            cat = self._categorize_single_feature(feat, lagged_features, original_factors)
            categories[cat]['count'] += 1
            categories[cat]['total_importance'] += imp

        return categories

    def _categorize_single_feature(
        self,
        feat_name: str,
        lagged_features: List[str],
        original_factors: List[str]
    ) -> str:
        """单个特征分类"""
        if feat_name in lagged_features:
            return 'lagged'
        elif feat_name in original_factors:
            return 'original_factor'
        elif '_lag_' in feat_name:
            return 'lag_derived'
        elif any(kw in feat_name for kw in ['_rolling_', '_ewm_', '_expanding_']):
            return 'rolling_derived'
        elif '_pct_' in feat_name:
            return 'pct_change'
        elif '_x_' in feat_name or '_div_' in feat_name:
            return 'interaction'
        else:
            return 'original_factor'  # 默认归类为原始因子

    def _plot_importance(
        self,
        top_features: List[Tuple[str, float]],
        category_stats: Optional[Dict],
        config: ModelStudyConfig
    ):
        """绘制特征重要性图表"""
        if category_stats is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=config['figsize'])
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(config['figsize'][0] // 2, config['figsize'][1]))

        # 左图：Top N特征条形图
        features, importances = zip(*top_features)
        y_pos = np.arange(len(features))

        ax1.barh(y_pos, importances, alpha=0.8)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(features, fontsize=10)
        ax1.invert_yaxis()
        ax1.set_xlabel('Importance (Gain)', fontsize=12)
        ax1.set_title(f'Top {len(features)} Feature Importance', fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)

        # 右图：分类饼图
        if category_stats is not None:
            labels = []
            sizes = []
            for cat, info in category_stats.items():
                if info['count'] > 0:
                    labels.append(f"{cat}\n({info['count']})")
                    sizes.append(info['total_importance'])

            ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Feature Importance by Category', fontsize=14, fontweight='bold')

        plt.tight_layout()

        if config['save_plots']:
            plot_path = os.path.join(config['plot_dir'], 'feature_importance.png')
            plt.savefig(plot_path, dpi=config['dpi'], bbox_inches='tight')
            if config['verbose']:
                print(f"✅ 图片已保存: {plot_path}")

        plt.show()


# ============================================================
# 3. 时序稳定性分析器
# ============================================================

class TemporalStabilityAnalyzer:
    """
    时序稳定性分析器（模型无关）

    分析内容:
    - IC（信息系数）随时间的衰减趋势
    - Score Metric随时间的变化
    - 预测稳定性诊断
    """

    def analyze(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        config: Union[Dict, ModelStudyConfig] = None
    ) -> Dict[str, Any]:
        """
        执行时序稳定性分析

        Args:
            model: 已训练的模型
            X: 完整数据集（训练+验证）
            y: 完整标签
            config: 配置字典或ModelStudyConfig对象

        Returns:
            results: 分析结果字典
                - ic_series: IC时间序列
                - score_series: Score Metric时间序列
                - ic_trend: IC趋势（上升/下降/稳定）
                - ic_decay_rate: IC衰减率 (%/window)
        """
        if config is None:
            config = ModelStudyConfig()
        elif isinstance(config, dict):
            config = ModelStudyConfig(**config)

        if config['verbose']:
            print("\n" + "="*80)
            print("【时序稳定性分析】")
            print("="*80)

        # 划分时间窗口
        n_windows = config['n_temporal_windows']
        window_size = len(X) // n_windows

        ic_series = []
        score_series = []
        window_indices = []

        for i in range(n_windows):
            start_idx = i * window_size
            end_idx = (i + 1) * window_size if i < n_windows - 1 else len(X)

            X_window = X.iloc[start_idx:end_idx]
            y_window = y.iloc[start_idx:end_idx]

            # 预测
            y_pred = model.predict(X_window)

            # 计算IC
            ic, _ = spearmanr(y_pred, y_window)

            # 计算Score
            score = calculate_score_metric(y_pred, y_window.values)

            ic_series.append(ic)
            score_series.append(score)
            window_indices.append(i + 1)

        # IC趋势分析
        ic_trend = self._analyze_trend(ic_series)
        ic_decay_rate = self._calculate_decay_rate(ic_series)

        if config['verbose']:
            print(f"\n✅ 分析完成:")
            print(f"   - IC趋势: {ic_trend}")
            print(f"   - IC衰减率: {ic_decay_rate:.2f}%/window")
            print(f"   - 平均IC: {np.mean(ic_series):.4f}")
            print(f"   - IC标准差: {np.std(ic_series):.4f}")

        # 可视化
        self._plot_temporal_stability(
            window_indices, ic_series, score_series, config
        )

        # 诊断
        if config['verbose']:
            self._print_diagnostics(ic_trend, ic_decay_rate, ic_series)

        return {
            'ic_series': ic_series,
            'score_series': score_series,
            'window_indices': window_indices,
            'ic_trend': ic_trend,
            'ic_decay_rate': ic_decay_rate,
        }

    def _analyze_trend(self, ic_series: List[float]) -> str:
        """分析IC趋势"""
        from scipy.stats import linregress

        x = np.arange(len(ic_series))
        slope, _, _, p_value, _ = linregress(x, ic_series)

        if p_value > 0.05:
            return '稳定'
        elif slope > 0:
            return '上升'
        else:
            return '下降'

    def _calculate_decay_rate(self, ic_series: List[float]) -> float:
        """计算IC衰减率（%/window）"""
        if len(ic_series) < 2:
            return 0.0

        first_half_mean = np.mean(ic_series[:len(ic_series)//2])
        second_half_mean = np.mean(ic_series[len(ic_series)//2:])

        if first_half_mean == 0:
            return 0.0

        decay_rate = (first_half_mean - second_half_mean) / first_half_mean * 100
        return decay_rate

    def _plot_temporal_stability(
        self,
        window_indices: List[int],
        ic_series: List[float],
        score_series: List[float],
        config: ModelStudyConfig
    ):
        """绘制时序稳定性图表"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=config['figsize'])

        # 左图：IC随时间变化
        ax1.plot(window_indices, ic_series, marker='o', linewidth=2, markersize=6, alpha=0.8)
        ax1.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax1.axhline(np.mean(ic_series), color='green', linestyle='--', alpha=0.5,
                   label=f'Mean IC={np.mean(ic_series):.4f}')
        ax1.set_xlabel('Time Window', fontsize=12)
        ax1.set_ylabel('IC (Spearman)', fontsize=12)
        ax1.set_title('IC Over Time', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 右图：Score Metric随时间变化
        ax2.plot(window_indices, score_series, marker='s', linewidth=2, markersize=6, alpha=0.8, color='orange')
        ax2.axhline(np.mean(score_series), color='green', linestyle='--', alpha=0.5,
                   label=f'Mean Score={np.mean(score_series):.4f}')
        ax2.set_xlabel('Time Window', fontsize=12)
        ax2.set_ylabel('Score Metric', fontsize=12)
        ax2.set_title('Score Metric Over Time', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if config['save_plots']:
            plot_path = os.path.join(config['plot_dir'], 'temporal_stability.png')
            plt.savefig(plot_path, dpi=config['dpi'], bbox_inches='tight')
            if config['verbose']:
                print(f"✅ 图片已保存: {plot_path}")

        plt.show()

    def _print_diagnostics(self, ic_trend: str, ic_decay_rate: float, ic_series: List[float]):
        """打印诊断建议"""
        print("\n" + "="*80)
        print("【诊断建议】")
        print("="*80)

        if ic_trend == '下降':
            print(f"⚠️ IC呈下降趋势（衰减率={ic_decay_rate:.2f}%/window）")
            print(f"   说明: 模型预测能力随时间衰减，可能存在过拟合或市场环境变化")
            print(f"   建议:")
            print(f"         - 检查训练数据是否过于历史（增加时间权重）")
            print(f"         - 考虑在线学习或定期重训练")
            print(f"         - 增加时序特征（移动平均、动量等）")
        elif ic_trend == '上升':
            print(f"✅ IC呈上升趋势")
            print(f"   说明: 模型泛化能力良好")
        else:
            print(f"✅ IC稳定（标准差={np.std(ic_series):.4f}）")

        # IC绝对值诊断
        mean_ic = np.mean(ic_series)
        if abs(mean_ic) < 0.01:
            print(f"\n⚠️ IC绝对值过低（{mean_ic:.4f}）")
            print(f"   说明: 模型几乎没有预测能力")
            print(f"   建议: 重新设计特征或更换模型")
        elif abs(mean_ic) < 0.05:
            print(f"\n⚠️ IC较弱（{mean_ic:.4f}）")
            print(f"   建议: 增强特征工程或调优模型")
        else:
            print(f"\n✅ IC正常（{mean_ic:.4f}）")

        print("="*80)


# ============================================================
# 4. 一键分析接口
# ============================================================

def run_full_analysis(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    feature_names: List[str],
    lagged_features: Optional[List[str]] = None,
    original_factors: Optional[List[str]] = None,
    model_type: str = 'lightgbm',
    config: Optional[Union[Dict, ModelStudyConfig]] = None
) -> Dict[str, Any]:
    """
    一键完成所有模型分析

    Args:
        model: 已训练的模型
        X_train: 训练集特征
        y_train: 训练集标签
        X_valid: 验证集特征
        y_valid: 验证集标签
        feature_names: 特征名称列表
        lagged_features: lagged特征列表
        original_factors: 原始因子列表
        model_type: 模型类型 ('lightgbm', 'xgboost', 'catboost')
        config: 配置字典或ModelStudyConfig对象

    Returns:
        results: 包含所有分析结果的字典
            - convergence: 收敛分析结果
            - importance: 特征重要性分析结果
            - temporal: 时序稳定性分析结果
    """
    if config is None:
        config = ModelStudyConfig()
    elif isinstance(config, dict):
        config = ModelStudyConfig(**config)

    print("\n" + "="*80)
    print("【模型综合分析】开始...")
    print("="*80)

    results = {}

    # 1. 收敛分析
    try:
        conv_analyzer = ConvergenceAnalyzer(model_type=model_type)
        results['convergence'] = conv_analyzer.analyze(
            model, X_train, y_train, X_valid, y_valid, config
        )
    except Exception as e:
        print(f"\n⚠️ 收敛分析失败: {e}")
        results['convergence'] = None

    # 2. 特征重要性分析
    try:
        imp_analyzer = FeatureImportanceAnalyzer(importance_type='gain')
        results['importance'] = imp_analyzer.analyze(
            model, X_train, y_train, feature_names,
            lagged_features, original_factors, config
        )
    except Exception as e:
        print(f"\n⚠️ 特征重要性分析失败: {e}")
        results['importance'] = None

    # 3. 时序稳定性分析
    try:
        temp_analyzer = TemporalStabilityAnalyzer()
        X_full = pd.concat([X_train, X_valid], axis=0)
        y_full = pd.concat([y_train, y_valid], axis=0)
        results['temporal'] = temp_analyzer.analyze(
            model, X_full, y_full, config
        )
    except Exception as e:
        print(f"\n⚠️ 时序稳定性分析失败: {e}")
        results['temporal'] = None

    print("\n" + "="*80)
    print("【模型综合分析】完成！")
    print("="*80)

    # 清理内存
    gc.collect()

    return results


# ============================================================
# 5. 完整分析Pipeline（封装数据准备+训练+分析）
# ============================================================

class ModelStudyPipeline:
    """
    完整的模型分析流程
    
    封装功能：
    1. 数据加载和准备（50因子+lagged特征+预处理+特征工程）
    2. 数据切分（简单train/valid split）
    3. 模型训练（使用指定超参数）
    4. 三大分析（收敛+特征重要性+时序稳定性）
    
    使用示例：
        >>> from toollab.model_study import ModelStudyPipeline
        >>> pipeline = ModelStudyPipeline(
        ...     config=MODEL_STUDY_CONFIG,
        ...     hyperparams=LGBM_HYPERPARAMS
        ... )
        >>> results = pipeline.run_full_analysis()
    """
    
    def __init__(
        self,
        config: Union[Dict, ModelStudyConfig],
        hyperparams: Dict,
        paths: Optional[Dict] = None,
        global_config: Optional[Dict] = None,
        model_type: str = 'lightgbm'
    ):
        """
        Args:
            config: Model Study配置（ModelStudyConfig或字典）
            hyperparams: 模型超参数字典
            paths: 文件路径配置（可选）
            global_config: 全局配置（如USE_FEATURE_ENGINEERING等，可选）
            model_type: 模型类型 ('lightgbm', 'lstm', 'catboost')
        """
        if isinstance(config, dict):
            self.config = ModelStudyConfig(**config)
        else:
            self.config = config

        self.hyperparams = hyperparams
        self.paths = paths or self._default_paths()
        self.global_config = global_config or self._default_global_config()
        self.model_type = model_type.lower()

        # 数据缓存
        self.feature_names = None
        self.lagged_features = None
        self.original_factors = None

        # LSTM专用
        self.x_scaler = None
        self.y_scaler = None
        self.device = None
        if self.model_type == 'lstm' and PYTORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"LSTM模式：使用设备 {self.device}")
    
    def _default_paths(self) -> Dict:
        """默认文件路径"""
        return {
            'train_data': 'train.csv',
            'top50_factors': 'v5_factor/top50_stable_factors.json',
            'preprocessing_rules': 'v5_factor/factor_preprocessing_rules.json',
        }
    
    def _default_global_config(self) -> Dict:
        """默认全局配置"""
        return {
            'TARGET': 'market_forward_excess_returns',
            'TRAIN_WINDOW_SIZE': 2000,
            'VALIDATION_SPLIT': 0.0035,
            'TIME_OFFSET': 14,  # 往前移动14天，避免使用最新数据
            'USE_FEATURE_ENGINEERING': True,
            'USE_SLIM_FEATURES': True,
            'LAG_PERIODS': [1, 3, 5, 7, 14, 20],
            'ROLLING_WINDOWS': [3, 6, 10, 20, 60],
        }
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """
        一键完成所有分析
        
        Returns:
            results: 包含model和3个分析结果的字典
                - model: 训练好的模型
                - convergence: 收敛分析结果
                - importance: 特征重要性结果  
                - temporal: 时序稳定性结果
                - data_info: 数据信息
        """
        print("\n" + "="*80)
        print("【ModelStudyPipeline 完整分析流程】")
        print("="*80)
        
        # 步骤1: 数据准备
        X_train, y_train, X_valid, y_valid, feature_info = self._prepare_data()
        
        # 步骤2: 训练模型
        model = self._train_model(X_train, y_train, X_valid, y_valid)
        
        # 步骤3: 三大分析
        analysis_results = self._run_analyses(
            model, X_train, y_train, X_valid, y_valid, feature_info
        )
        
        # 步骤4: 综合报告
        self._print_summary(analysis_results)
        
        # 返回完整结果
        results = {
            'model': model,
            'convergence': analysis_results['convergence'],
            'importance': analysis_results['importance'],
            'temporal': analysis_results['temporal'],
            'output_dir': analysis_results.get('output_dir', self.config['plot_dir']),
            'data_info': {
                'n_train': len(X_train),
                'n_valid': len(X_valid),
                'n_features': len(feature_info['feature_names']),
            }
        }

        print("\n" + "="*80)
        print("【Pipeline分析完成】")
        print("="*80)
        print(f"所有图片已保存到: {results['output_dir']}")

        return results
    
    def _prepare_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Dict]:
        """
        数据准备流程
        
        Returns:
            X_train, y_train, X_valid, y_valid, feature_info
        """
        if self.config['verbose']:
            print("\n步骤1: 数据准备...")
        
        # 1.1 加载配置
        with open(self.paths['top50_factors'], 'r') as f:
            top50_factors = json.load(f)
        
        with open(self.paths['preprocessing_rules'], 'r') as f:
            preprocessing_rules = json.load(f)
        
        self.original_factors = top50_factors
        
        if self.config['verbose']:
            print(f"  ✅ 已加载50个稳定因子和预处理规则")
        
        # 1.2 加载原始数据
        df_raw = pd.read_csv(self.paths['train_data'])
        if self.config['verbose']:
            print(f"  ✅ 原始数据: {df_raw.shape}")
        
        # 1.3 创建lagged特征
        from .feature_engineer import FeatureEngineer  # 动态导入避免循环依赖
        
        # 创建lagged特征（需要在notebook中定义create_lagged_features函数）
        # 这里简化处理：假设数据已包含lagged列或通过shift创建
        self.lagged_features = [
            'lagged_forward_returns',
            'lagged_risk_free_rate',
            'lagged_market_forward_excess_returns'
        ]
        
        # 创建lagged特征
        for col in ['forward_returns', 'risk_free_rate', 'market_forward_excess_returns']:
            if col in df_raw.columns:
                df_raw[f'lagged_{col}'] = df_raw[col].shift(1)
        
        # 1.4 筛选基础特征
        base_features = top50_factors + self.lagged_features
        TARGET = self.global_config['TARGET']
        
        feature_cols = base_features + [TARGET, 'date_id']
        df_selected = df_raw[[c for c in feature_cols if c in df_raw.columns]].copy()
        
        # 1.5 应用预处理（保护lagged特征）
        df_selected = self._apply_preprocessing(df_selected, preprocessing_rules, TARGET)
        
        # 1.6 特征工程
        if self.global_config['USE_FEATURE_ENGINEERING']:
            if self.config['verbose']:
                print(f"  执行特征工程: {'精简版' if self.global_config['USE_SLIM_FEATURES'] else '完整版'}")
            
            fe = FeatureEngineer(target_col=TARGET, verbose=False)
            if self.global_config['USE_SLIM_FEATURES']:
                df_feat = fe.create_features_slim(df_selected)
            else:
                df_feat = fe.create_features(
                    df_selected,
                    feature_cols=base_features,
                    lag_periods=self.global_config['LAG_PERIODS'],
                    rolling_windows=self.global_config['ROLLING_WINDOWS']
                )
        else:
            df_feat = df_selected.copy()
        
        df_feat.dropna(subset=[TARGET], inplace=True)
        
        # 1.7 准备X, y
        feature_names = [c for c in df_feat.columns if c not in [TARGET, 'date_id']]
        X = df_feat[feature_names]
        y = df_feat[TARGET]
        
        self.feature_names = feature_names
        
        if self.config['verbose']:
            print(f"  ✅ 数据准备完成: {len(X)}样本 × {len(feature_names)}特征")
        
        # 1.8 数据切分
        X_train, y_train, X_valid, y_valid = self._split_data(X, y)
        
        feature_info = {
            'feature_names': feature_names,
            'lagged_features': self.lagged_features,
            'original_factors': self.original_factors,
        }
        
        return X_train, y_train, X_valid, y_valid, feature_info
    
    def _apply_preprocessing(self, df: pd.DataFrame, rules: Dict, TARGET: str) -> pd.DataFrame:
        """应用预处理规则（保护lagged特征）"""
        # 保存需要保护的列
        preserve_cols = {}
        if TARGET in df.columns:
            preserve_cols[TARGET] = df[TARGET].copy()
        if 'date_id' in df.columns:
            preserve_cols['date_id'] = df['date_id'].copy()
        for lagged_col in self.lagged_features:
            if lagged_col in df.columns:
                preserve_cols[lagged_col] = df[lagged_col].copy()
        
        # 应用IC翻转
        flip_rules = rules.get('flip_rules', [])
        for col in flip_rules:
            if col in df.columns and col not in self.lagged_features:
                df[col] = df[col] * (-1)
        
        # 应用变换
        from .feature_preprocessor import FeaturePreprocessor
        
        transform_rules = rules.get('transform_rules', {})
        for col, transform in transform_rules.items():
            if col not in df.columns or transform == 'none' or col in self.lagged_features:
                continue
            try:
                if transform == 'winsorize':
                    df[col] = FeaturePreprocessor.winsorize_3sigma(df[col])
                elif transform == 'log':
                    df[col] = FeaturePreprocessor.log_transform(df[col])
                elif transform == 'rank':
                    df[col] = FeaturePreprocessor.rank_transform(df[col])
            except:
                continue
        
        # 恢复保护的列
        for col, values in preserve_cols.items():
            df[col] = values
        
        return df
    
    def _split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """简单train/valid划分（往前移7天，避免使用最新数据）"""
        train_window_size = self.global_config['TRAIN_WINDOW_SIZE']
        validation_split = self.global_config['VALIDATION_SPLIT']
        time_offset = self.global_config.get('TIME_OFFSET', 7)  # 往前移动天数，默认7天

        # 往前移7天：去掉最后7天，然后取倒数第N天
        if len(X) > time_offset:
            X_shifted = X.iloc[:-time_offset] if time_offset > 0 else X
            y_shifted = y.iloc[:-time_offset] if time_offset > 0 else y
        else:
            X_shifted = X
            y_shifted = y

        # 取最后N天（已经往前移了7天）
        X_last = X_shifted.tail(train_window_size)
        y_last = y_shifted.tail(train_window_size)

        # 按比例切分
        split_idx = int(len(X_last) * (1 - validation_split))

        X_train = X_last.iloc[:split_idx]
        y_train = y_last.iloc[:split_idx]
        X_valid = X_last.iloc[split_idx:]
        y_valid = y_last.iloc[split_idx:]

        if self.config['verbose']:
            print(f"  ✅ 数据切分（往前移{time_offset}天）: {len(X_train)}训练 + {len(X_valid)}验证")
            print(f"     数据范围: 倒数第{len(X) - len(X_shifted)}到倒数第{len(X) - len(X_shifted) + train_window_size}天")

        return X_train, y_train, X_valid, y_valid
    
    def _prepare_sequence_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sequence_length: int = 60
    ):
        """
        将数据转换为LSTM序列格式

        Args:
            X: 特征数据
            y: 目标数据
            sequence_length: 序列长度

        Returns:
            X_seq: 3D numpy array (n_samples, sequence_length, n_features)
            y_seq: 1D numpy array
        """
        if len(X) < sequence_length:
            raise ValueError(f"数据长度({len(X)})小于序列长度({sequence_length})")

        X_seq_list = []
        y_seq_list = []

        for i in range(sequence_length, len(X)):
            X_seq_list.append(X.iloc[i-sequence_length:i].values)
            y_seq_list.append(y.iloc[i])

        X_seq = np.array(X_seq_list)
        y_seq = np.array(y_seq_list)

        return X_seq, y_seq

    def _train_lstm_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series
    ):
        """训练LSTM模型"""
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Cannot train LSTM.")

        if self.config['verbose']:
            print("\n步骤2: 训练LSTM模型...")

        # 获取超参数
        input_dim = self.hyperparams['input_dim']
        hidden_dim = self.hyperparams.get('hidden_dim', 32)
        num_layers = self.hyperparams.get('num_layers', 1)
        dropout = self.hyperparams.get('dropout', 0)
        sequence_length = self.hyperparams.get('sequence_length', 60)
        batch_size = self.hyperparams.get('batch_size', 32)
        learning_rate = self.hyperparams.get('learning_rate', 0.001)
        epochs = self.hyperparams.get('epochs', 50)

        # 标准化
        from sklearn.preprocessing import StandardScaler
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

        X_train_scaled = self.x_scaler.fit_transform(X_train)
        y_train_scaled = self.y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()

        X_valid_scaled = self.x_scaler.transform(X_valid)
        y_valid_scaled = self.y_scaler.transform(y_valid.values.reshape(-1, 1)).ravel()

        # 转换为DataFrame/Series（保持索引）
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        y_train_scaled = pd.Series(y_train_scaled, index=y_train.index)
        X_valid_scaled = pd.DataFrame(X_valid_scaled, columns=X_valid.columns, index=X_valid.index)
        y_valid_scaled = pd.Series(y_valid_scaled, index=y_valid.index)

        # 准备序列数据
        X_train_seq, y_train_seq = self._prepare_sequence_data(X_train_scaled, y_train_scaled, sequence_length)
        X_valid_seq, y_valid_seq = self._prepare_sequence_data(X_valid_scaled, y_valid_scaled, sequence_length)

        if self.config['verbose']:
            print(f"  训练序列: {X_train_seq.shape}, 验证序列: {X_valid_seq.shape}")

        # 创建DataLoader
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_seq),
            torch.FloatTensor(y_train_seq)
        )
        valid_dataset = TensorDataset(
            torch.FloatTensor(X_valid_seq),
            torch.FloatTensor(y_valid_seq)
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        # 创建模型
        model = LSTMPredictor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # 训练循环
        model.train_losses = []
        model.valid_losses = []
        model.train_scores = []
        model.valid_scores = []
        model.best_epoch = 0
        model.best_valid_score = -np.inf

        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # 验证阶段
            model.eval()
            valid_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in valid_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    outputs = model(X_batch).squeeze()
                    loss = criterion(outputs, y_batch)
                    valid_loss += loss.item()

            valid_loss /= len(valid_loader)

            # 计算Score（需要反标准化）
            with torch.no_grad():
                train_pred_scaled = model(torch.FloatTensor(X_train_seq).to(self.device)).cpu().numpy().squeeze()
                valid_pred_scaled = model(torch.FloatTensor(X_valid_seq).to(self.device)).cpu().numpy().squeeze()

            train_pred = self.y_scaler.inverse_transform(train_pred_scaled.reshape(-1, 1)).ravel()
            valid_pred = self.y_scaler.inverse_transform(valid_pred_scaled.reshape(-1, 1)).ravel()

            train_true = y_train.iloc[sequence_length:].values
            valid_true = y_valid.iloc[sequence_length:].values

            train_score = calculate_score_metric(train_pred, train_true)
            valid_score = calculate_score_metric(valid_pred, valid_true)

            model.train_losses.append(train_loss)
            model.valid_losses.append(valid_loss)
            model.train_scores.append(train_score)
            model.valid_scores.append(valid_score)

            if valid_score > model.best_valid_score:
                model.best_valid_score = valid_score
                model.best_epoch = epoch

            if self.config['verbose'] and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, Valid Loss={valid_loss:.4f}, "
                      f"Train Score={train_score:.4f}, Valid Score={valid_score:.4f}")

        if self.config['verbose']:
            print(f"  ✅ 训练完成（best_epoch={model.best_epoch}, best_score={model.best_valid_score:.4f}）")

        return model

    def _train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series
    ):
        """训练模型（支持LightGBM和LSTM）"""
        if self.model_type == 'lstm':
            return self._train_lstm_model(X_train, y_train, X_valid, y_valid)

        # 原有的LightGBM训练逻辑
        if self.config['verbose']:
            print("\n步骤2: 训练模型...")

        from lightgbm import LGBMRegressor, early_stopping, log_evaluation

        train_params = self.hyperparams.copy()
        train_params['verbosity'] = -1

        model = LGBMRegressor(**train_params)

        eval_set = [(X_train, y_train), (X_valid, y_valid)]
        eval_names = ['train', 'valid']

        # Model Study中不使用early stopping，让模型完整训练
        # 这样可以观察完整的收敛曲线，即使验证集很小
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_names=eval_names,
            callbacks=[
                log_evaluation(period=0)  # 不输出日志
            ]
        )

        if self.config['verbose']:
            print(f"  ✅ 训练完成（best_iteration={model.best_iteration_}）")

        return model
    
    def _run_analyses(
        self,
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        feature_info: Dict
    ) -> Dict:
        """运行三大分析"""
        if self.config['verbose']:
            print("\n步骤3: 三大分析...")

        # 创建带时间戳的子目录
        import datetime
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        # 在原始plot_dir下创建子目录
        base_plot_dir = self.config['plot_dir']
        timestamped_plot_dir = os.path.join(base_plot_dir, timestamp)
        os.makedirs(timestamped_plot_dir, exist_ok=True)

        if self.config['verbose']:
            print(f"  ✅ 创建输出目录: {timestamped_plot_dir}")

        # 创建临时config，使用带时间戳的目录
        temp_config = self.config.copy()
        temp_config['plot_dir'] = timestamped_plot_dir

        results = {}

        # 分析1: 收敛分析
        convergence_analyzer = ConvergenceAnalyzer(model_type=self.model_type)
        results['convergence'] = convergence_analyzer.analyze(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            config=temp_config,
            hyperparams=self.hyperparams
        )
        
        # 分析2: 特征重要性
        importance_analyzer = FeatureImportanceAnalyzer(importance_type='gain')
        results['importance'] = importance_analyzer.analyze(
            model=model,
            X_train=X_train,
            y_train=y_train,
            feature_names=feature_info['feature_names'],
            lagged_features=feature_info['lagged_features'],
            original_factors=feature_info['original_factors'],
            config=temp_config
        )

        # 分析3: 时序稳定性
        X_full = pd.concat([X_train, X_valid], axis=0)
        y_full = pd.concat([y_train, y_valid], axis=0)

        temporal_analyzer = TemporalStabilityAnalyzer()
        results['temporal'] = temporal_analyzer.analyze(
            model=model,
            X=X_full,
            y=y_full,
            config=temp_config
        )

        # 保存时间戳目录信息到结果中
        results['output_dir'] = timestamped_plot_dir

        return results
    
    def _print_summary(self, analysis_results: Dict):
        """打印综合报告"""
        print("\n" + "="*80)
        print("【综合诊断报告】")
        print("="*80)
        
        conv = analysis_results['convergence']
        imp = analysis_results['importance']
        temp = analysis_results['temporal']
        
        print(f"\n📊 模型性能:")
        print(f"   - 最佳迭代数: {conv['best_iteration']}")
        print(f"   - 最佳Valid Score: {conv['best_valid_score']:.6f}")
        print(f"   - 过拟合程度: {conv['overfitting_gap']:.1f}%")
        print(f"   - 收敛速度: {conv['convergence_speed']:.1f}%")
        
        print(f"\n📊 特征质量:")
        top_3_features = imp['top_features'][:3]
        for rank, (feat, importance) in enumerate(top_3_features, 1):
            print(f"   {rank}. {feat:40s} {importance:10.2f}")
        print(f"   - 特征/样本比: {imp['feature_sample_ratio']:.4f}")
        
        print(f"\n📊 时序稳定性:")
        print(f"   - IC趋势: {temp['ic_trend']}")
        print(f"   - IC衰减率: {temp['ic_decay_rate']:.2f}%/window")
        print(f"   - 平均IC: {np.mean(temp['ic_series']):.4f}")

    def run_rolling_window_experiment(
        self,
        time_offsets: List[int],
        train_window_size: int = 200,
        validation_split: float = 0.035,
        base_plot_dir: str = './model_study_plots/LGB/',
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        滚动窗口实验：训练多个模型，测试不同时间偏移对性能的影响

        Args:
            time_offsets: 时间偏移列表，例如 [7, 14, 21, ..., 98]
            train_window_size: 训练窗口大小（固定）
            validation_split: 验证集比例
            base_plot_dir: 图表保存基础目录
            verbose: 是否显示详细输出

        Returns:
            results_df: 包含所有窗口结果的DataFrame

        使用示例:
            >>> pipeline = ModelStudyPipeline(config=config, hyperparams=hyperparams)
            >>> results_df = pipeline.run_rolling_window_experiment(
            ...     time_offsets=list(range(7, 99, 7)),
            ...     train_window_size=200
            ... )
        """
        import time

        print("\n" + "="*80)
        print("【滚动窗口实验】多时间窗口对比分析")
        print("="*80)
        print(f"\n实验配置:")
        print(f"  窗口大小: {train_window_size}天（固定）")
        print(f"  时间偏移: {time_offsets[0]}-{time_offsets[-1]}天")
        print(f"  窗口数量: {len(time_offsets)}个")
        print(f"  验证集比例: {validation_split}")

        # 存储所有窗口的结果
        all_results = []

        # 备份原始配置
        original_config = self.config.config.copy()
        original_global_config = self.global_config.copy()

        # 循环训练多个模型
        for idx, time_offset in enumerate(time_offsets, 1):

            print(f"\n" + "="*80)
            print(f"【窗口 {idx}/{len(time_offsets)}】Time Offset = {time_offset}天")
            print("="*80)

            start_time = time.time()

            try:
                # 更新global config
                self.global_config.update({
                    'TRAIN_WINDOW_SIZE': train_window_size,
                    'VALIDATION_SPLIT': validation_split,
                    'TIME_OFFSET': time_offset,
                })

                # 更新plot_dir，每个窗口独立保存
                window_plot_dir = os.path.join(base_plot_dir, f'offset_{time_offset:02d}/')
                self.config.config['plot_dir'] = window_plot_dir
                self.config.config['verbose'] = verbose

                # 运行分析
                result = self.run_full_analysis()

                # 提取关键指标
                convergence = result.get('convergence', {})
                temporal = result.get('temporal', {})

                window_result = {
                    'offset': time_offset,
                    'window_size': train_window_size,
                    'train_score': convergence.get('train_scores', [0])[-1] if 'train_scores' in convergence else 0,
                    'valid_score': convergence.get('best_valid_score', 0),
                    'best_iteration': convergence.get('best_iteration', 0),
                    'train_rmse': convergence.get('train_rmses', [0])[-1] if 'train_rmses' in convergence else 0,
                    'valid_rmse': convergence.get('valid_rmses', [0])[-1] if 'valid_rmses' in convergence else 0,
                    'overfitting_gap': convergence.get('overfitting_gap', 0),
                    'ic_mean': temporal.get('ic_mean', 0),
                    'ic_std': temporal.get('ic_std', 0),
                    'ic_decay': temporal.get('ic_decay_rate', 0),
                    'training_time': time.time() - start_time,
                }

                all_results.append(window_result)

                if verbose:
                    print(f"\n✅ 窗口{idx}完成:")
                    print(f"   Valid Score: {window_result['valid_score']:.6f}")
                    print(f"   Train Score: {window_result['train_score']:.6f}")
                    print(f"   IC Mean: {window_result['ic_mean']:.4f}")
                    print(f"   训练时间: {window_result['training_time']:.1f}秒")

            except Exception as e:
                print(f"\n❌ 窗口{idx}失败: {e}")
                import traceback
                traceback.print_exc()

                window_result = {
                    'offset': time_offset,
                    'window_size': train_window_size,
                    'valid_score': 0,
                    'train_score': 0,
                    'error': str(e)
                }
                all_results.append(window_result)

            finally:
                # 清理内存
                gc.collect()

        # 恢复原始配置
        self.config.config = original_config
        self.global_config = original_global_config

        # 转换为DataFrame
        results_df = pd.DataFrame(all_results)

        # 保存汇总结果
        summary_path = os.path.join(base_plot_dir, 'rolling_window_summary.csv')
        results_df.to_csv(summary_path, index=False)

        print("\n" + "="*80)
        print("【实验完成】所有窗口训练完毕")
        print("="*80)

        # 显示汇总表
        if verbose:
            print("\n汇总结果（按Valid Score排序）:")
            display_cols = ['offset', 'valid_score', 'train_score', 'overfitting_gap', 'ic_mean', 'training_time']
            print(results_df[display_cols].sort_values('valid_score', ascending=False).to_string(index=False))

        # 找出最优窗口
        best_window = results_df.loc[results_df['valid_score'].idxmax()]
        print(f"\n🏆 最优配置:")
        print(f"   Time Offset: {int(best_window['offset'])}天")
        print(f"   Valid Score: {best_window['valid_score']:.6f}")
        print(f"   Train Score: {best_window['train_score']:.6f}")
        print(f"   IC Mean: {best_window['ic_mean']:.4f}")

        print(f"\n✅ 结果已保存: {summary_path}")
        print("="*80)

        return results_df

    @staticmethod
    def visualize_rolling_window_results(
        summary_csv_path: str = './model_study_plots/LGB/rolling_window_summary.csv',
        output_dir: str = './model_study_plots/LGB/window_comparison/',
        show_plots: bool = True
    ):
        """
        可视化滚动窗口实验结果

        Args:
            summary_csv_path: 汇总CSV文件路径
            output_dir: 输出图表目录
            show_plots: 是否显示图表

        使用示例:
            >>> ModelStudyPipeline.visualize_rolling_window_results()
        """
        if not os.path.exists(summary_csv_path):
            print(f"❌ 找不到汇总文件: {summary_csv_path}")
            print("   请先运行 run_rolling_window_experiment()")
            return

        df = pd.read_csv(summary_csv_path)
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "="*80)
        print("【滚动窗口对比可视化】生成对比图表")
        print("="*80)
        print(f"\n加载 {len(df)} 个窗口的实验结果")

        # 最优窗口
        best_idx = df['valid_score'].idxmax()
        best_offset = df.loc[best_idx, 'offset']
        best_score = df.loc[best_idx, 'valid_score']

        # 图1: Valid Score趋势
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(df['offset'], df['valid_score'], marker='o', linewidth=2.5,
                markersize=8, color='#d62728', label='Valid Score', alpha=0.9)
        ax.fill_between(df['offset'], 0, df['valid_score'], alpha=0.2, color='#d62728')
        ax.scatter([best_offset], [best_score], s=300, color='gold', edgecolor='black',
                   linewidth=2, zorder=5, label=f'Best: Offset={best_offset}, Score={best_score:.4f}')
        ax.set_xlabel('Time Offset (Days)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Valid Score', fontsize=13, fontweight='bold')
        ax.set_title('Valid Score Across Different Time Offsets', fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '1_valid_score_trend.png'), dpi=150, bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close()

        # 图2: Train vs Valid
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(df['offset'], df['train_score'], marker='s', linewidth=2,
                markersize=7, color='#9467bd', label='Train Score', alpha=0.8)
        ax.plot(df['offset'], df['valid_score'], marker='o', linewidth=2.5,
                markersize=8, color='#d62728', label='Valid Score', alpha=0.9)
        ax.axvline(best_offset, color='gold', linestyle='--', linewidth=2, alpha=0.6)
        ax.set_xlabel('Time Offset (Days)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Score', fontsize=13, fontweight='bold')
        ax.set_title('Train vs Valid Score Comparison', fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '2_train_vs_valid.png'), dpi=150, bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close()

        # 图3: IC分析
        fig, ax1 = plt.subplots(figsize=(14, 6))
        ax1.plot(df['offset'], df['ic_mean'], marker='o', linewidth=2.5,
                 markersize=8, color='#1f77b4', label='IC Mean', alpha=0.9)
        ax1.set_xlabel('Time Offset (Days)', fontsize=13, fontweight='bold')
        ax1.set_ylabel('IC Mean', fontsize=13, fontweight='bold', color='#1f77b4')
        ax1.tick_params(axis='y', labelcolor='#1f77b4')
        ax1.grid(True, alpha=0.3)
        ax2 = ax1.twinx()
        ax2.plot(df['offset'], df['ic_std'], marker='s', linewidth=2,
                 markersize=7, color='#ff7f0e', label='IC Std', alpha=0.8, linestyle='--')
        ax2.set_ylabel('IC Std', fontsize=13, fontweight='bold', color='#ff7f0e')
        ax2.tick_params(axis='y', labelcolor='#ff7f0e')
        ax1.axvline(best_offset, color='gold', linestyle='--', linewidth=2, alpha=0.6)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=11)
        fig.suptitle('Information Coefficient (IC) Analysis', fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '3_ic_analysis.png'), dpi=150, bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close()

        print(f"\n✅ 图表已保存至: {output_dir}")
        print(f"   - 1_valid_score_trend.png")
        print(f"   - 2_train_vs_valid.png")
        print(f"   - 3_ic_analysis.png")
        print("="*80)
