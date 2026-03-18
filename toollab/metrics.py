"""
评分指标函数
包含竞赛官方评分函数和其他辅助指标
"""

import numpy as np


def calculate_score_metric(y_pred, y_true, risk_free_rate=0.0, use_percentile=True,
                          low_threshold=0.3, high_threshold=0.7):
    """
    竞赛官方评分函数 - 调整后的夏普比率

    Args:
        y_pred: 预测的超额收益（连续值）
        y_true: 真实的超额收益
        risk_free_rate: 无风险利率（默认0）
        use_percentile: 是否使用分位数方法（默认True）
        low_threshold: 空仓阈值（默认0.3，即30分位以下）
        high_threshold: 满仓阈值（默认0.7，即70分位以上）

    Returns:
        adjusted_sharpe: 调整后的夏普比率（越高越好）

    Notes:
        - 仓位转换方法1（固定阈值）: pred <= 0 → 0仓, 0 < pred <= 0.001 → 1仓, pred > 0.001 → 2仓
        - 仓位转换方法2（分位数）: 基于预测值在历史中的分位数决定仓位
        - 包含波动率惩罚（超过市场1.2倍）和收益惩罚（低于市场）
    """
    if len(y_pred) == 0 or len(y_true) == 0:
        return 0.0

    # 1. 仓位转换
    if use_percentile and len(y_pred) > 1:
        # 方法2：基于分位数的自适应仓位
        position = np.zeros(len(y_pred))
        for i in range(len(y_pred)):
            # 计算当前预测值在所有预测值中的分位数
            percentile = (y_pred < y_pred[i]).sum() / len(y_pred)

            if percentile >= high_threshold:
                position[i] = 2.0  # 满仓
            elif percentile <= low_threshold:
                position[i] = 0.0  # 空仓
            else:
                position[i] = 1.0  # 1仓
    else:
        # 方法1：固定阈值（回退方案）
        position = np.where(y_pred > 0.001, 2.0, np.where(y_pred > 0, 1.0, 0.0))

    # 2. 反推forward_returns
    forward_returns = y_true + risk_free_rate

    # 3. 策略收益
    strategy_returns = risk_free_rate * (1 - position) + position * forward_returns

    # 4. 策略超额收益
    strategy_excess_returns = strategy_returns - risk_free_rate
    strategy_excess_cumulative = (1 + strategy_excess_returns).prod()

    if strategy_excess_cumulative <= 0:
        return 0.0

    strategy_mean_excess_return = (strategy_excess_cumulative) ** (1 / len(y_pred)) - 1

    # 5. 策略标准差
    strategy_std = strategy_returns.std()

    if strategy_std == 0 or strategy_std < 1e-8:
        return 0.0

    # 6. 年化夏普比率
    trading_days_per_yr = 252
    sharpe = strategy_mean_excess_return / strategy_std * np.sqrt(trading_days_per_yr)

    # 7. 策略年化波动率
    strategy_volatility = float(strategy_std * np.sqrt(trading_days_per_yr) * 100)

    # 8. 市场基准指标
    market_excess_returns = forward_returns - risk_free_rate
    market_excess_cumulative = (1 + market_excess_returns).prod()

    if market_excess_cumulative <= 0:
        return sharpe

    market_mean_excess_return = (market_excess_cumulative) ** (1 / len(y_pred)) - 1
    market_std = forward_returns.std()

    if market_std == 0:
        return sharpe

    market_volatility = float(market_std * np.sqrt(trading_days_per_yr) * 100)

    # 9. 波动率惩罚
    excess_vol = max(0, strategy_volatility / market_volatility - 1.2) if market_volatility > 0 else 0
    vol_penalty = 1 + excess_vol

    # 10. 收益差距惩罚
    return_gap = max(0, (market_mean_excess_return - strategy_mean_excess_return) * 100 * trading_days_per_yr)
    return_penalty = 1 + (return_gap**2) / 100

    # 11. 调整后夏普比率
    adjusted_sharpe = sharpe / (vol_penalty * return_penalty)

    return min(float(adjusted_sharpe), 1_000_000)
