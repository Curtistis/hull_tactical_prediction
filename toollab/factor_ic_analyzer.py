"""
【因子IC分析器】Factor IC Analyzer
用于评估特征的预测能力
  - Fitness: 综合评分 = IC × RankIC × √(RICIR) × 10000
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


class FactorICAnalyzer:
    """因子IC分析器 - 评估特征预测能力"""

    def __init__(self, window_size=20):
        """
        Args:
            window_size: 滚动窗口大小（用于计算时间序列IC）
        """
        self.window_size = window_size

    def calculate_single_factor_metrics(self, factor_values, target_values):
        """
        计算单个因子的IC指标

        Args:
            factor_values: 因子值序列（numpy array）
            target_values: 目标变量序列（numpy array）

        Returns:
            dict: 包含IC、RIC、ICIR、RICIR、Fitness等指标
        """
        # 移除NaN值
        valid_mask = ~(np.isnan(factor_values) | np.isnan(target_values))
        factor_clean = factor_values[valid_mask]
        target_clean = target_values[valid_mask]

        if len(factor_clean) < self.window_size:
            return {
                'IC': 0.0, 'RIC': 0.0, 'ICIR': 0.0, 'RICIR': 0.0,
                'IC_mean': 0.0, 'RIC_mean': 0.0, 'Fitness': 0.0,
                'valid_samples': len(factor_clean)
            }

        # 1. 整体IC和RIC（Pearson和Spearman相关系数）
        try:
            ic_overall, _ = pearsonr(factor_clean, target_clean)
            ric_overall, _ = spearmanr(factor_clean, target_clean)
            # 检查是否为nan
            if np.isnan(ic_overall):
                ic_overall = 0.0
            if np.isnan(ric_overall):
                ric_overall = 0.0
        except:
            ic_overall, ric_overall = 0.0, 0.0

        # 2. 滚动窗口计算IC序列
        ic_series = []
        ric_series = []

        for i in range(len(factor_clean) - self.window_size + 1):
            window_factor = factor_clean[i:i+self.window_size]
            window_target = target_clean[i:i+self.window_size]

            try:
                ic_val, _ = pearsonr(window_factor, window_target)
                ric_val, _ = spearmanr(window_factor, window_target)
                ic_series.append(ic_val)
                ric_series.append(ric_val)
            except:
                continue

        ic_series = np.array(ic_series)
        ric_series = np.array(ric_series)

        # 3. 计算IR（Information Ratio）
        if len(ic_series) > 0 and np.std(ic_series) > 0:
            ic_mean = np.mean(ic_series)
            icir = ic_mean / np.std(ic_series)
        else:
            ic_mean, icir = 0.0, 0.0

        if len(ric_series) > 0 and np.std(ric_series) > 0:
            ric_mean = np.mean(ric_series)
            ricir = ric_mean / np.std(ric_series)
        else:
            ric_mean, ricir = 0.0, 0.0

        # 4. 计算综合Fitness评分
        # Fitness = IC × RankIC × √(RankICIR) × 10000
        if ricir > 0:
            fitness = ic_overall * ric_overall * np.sqrt(abs(ricir)) * 10000
        else:
            fitness = 0.0

        return {
            'IC': ic_overall, 'RIC': ric_overall,
            'ICIR': icir, 'RICIR': ricir,
            'IC_mean': ic_mean, 'RIC_mean': ric_mean,
            'Fitness': fitness, 'valid_samples': len(factor_clean)
        }

    def analyze_dataset(self, train_data, target_col='market_forward_excess_returns',
                       save_path=None, verbose=True, show_top_n=20, show_bottom_n=10,
                       recent_n_samples: int | None = None, recent_fraction: float | None = None):
        """
        分析数据集中所有特征的IC指标

        Args:
            train_data: 训练数据（polars或pandas DataFrame）
            target_col: 目标变量列名
            save_path: 如果提供，保存结果到此路径
            verbose: 是否打印详细统计信息
            show_top_n: 显示排名前N的因子
            show_bottom_n: 显示排名后N的因子
            recent_n_samples: 如果指定，仅使用最近N个样本进行分析
            recent_fraction: 如果指定，仅使用最近的一定比例样本（0-1之间）

        Returns:
            pd.DataFrame: 包含所有特征的IC分析结果
        """
        if verbose:
            print("\n开始评估所有特征的预测能力...")

        # 准备数据
        if hasattr(train_data, 'to_pandas'):
            train_pd = train_data.to_pandas()
        else:
            train_pd = train_data

        # 根据参数截取最近数据
        original_len = len(train_pd)
        if recent_n_samples is not None:
            train_pd = train_pd.tail(recent_n_samples)
            if verbose:
                print(f"使用最近 {recent_n_samples} 个样本进行分析 (原始样本数: {original_len})")
        elif recent_fraction is not None:
            n_samples = int(original_len * recent_fraction)
            train_pd = train_pd.tail(n_samples)
            if verbose:
                print(f"使用最近 {recent_fraction*100:.1f}% 样本进行分析 ({n_samples}/{original_len})")

        target = pd.to_numeric(train_pd[target_col], errors='coerce').values

        # 排除不应该作为因子的列（包含未来信息或非因子列）
        exclude_cols = {
            target_col,              # 目标列
            'date_id',               # 日期ID
            'forward_returns',       # 未来收益（数据泄漏！）
            'risk_free_rate'         # 无风险利率
        }
        feature_cols = [col for col in train_pd.columns if col not in exclude_cols]

        if verbose:
            print(f"总共{len(feature_cols)}个特征需要评估...")

        # 评估每个特征
        factor_results = []
        for idx, col in enumerate(feature_cols, 1):
            if verbose and idx % 20 == 0:
                print(f"  已完成 {idx}/{len(feature_cols)} 个特征...")

            factor_values = train_pd[col].values
            metrics = self.calculate_single_factor_metrics(factor_values, target)

            factor_results.append({
                '特征': col,
                '系列': col[0] if len(col) > 0 else '',
                'IC': metrics['IC'],
                'RIC': metrics['RIC'],
                'ICIR': metrics['ICIR'],
                'RICIR': metrics['RICIR'],
                'IC均值': metrics['IC_mean'],
                'RIC均值': metrics['RIC_mean'],
                'Fitness': metrics['Fitness'],
                '有效样本': metrics['valid_samples']
            })

        factor_df = pd.DataFrame(factor_results)
        factor_df = factor_df.sort_values('IC', ascending=False, key=lambda x: abs(x))

        if verbose:
            print(f"\n✅ 评估完成！")
            self._print_summary(factor_df, show_top_n, show_bottom_n)

        # 保存结果
        if save_path is not None:
            factor_df.to_csv(save_path, index=False)
            if verbose:
                print(f"\n✅ 因子评估结果已保存到: {save_path}")

        if verbose:
            print("\n" + "="*80)
            print("【单因子评估完成】")
            print("="*80)

        return factor_df

    def _print_summary(self, factor_df, show_top_n, show_bottom_n):
        """打印统计摘要"""
        print("\n" + "="*80)
        print(f"【Top {show_top_n} 最佳因子】")
        print("="*80)
        print(factor_df.head(show_top_n).to_string(index=False))

        print("\n" + "="*80)
        print(f"【Bottom {show_bottom_n} 最差因子】")
        print("="*80)
        print(factor_df.tail(show_bottom_n).to_string(index=False))

        # 按系列汇总
        print("\n" + "="*80)
        print("【按系列汇总】")
        print("="*80)
        series_summary = factor_df.groupby('系列').agg({
            'Fitness': ['mean', 'max', 'min'],
            'IC': 'mean', 'RIC': 'mean', 'ICIR': 'mean', 'RICIR': 'mean'
        }).round(4)
        print(series_summary)

        # 整体统计
        print("\n" + "="*80)
        print("【整体统计】")
        print("="*80)
        print(f"平均Fitness:        {factor_df['Fitness'].mean():.4f}")
        print(f"Fitness标准差:      {factor_df['Fitness'].std():.4f}")
        print(f"最大Fitness:        {factor_df['Fitness'].max():.4f} ({factor_df.iloc[0]['特征']})")
        print(f"最小Fitness:        {factor_df['Fitness'].min():.4f} ({factor_df.iloc[-1]['特征']})")
        print(f"\n平均IC:             {factor_df['IC'].mean():.4f}")
        print(f"平均RIC:            {factor_df['RIC'].mean():.4f}")
        print(f"平均ICIR:           {factor_df['ICIR'].mean():.4f}")
        print(f"平均RICIR:          {factor_df['RICIR'].mean():.4f}")

        # 筛选优质因子
        print("\n" + "="*80)
        print("【优质因子筛选建议】")
        print("="*80)
        fitness_threshold = factor_df['Fitness'].quantile(0.75)
        good_factors = factor_df[
            (factor_df['Fitness'] >= fitness_threshold) |
            (abs(factor_df['IC']) >= 0.02) |
            (abs(factor_df['RICIR']) >= 0.1)
        ]
        print(f"\n📊 推荐使用的因子 (Fitness ≥ {fitness_threshold:.4f} 或 |IC| ≥ 0.02 或 |RICIR| ≥ 0.1):")
        print(f"   共 {len(good_factors)} 个因子\n")
        print(good_factors[['特征', 'IC', 'RIC', 'ICIR', 'RICIR', 'Fitness']].to_string(index=False))

        print("\n💡 特征选择建议:")
        print("   1. 【优先使用】Top 20因子（Fitness最高）")
        print("   2. 【重点关注】|IC| > 0.03 且 |RICIR| > 0.15 的因子")
        print("   3. 【谨慎使用】Fitness < 0.01 的因子（预测能力弱）")
        print("   4. 【建议组合】从不同系列中选择Top因子，避免高度相关")

    def analyze_dataset_by_regime(self, train_data, regime_col='Regime',
                                  target_col='market_forward_excess_returns',
                                  save_dir=None, verbose=True, show_top_n=20, show_bottom_n=10):
        """
        按Regime拆分数据后分别做单因子IC分析

        Args:
            train_data: 训练数据（polars或pandas DataFrame）
            regime_col: Regime列名
            target_col: 目标变量列名
            save_dir: 如果提供，保存每个Regime的结果到此目录
            verbose: 是否打印详细统计信息
            show_top_n: 显示排名前N的因子
            show_bottom_n: 显示排名后N的因子

        Returns:
            dict[str, pd.DataFrame]: 每个Regime的IC分析结果字典
        """
        if verbose:
            print("\n" + "="*80)
            print("【开始按Regime分组进行因子IC分析】")
            print("="*80)

        # 准备数据
        if hasattr(train_data, 'to_pandas'):
            train_pd = train_data.to_pandas()
        else:
            train_pd = train_data

        # 检查regime列是否存在
        if regime_col not in train_pd.columns:
            raise ValueError(f"数据中不存在列: {regime_col}")

        # 获取所有Regime
        regimes = sorted(train_pd[regime_col].unique())

        if verbose:
            print(f"\n发现 {len(regimes)} 个Regime: {regimes}")
            for regime in regimes:
                regime_data = train_pd[train_pd[regime_col] == regime]
                print(f"  Regime {regime}: {len(regime_data)} 个样本")

        # 对每个Regime分别分析
        results_by_regime = {}

        for regime in regimes:
            if verbose:
                print("\n" + "="*80)
                print(f"【Regime {regime} 的因子IC分析】")
                print("="*80)

            # 筛选当前Regime的数据
            regime_data = train_pd[train_pd[regime_col] == regime]

            # 如果指定了保存目录，为每个Regime生成保存路径
            save_path = None
            if save_dir is not None:
                import os
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"ic_analysis_regime_{regime}.csv")

            # 对当前Regime执行IC分析
            regime_result = self.analyze_dataset(
                train_data=regime_data,
                target_col=target_col,
                save_path=save_path,
                verbose=verbose,
                show_top_n=show_top_n,
                show_bottom_n=show_bottom_n
            )

            # 添加Regime标识列
            regime_result.insert(0, 'Regime', regime)
            results_by_regime[str(regime)] = regime_result

        if verbose:
            print("\n" + "="*80)
            print("【按Regime的因子IC分析完成】")
            print("="*80)

            # 打印各Regime的整体对比
            print("\n【各Regime因子表现对比】")
            print("="*80)
            for regime, df in results_by_regime.items():
                print(f"\nRegime {regime}:")
                print(f"  平均Fitness: {df['Fitness'].mean():.4f}")
                print(f"  最大Fitness: {df['Fitness'].max():.4f} ({df.iloc[0]['特征']})")
                print(f"  平均|IC|:    {abs(df['IC']).mean():.4f}")
                print(f"  平均|RIC|:   {abs(df['RIC']).mean():.4f}")

        return results_by_regime
