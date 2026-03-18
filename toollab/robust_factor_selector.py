"""
鲁棒因子选择器
用于选择时间稳定的因子
"""

import numpy as np
import pandas as pd
from .feature_preprocessor import FeaturePreprocessor
from .factor_ic_analyzer import FactorICAnalyzer


class RobustFactorSelector:
    """
    鲁棒因子选择器 - 选择时间稳定的因子
    注意：本类仅供参考，实际使用预计算的结果
    """

    def __init__(self, n_windows=20, window_size=500, target_col='market_forward_excess_returns',
                 d_factor_prefix='d', keep_all_d_factors=True, use_preprocessing=True,
                 skip_transform_for_d=True):
        """
        Args:
            n_windows: 分成多少个时间窗口
            window_size: 每个窗口的大小
            target_col: 目标变量列名
            d_factor_prefix: D类因子前缀（列名以此开头）
            keep_all_d_factors: 是否保留全部D类因子
            use_preprocessing: 是否启用预处理（IC翻转 + 三轮变换）
            skip_transform_for_d: D类因子是否跳过Log/Rank变换
        """
        self.n_windows = n_windows
        self.window_size = window_size
        self.target_col = target_col
        self.d_factor_prefix = d_factor_prefix
        self.keep_all_d_factors = keep_all_d_factors
        self.use_preprocessing = use_preprocessing
        self.skip_transform_for_d = skip_transform_for_d
        self.analyzer = FactorICAnalyzer(window_size=20)

        # 预处理规则（将在preprocess_factors中填充）
        self.preprocessing_rules = {}

    def _is_dummy_factor(self, df, feature_name):
        """判断因子是否为D类因子（dummy variable）"""
        if not isinstance(feature_name, str):
            return False

        if feature_name.startswith(self.d_factor_prefix):
            return True

        if feature_name in df.columns:
            unique_values = set(df[feature_name].dropna().unique())
            if unique_values.issubset({0, 1, 0.0, 1.0}):
                return True

        return False

    def _classify_factors(self, df, feature_cols):
        """将因子分为D类和非D类"""
        d_factors = []
        non_d_factors = []

        for feat in feature_cols:
            if self._is_dummy_factor(df, feat):
                d_factors.append(feat)
            else:
                non_d_factors.append(feat)

        print(f"\n【因子分类】")
        print(f"  D类因子: {len(d_factors)} 个")
        print(f"  非D类因子: {len(non_d_factors)} 个")
        if len(d_factors) > 0:
            print(f"  D类示例: {d_factors[:5]}")

        return d_factors, non_d_factors

    def preprocess_factors(self, df, feature_cols, d_factors):
        """
        因子预处理：IC翻转 + 三轮变换优化

        Returns:
            df_processed: 预处理后的数据
            preprocessing_rules: 预处理规则字典
        """
        print("\n" + "="*80)
        print("【因子预处理】IC翻转 + 三轮变换")
        print("="*80)

        df_processed = df.copy()
        target = df[self.target_col].values

        # 步骤1: 初步IC分析（识别负IC）
        print("\n步骤1: 初步IC分析...")
        initial_ic = {}
        for col in feature_cols:
            try:
                metrics = self.analyzer.calculate_single_factor_metrics(
                    df[col].values, target
                )
                initial_ic[col] = metrics['IC']
            except:
                initial_ic[col] = 0.0

        # 步骤2: IC翻转
        print("\n步骤2: IC负值翻转...")
        flip_rules = [col for col, ic in initial_ic.items() if ic < 0]
        print(f"  发现 {len(flip_rules)} 个负IC因子需要翻转")

        for col in flip_rules:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col] * (-1)

        print(f"  ✅ 已翻转 {len(flip_rules)} 个因子")

        # 步骤3: 三轮变换优化
        print("\n步骤3: 三轮变换优化...")
        transform_rules = {}

        transforms = {
            'winsorize': FeaturePreprocessor.winsorize_3sigma,
            'log': FeaturePreprocessor.log_transform,
            'rank': FeaturePreprocessor.rank_transform
        }

        improved_count = 0
        for idx, col in enumerate(feature_cols, 1):
            if idx % 20 == 0:
                print(f"  已完成 {idx}/{len(feature_cols)} 个因子...")

            if col not in df_processed.columns:
                continue

            # 获取当前IC（可能已翻转）
            try:
                current_metrics = self.analyzer.calculate_single_factor_metrics(
                    df_processed[col].values, target
                )
                best_fitness = abs(current_metrics['Fitness'])
                best_transform = 'none'
                best_series = df_processed[col].copy()

                # D类因子的特殊处理
                if self.skip_transform_for_d and col in d_factors:
                    # D类因子只尝试3-Sigma
                    try_transforms = {'winsorize': transforms['winsorize']}
                else:
                    try_transforms = transforms

                # 尝试各种变换
                for name, func in try_transforms.items():
                    try:
                        transformed = func(df_processed[col])
                        new_metrics = self.analyzer.calculate_single_factor_metrics(
                            transformed.values, target
                        )
                        new_fitness = abs(new_metrics['Fitness'])

                        if new_fitness > best_fitness:
                            best_fitness = new_fitness
                            best_transform = name
                            best_series = transformed
                            improved_count += 1
                    except:
                        continue

                # 应用最优变换
                if best_transform != 'none':
                    df_processed[col] = best_series

                transform_rules[col] = best_transform
            except:
                transform_rules[col] = 'none'

        print(f"\n  ✅ 三轮变换完成")
        print(f"     改进因子数: {improved_count}/{len(feature_cols)}")

        # 整理规则
        preprocessing_rules = {
            'flip_rules': flip_rules,
            'transform_rules': transform_rules
        }

        print("\n" + "="*80)
        print("【因子预处理完成】")
        print(f"  翻转因子: {len(flip_rules)} 个")
        print(f"  优化因子: {improved_count} 个")
        print("="*80)

        return df_processed, preprocessing_rules

    def calculate_rolling_ic(self, df, feature_cols):
        """计算滚动窗口IC"""
        n_samples = len(df)
        step = max(1, (n_samples - self.window_size) // (self.n_windows - 1))

        ic_results = []

        for i in range(self.n_windows):
            start = i * step
            end = start + self.window_size

            if end > n_samples:
                end = n_samples
                start = max(0, end - self.window_size)

            window_data = df.iloc[start:end]

            print(f"Window {i+1}/{self.n_windows}: [{start}:{end}]")

            window_ics = {}
            for feat in feature_cols:
                if feat not in window_data.columns:
                    continue

                try:
                    metrics = self.analyzer.calculate_single_factor_metrics(
                        window_data[feat].values,
                        window_data[self.target_col].values
                    )

                    window_ics[feat] = {
                        'IC': metrics['IC'],
                        'RIC': metrics['RIC'],
                        'Fitness': metrics['Fitness']
                    }
                except Exception as e:
                    print(f"  Error processing {feat}: {e}")
                    window_ics[feat] = {'IC': 0.0, 'RIC': 0.0, 'Fitness': 0.0}

            ic_results.append(pd.DataFrame(window_ics).T)

        return ic_results

    def calculate_stability_metrics(self, ic_results, metric='IC'):
        """计算每个特征的时间稳定性指标（使用abs(IC)）"""
        stability_metrics = []

        all_features = ic_results[0].index.tolist()

        n_windows = len(ic_results)
        time_weights = np.array([0.90 ** (n_windows - i - 1) for i in range(n_windows)])
        time_weights = time_weights / time_weights.sum()

        for feat in all_features:
            ic_series = [ic_df.loc[feat, metric] if feat in ic_df.index else 0.0
                        for ic_df in ic_results]

            # 基础统计
            ic_mean = np.mean(ic_series)
            ic_std = np.std(ic_series)
            ic_sharpe = ic_mean / (ic_std + 1e-8)
            positive_ratio = np.mean([ic > 0 for ic in ic_series])
            ic_min = np.min(ic_series)
            ic_recent = ic_series[-1]

            # 加权IC均值
            weighted_ic = np.average(ic_series, weights=time_weights)
            # 加权IC绝对值均值（用于排序）
            weighted_ic_abs = np.average([abs(ic) for ic in ic_series], weights=time_weights)

            # 最近窗口惩罚
            recent_penalty = 1.0 if abs(ic_recent) > 0.01 else 0.3

            # 综合稳定性得分（使用绝对值）
            stability_score = (
                weighted_ic_abs * 0.4 +     # 加权IC绝对值（最重要）
                abs(ic_sharpe) * 0.2 +      # 稳定性
                positive_ratio * 0.2 +      # 正IC占比
                abs(ic_min) * 0.1 +         # 最差情况（绝对值）
                abs(ic_recent) * 0.1        # 最近表现（绝对值）
            ) * recent_penalty

            stability_metrics.append({
                'feature': feat,
                'mean': ic_mean,
                'std': ic_std,
                'sharpe': ic_sharpe,
                'positive_ratio': positive_ratio,
                'min': ic_min,
                'recent': ic_recent,
                'weighted_mean': weighted_ic,
                'weighted_mean_abs': weighted_ic_abs,
                'stability_score': stability_score
            })

        return pd.DataFrame(stability_metrics).sort_values('stability_score', ascending=False)

    def select_diverse_factors(self, df, candidate_features, n_select=50, max_corr=0.8):
        """选择多样化的因子（降低相关性）"""
        print(f"\n选择多样化因子 (目标: {n_select}个)...")

        try:
            corr_matrix = df[candidate_features].corr()
        except Exception as e:
            print(f"无法计算相关性矩阵: {e}")
            return candidate_features[:n_select]

        selected = []
        remaining = candidate_features.copy()

        while len(selected) < n_select and len(remaining) > 0:
            if len(selected) == 0:
                selected.append(remaining[0])
                remaining.remove(remaining[0])
            else:
                best_candidate = None
                best_score = -float('inf')

                for candidate in remaining:
                    try:
                        max_corr_with_selected = max([
                            abs(corr_matrix.loc[candidate, sel])
                            for sel in selected
                        ])

                        if max_corr_with_selected > max_corr:
                            continue

                        avg_corr = np.mean([
                            abs(corr_matrix.loc[candidate, sel])
                            for sel in selected
                        ])

                        candidate_idx = len(candidate_features) - candidate_features.index(candidate)
                        diversity_score = candidate_idx / (1 + avg_corr * 10)

                        if diversity_score > best_score:
                            best_score = diversity_score
                            best_candidate = candidate
                    except:
                        continue

                if best_candidate is None:
                    best_candidate = remaining[0]

                selected.append(best_candidate)
                remaining.remove(best_candidate)

        return selected

    def select_robust_factors(self, df, feature_cols, n_select=50, use_diversity=True):
        """
        完整流程：选择鲁棒因子（支持D类保护 + 预处理）

        Returns:
            selected_features: 选中的特征列表
            stability_df: 稳定性分析结果
            ic_results: 滚动窗口IC结果
            preprocessing_rules: 预处理规则
        """
        print("="*80)
        print("鲁棒因子筛选")
        print("="*80)

        # 步骤0: 因子分类
        d_factors, non_d_factors = self._classify_factors(df, feature_cols)

        # 步骤0.5: 因子预处理（新增）
        if self.use_preprocessing:
            df_processed, preprocessing_rules = self.preprocess_factors(df, feature_cols, d_factors)
            self.preprocessing_rules = preprocessing_rules
        else:
            df_processed = df
            preprocessing_rules = {'flip_rules': [], 'transform_rules': {}}

        # 步骤1: 计算滚动窗口IC
        print(f"\n步骤1: 计算滚动窗口IC (n_windows={self.n_windows}, window_size={self.window_size})...")
        ic_results = self.calculate_rolling_ic(df_processed, feature_cols)

        # 步骤2: 计算稳定性指标
        print(f"\n步骤2: 计算稳定性指标...")
        stability_df = self.calculate_stability_metrics(ic_results, metric='IC')

        print(f"\n稳定性Top 10:")
        print(stability_df.head(10)[['feature', 'stability_score', 'weighted_mean_abs', 'sharpe', 'recent']].to_string())

        # 步骤3: 双轨选择
        if self.keep_all_d_factors and len(d_factors) > 0:
            print(f"\n步骤3: 双轨选择（D类全保留模式）")

            selected_d_factors = d_factors.copy()
            print(f"  ✅ D类因子全部保留: {len(selected_d_factors)} 个")

            n_non_d = n_select - len(selected_d_factors)

            if n_non_d <= 0:
                print(f"  ⚠️ 警告: D类因子数({len(selected_d_factors)}) >= 目标数({n_select})")
                selected_features = selected_d_factors
            else:
                print(f"  从非D类因子中选择: {n_non_d} 个")

                non_d_stability = stability_df[stability_df['feature'].isin(non_d_factors)]

                if use_diversity:
                    candidate_features = non_d_stability['feature'].tolist()[:min(n_non_d*3, len(non_d_stability))]
                    selected_non_d = self.select_diverse_factors(
                        df_processed, candidate_features, n_select=n_non_d, max_corr=0.8
                    )
                else:
                    selected_non_d = non_d_stability.head(n_non_d)['feature'].tolist()

                print(f"  ✅ 非D类因子选择完成: {len(selected_non_d)} 个")

                selected_features = selected_d_factors + selected_non_d

        else:
            print(f"\n步骤3: 传统选择（无D类保护）")

            if use_diversity:
                candidate_features = stability_df['feature'].tolist()[:min(n_select*3, len(stability_df))]
                selected_features = self.select_diverse_factors(
                    df_processed, candidate_features, n_select=n_select, max_corr=0.8
                )
            else:
                selected_features = stability_df.head(n_select)['feature'].tolist()

        print(f"\n最终选择的{len(selected_features)}个因子:")

        selected_d = [f for f in selected_features if f in d_factors]
        selected_non_d = [f for f in selected_features if f in non_d_factors]
        print(f"  D类: {len(selected_d)} 个, 非D类: {len(selected_non_d)} 个")

        selected_info = stability_df[stability_df['feature'].isin(selected_features)]
        print(selected_info[['feature', 'stability_score', 'weighted_mean_abs', 'recent']].head(20).to_string())

        return selected_features, stability_df, ic_results, preprocessing_rules
