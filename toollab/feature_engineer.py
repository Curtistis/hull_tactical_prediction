"""
【特征工程函数库】Feature Engineer
将原始特征扩展到衍生特征
支持完整版和精简版两种模式
"""

import numpy as np
import pandas as pd


class FeatureEngineer:
    """特征工程器 - 创建时间序列衍生特征"""

    def __init__(self, target_col='market_forward_excess_returns', verbose=True):
        """
        Args:
            target_col: 目标列名
            verbose: 是否打印详细信息
        """
        self.target_col = target_col
        self.verbose = verbose

    # ==================== 基础特征变换方法 ====================

    def add_lag_features(self, df: pd.DataFrame, cols, lags):
        """
        【功能】创建滞后特征（历史值特征）

        【参数】
            df (pd.DataFrame): 输入的DataFrame
            cols (list): 要处理的列名列表
            lags (list): 滞后期列表，例如 [1, 3, 5, 7, 14, 20]

        【返回】
            pd.DataFrame: 添加了滞后特征的DataFrame

        【原理】
            时间序列的自相关性：过去的值可以预测未来
            - df[col].shift(1): 向后移1位，得到"昨天"的值
            - df[col].shift(7): 向后移7位，得到"一周前"的值

        【生成数量】
            len(cols) × len(lags) 个新特征
        """
        for col in cols:
            if col not in df.columns:
                continue
            for lag in lags:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)
        return df

    def add_rolling_features(self, df: pd.DataFrame, cols, windows, funcs=['mean', 'std', 'min', 'max', 'median']):
        """
        【功能】创建滚动窗口统计特征

        【参数】
            df (pd.DataFrame): 输入的DataFrame
            cols (list): 要处理的列名列表
            windows (list): 窗口大小列表，例如 [3, 6, 10, 20, 60]
            funcs (list): 统计函数列表

        【返回】
            pd.DataFrame: 添加了滚动统计特征的DataFrame

        【原理】
            对每个窗口计算统计量：
            1. rolling mean: 移动平均，平滑噪声，显示趋势
            2. rolling std: 移动标准差，衡量波动率
            3. rolling min: 窗口内最小值，支撑位
            4. rolling max: 窗口内最大值，阻力位
            5. rolling median: 中位数，抗离群值
        """
        for col in cols:
            if col not in df.columns:
                continue
            for w in windows:
                for func in funcs:
                    df[f"{col}_roll_{func}_{w}"] = df[col].rolling(w, min_periods=1).agg(func)
        return df

    def add_pct_change_features(self, df: pd.DataFrame, cols, periods=[1, 3, 7]):
        """
        【功能】创建百分比变化特征（收益率/动量特征）

        【参数】
            df (pd.DataFrame): 输入的DataFrame
            cols (list): 要处理的列名列表
            periods (list): 变化周期，默认 [1, 3, 7]

        【返回】
            pd.DataFrame: 添加了百分比变化特征的DataFrame

        【公式】
            pct_change = (当前值 - 过去值) / 过去值
        """
        for col in cols:
            if col not in df.columns:
                continue
            for p in periods:
                df[f"{col}_pct_change_{p}"] = df[col].pct_change(periods=p)
        return df

    def add_diff_features(self, df: pd.DataFrame, cols, periods=[1, 7]):
        """
        【功能】创建一阶差分特征

        【参数】
            df (pd.DataFrame): 输入的DataFrame
            cols (list): 要处理的列名列表
            periods (list): 差分周期

        【公式】
            diff = 当前值 - 过去值
        """
        for col in cols:
            if col not in df.columns:
                continue
            for p in periods:
                df[f"{col}_diff_{p}"] = df[col].diff(p)
        return df

    def add_ewm_features(self, df: pd.DataFrame, cols, spans=[7, 14, 30]):
        """
        【功能】创建指数加权移动平均特征

        【参数】
            df (pd.DataFrame): 输入的DataFrame
            cols (list): 要处理的列名列表
            spans (list): EWM的span参数，默认 [7, 14, 30]

        【返回】
            pd.DataFrame: 添加了EWM特征的DataFrame

        【原理】
            与简单移动平均不同，EWM对近期数据赋予更大权重：
            - 近期数据: 权重大
            - 远期数据: 权重指数衰减
            - 比SMA更快响应趋势变化
        """
        for col in cols:
            if col not in df.columns:
                continue
            for s in spans:
                df[f"{col}_ewm_mean_{s}"] = df[col].ewm(span=s, adjust=False, min_periods=1).mean()
                df[f"{col}_ewm_std_{s}"] = df[col].ewm(span=s, adjust=False, min_periods=1).std()
        return df

    def add_interaction_features(self, df: pd.DataFrame, cols):
        """
        【功能】创建特征交互项（非线性组合）

        【参数】
            df (pd.DataFrame): 输入的DataFrame
            cols (list): 要处理的列名列表

        【返回】
            pd.DataFrame: 添加了交互特征的DataFrame

        【原理】
            单个特征可能线性无关，但组合后可能显示强关系：
            - 乘法交互: c1 × c2，捕捉协同效应
            - 除法交互: c1 / c2，捕捉相对强弱
        """
        numeric_cols = [c for c in cols if c in df.columns and df[c].dtype != "object"]

        for i, c1 in enumerate(numeric_cols):
            for c2 in numeric_cols[i+1:]:
                df[f"{c1}_x_{c2}"] = df[c1] * df[c2]
                df[f"{c1}_div_{c2}"] = df[c1] / (df[c2] + 1e-8)
        return df

    def add_rolling_normalized_features(self, df: pd.DataFrame, cols, windows=[14, 30]):
        """
        【功能】创建滚动标准化特征（Rolling Z-score）

        【公式】
            z-score = (当前值 - 滚动均值) / 滚动标准差

        【原理】
            将数据归一化到均值0、标准差1：
            - z > 2: 异常高，可能回归
            - z < -2: 异常低，可能反弹
            - |z| < 1: 正常范围
        """
        for col in cols:
            if col not in df.columns:
                continue

            for w in windows:
                m = df[col].rolling(w, min_periods=1).mean()
                s = df[col].rolling(w, min_periods=1).std()
                df[f"{col}_roll_zscore_{w}"] = (df[col] - m) / (s + 1e-8)
        return df

    # ==================== 主特征工程函数 ====================

    def create_features(self, df: pd.DataFrame, feature_cols,
                       lag_periods=[1, 3, 5, 7, 14, 20],
                       rolling_windows=[3, 6, 10, 20, 60]) -> pd.DataFrame:
        """
        【功能】完整版特征工程 - 生成所有衍生特征

        【参数】
            df (pd.DataFrame): 包含原始特征的DataFrame
            feature_cols (list): 要进行特征工程的列名列表
            lag_periods (list): 滞后期列表
            rolling_windows (list): 滚动窗口大小列表

        【返回】
            pd.DataFrame: 包含所有衍生特征的DataFrame

        【特征数量】
            输入: N个原始特征
            输出: ~70N个特征（包括原始特征）
        """
        if self.verbose:
            print(f"\n【完整特征工程】输入: {df.shape}")

        df = df.copy()

        # 应用各种变换
        if self.verbose:
            print("  创建Lag特征...")
        df = self.add_lag_features(df, feature_cols, lag_periods)

        if self.verbose:
            print("  创建Rolling特征...")
        df = self.add_rolling_features(df, feature_cols, rolling_windows)

        if self.verbose:
            print("  创建Pct_change特征...")
        df = self.add_pct_change_features(df, feature_cols)

        if self.verbose:
            print("  创建Diff特征...")
        df = self.add_diff_features(df, feature_cols)

        if self.verbose:
            print("  创建EWM特征...")
        df = self.add_ewm_features(df, feature_cols)

        if self.verbose:
            print("  创建交互特征...")
        df = self.add_interaction_features(df, feature_cols)

        if self.verbose:
            print("  创建Rolling Z-score特征...")
        df = self.add_rolling_normalized_features(df, feature_cols)

        # 缺失值处理
        if self.verbose:
            print("  处理缺失值...")
        df.ffill(inplace=True)
        df.bfill(inplace=True)

        for c in df.columns:
            if df[c].isnull().any():
                df[c].fillna(df[c].median(), inplace=True)

        if self.verbose:
            print(f"【完整特征工程】输出: {df.shape}")

        return df

    def add_market_state_features(self, df: pd.DataFrame, vol_windows=(20, 60), regime_col='vol_regime'):
        """
        【功能】基于目标列构造市场状态特征（波动率 + Regime 标签）

        【参数】
            df (pd.DataFrame): 输入的DataFrame，必须包含目标列
            vol_windows (tuple): 波动率计算窗口，默认 (20, 60)
            regime_col (str): 生成的Regime列名，默认 'vol_regime'

        【返回】
            pd.DataFrame: 添加了市场状态特征的DataFrame

        【生成特征】
            1. 波动率特征：
               - {target}_volatility_{window}: 滚动窗口标准差
               - {target}_volatility_ratio: 短期波动率 / 长期波动率

            2. Regime标签：
               - {regime_col}: 基于波动率的市场状态分类
                 * 0: 低波动 (volatility < 25th percentile)
                 * 1: 中波动 (25th <= volatility < 75th percentile)
                 * 2: 高波动 (volatility >= 75th percentile)

        【应用场景】
            用于自适应特征工程和模型调参，不同市场状态使用不同的特征/参数
        """
        if self.verbose:
            print(f"\n【市场状态特征工程】输入: {df.shape}")

        df = df.copy()

        # 检查目标列是否存在
        if self.target_col not in df.columns:
            raise ValueError(f"目标列 '{self.target_col}' 不存在于DataFrame中")

        short_window, long_window = vol_windows

        # 1. 计算波动率特征
        if self.verbose:
            print(f"  计算波动率特征 (窗口: {short_window}, {long_window})...")

        # 短期波动率
        df[f'{self.target_col}_volatility_{short_window}'] = \
            df[self.target_col].rolling(window=short_window, min_periods=1).std()

        # 长期波动率
        df[f'{self.target_col}_volatility_{long_window}'] = \
            df[self.target_col].rolling(window=long_window, min_periods=1).std()

        # 波动率比率（短期/长期）
        df[f'{self.target_col}_volatility_ratio'] = \
            df[f'{self.target_col}_volatility_{short_window}'] / \
            (df[f'{self.target_col}_volatility_{long_window}'] + 1e-8)

        # 2. 构造Regime标签（基于长期波动率的分位数）
        if self.verbose:
            print(f"  构造Regime标签...")

        vol_col = f'{self.target_col}_volatility_{long_window}'

        # 计算波动率的25%和75%分位数
        vol_25 = df[vol_col].quantile(0.25)
        vol_75 = df[vol_col].quantile(0.75)

        # 分配Regime标签
        df[regime_col] = 1  # 默认中波动
        df.loc[df[vol_col] < vol_25, regime_col] = 0  # 低波动
        df.loc[df[vol_col] >= vol_75, regime_col] = 2  # 高波动

        # 确保Regime是整数类型
        df[regime_col] = df[regime_col].astype(int)

        if self.verbose:
            regime_counts = df[regime_col].value_counts().sort_index()
            print(f"\n  Regime分布:")
            print(f"    Regime 0 (低波动): {regime_counts.get(0, 0)} 个样本")
            print(f"    Regime 1 (中波动): {regime_counts.get(1, 0)} 个样本")
            print(f"    Regime 2 (高波动): {regime_counts.get(2, 0)} 个样本")
            print(f"\n  波动率阈值:")
            print(f"    25%分位数: {vol_25:.6f}")
            print(f"    75%分位数: {vol_75:.6f}")

        if self.verbose:
            print(f"【市场状态特征工程】输出: {df.shape}")

        return df

    def create_features_dynamic(self, df: pd.DataFrame, feature_cols, regime_col='vol_regime'):
        """
        【功能】根据市场状态 Regime 自适应调整特征工程窗口长度

        【参数】
            df (pd.DataFrame): 输入的DataFrame，必须包含Regime列
            feature_cols (list): 要进行特征工程的列名列表
            regime_col (str): Regime列名，默认 'vol_regime'

        【返回】
            pd.DataFrame: 包含自适应特征的DataFrame

        【自适应策略】
            不同Regime使用不同的窗口参数：

            Regime 0 (低波动市场):
              - Lag periods: [1, 3, 5, 7, 14, 20, 30]  # 更长的滞后期
              - Rolling windows: [5, 10, 20, 40, 80]   # 更长的窗口
              - 逻辑：低波动时趋势稳定，使用更长周期捕捉长期信号

            Regime 1 (中波动市场):
              - Lag periods: [1, 3, 5, 7, 14, 20]      # 标准滞后期
              - Rolling windows: [3, 6, 10, 20, 60]    # 标准窗口
              - 逻辑：正常市场，使用标准参数

            Regime 2 (高波动市场):
              - Lag periods: [1, 2, 3, 5, 7]           # 更短的滞后期
              - Rolling windows: [3, 5, 7, 10, 20]     # 更短的窗口
              - 逻辑：高波动时变化快速，使用更短周期快速响应

        【特征命名】
            特征名包含Regime标识: {col}_lag_{lag}_r{regime}
            便于后续分析和特征选择
        """
        if self.verbose:
            print(f"\n【动态特征工程】输入: {df.shape}")

        # 检查Regime列是否存在
        if regime_col not in df.columns:
            raise ValueError(f"Regime列 '{regime_col}' 不存在于DataFrame中，请先运行 add_market_state_features()")

        df = df.copy()

        # 定义不同Regime下的参数
        regime_params = {
            0: {  # 低波动
                'lag_periods': [1, 3, 5, 7, 14, 20, 30],
                'rolling_windows': [5, 10, 20, 40, 80],
                'pct_periods': [1, 3, 7, 14],
                'ewm_spans': [10, 20, 40]
            },
            1: {  # 中波动
                'lag_periods': [1, 3, 5, 7, 14, 20],
                'rolling_windows': [3, 6, 10, 20, 60],
                'pct_periods': [1, 3, 7],
                'ewm_spans': [7, 14, 30]
            },
            2: {  # 高波动
                'lag_periods': [1, 2, 3, 5, 7],
                'rolling_windows': [3, 5, 7, 10, 20],
                'pct_periods': [1, 2, 3, 5],
                'ewm_spans': [3, 7, 14]
            }
        }

        # 获取所有Regime
        regimes = sorted(df[regime_col].unique())

        if self.verbose:
            print(f"  发现 {len(regimes)} 个Regime: {regimes}")
            for regime in regimes:
                regime_count = (df[regime_col] == regime).sum()
                print(f"    Regime {regime}: {regime_count} 个样本")

        # 为每个Regime创建特征
        regime_features = []

        for regime in regimes:
            if regime not in regime_params:
                if self.verbose:
                    print(f"  警告: Regime {regime} 没有预定义参数，跳过")
                continue

            if self.verbose:
                print(f"\n  处理 Regime {regime}...")

            # 获取当前Regime的索引
            regime_mask = df[regime_col] == regime
            regime_indices = df[regime_mask].index

            if len(regime_indices) == 0:
                continue

            # 获取参数
            params = regime_params[regime]
            lag_periods = params['lag_periods']
            rolling_windows = params['rolling_windows']
            pct_periods = params['pct_periods']
            ewm_spans = params['ewm_spans']

            # 创建临时DataFrame存储该Regime的特征
            regime_df = df.loc[regime_indices].copy()

            # 1. Lag特征
            if self.verbose:
                print(f"    创建Lag特征 (periods={lag_periods})...")
            for col in feature_cols:
                if col not in regime_df.columns:
                    continue
                for lag in lag_periods:
                    regime_df[f"{col}_lag_{lag}_r{regime}"] = regime_df[col].shift(lag)

            # 2. Rolling特征
            if self.verbose:
                print(f"    创建Rolling特征 (windows={rolling_windows})...")
            for col in feature_cols:
                if col not in regime_df.columns:
                    continue
                for w in rolling_windows:
                    regime_df[f"{col}_roll_mean_{w}_r{regime}"] = \
                        regime_df[col].rolling(w, min_periods=1).mean()
                    regime_df[f"{col}_roll_std_{w}_r{regime}"] = \
                        regime_df[col].rolling(w, min_periods=1).std()
                    regime_df[f"{col}_roll_max_{w}_r{regime}"] = \
                        regime_df[col].rolling(w, min_periods=1).max()

            # 3. Pct_change特征
            if self.verbose:
                print(f"    创建Pct_change特征 (periods={pct_periods})...")
            for col in feature_cols:
                if col not in regime_df.columns:
                    continue
                for p in pct_periods:
                    regime_df[f"{col}_pct_{p}_r{regime}"] = regime_df[col].pct_change(periods=p)

            # 4. EWM特征
            if self.verbose:
                print(f"    创建EWM特征 (spans={ewm_spans})...")
            for col in feature_cols:
                if col not in regime_df.columns:
                    continue
                for s in ewm_spans:
                    regime_df[f"{col}_ewm_{s}_r{regime}"] = \
                        regime_df[col].ewm(span=s, adjust=False, min_periods=1).mean()

            regime_features.append(regime_df)

        # 合并所有Regime的特征
        if self.verbose:
            print(f"\n  合并所有Regime的特征...")

        # 使用concat合并（按行索引对齐）
        df_merged = pd.concat(regime_features, axis=0)

        # 按原始索引排序
        df_merged = df_merged.sort_index()

        # 缺失值处理
        if self.verbose:
            print("  处理缺失值...")
        df_merged.ffill(inplace=True)
        df_merged.bfill(inplace=True)

        for c in df_merged.columns:
            if df_merged[c].isnull().any():
                df_merged[c].fillna(df_merged[c].median(), inplace=True)

        if self.verbose:
            original_cols = len(df.columns)
            new_cols = len(df_merged.columns) - original_cols
            print(f"\n【动态特征工程】输出: {df_merged.shape}")
            print(f"  原始特征数: {original_cols}")
            print(f"  新增特征数: {new_cols}")
            print(f"  总特征数: {len(df_merged.columns)}")

        return df_merged

    def create_features_slim(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【功能】精简特征工程函数 - 针对小窗口在线学习优化

        【参数】
            df (pd.DataFrame): 包含原始特征的DataFrame

        【返回】
            pd.DataFrame: 包含衍生特征的DataFrame

        【特征生成】
            输入：50个原始因子
            输出：约2000个衍生特征

            包含：
            - Lag特征：50 × 6 = 300
            - Rolling特征：50 × 5 × 3 = 750 (只用mean/std/max)
            - Pct_change：50 × 3 = 150
            - EWMA：50 × 3 = 150
            - 交互特征：C(20,2) × 2 = 380 (只用Top20因子)
            - Rolling Z-score：50 × 2 = 100

            总计：≈1830特征
        """
        if self.verbose:
            print(f"\n【精简特征工程】输入: {df.shape}")

        # 保留date_id和target列
        preserve_cols = []
        if 'date_id' in df.columns:
            preserve_cols.append('date_id')
        if self.target_col in df.columns:
            preserve_cols.append(self.target_col)

        # 获取特征列
        feature_cols = [c for c in df.columns if c not in preserve_cols]

        all_features = []

        # 1. Lag特征 (50 × 6 = 300)
        if self.verbose:
            print("  创建Lag特征...")
        lag_periods = [1, 3, 5, 7, 14, 20]
        for col in feature_cols:
            for lag in lag_periods:
                all_features.append(df[col].shift(lag).rename(f'{col}_lag_{lag}'))

        # 2. Rolling统计特征 (50 × 5 × 3 = 750)
        if self.verbose:
            print("  创建Rolling特征...")
        rolling_windows = [3, 6, 10, 20, 60]
        rolling_funcs = ['mean', 'std', 'max']  # 只用3个统计量

        for col in feature_cols:
            for window in rolling_windows:
                for func in rolling_funcs:
                    feature_name = f'{col}_roll{window}_{func}'
                    all_features.append(
                        df[col].rolling(window=window, min_periods=1).agg(func).rename(feature_name)
                    )

        # 3. Pct_change (50 × 3 = 150)
        if self.verbose:
            print("  创建Pct_change特征...")
        pct_periods = [1, 3, 7]
        for col in feature_cols:
            for period in pct_periods:
                all_features.append(
                    df[col].pct_change(periods=period).rename(f'{col}_pct{period}')
                )

        # 4. EWMA (50 × 3 = 150)
        if self.verbose:
            print("  创建EWMA特征...")
        ewma_spans = [7, 14, 30]
        for col in feature_cols:
            for span in ewma_spans:
                all_features.append(
                    df[col].ewm(span=span, min_periods=1).mean().rename(f'{col}_ewm{span}')
                )

        # 5. 交互特征 (只用Top20因子，C(20,2) × 2 = 380)
        if self.verbose:
            print("  创建交互特征...")
        # 取前20个特征（假设已经按重要性排序，或者按列名排序）
        top_features = feature_cols[:min(20, len(feature_cols))]

        for i, col1 in enumerate(top_features):
            for col2 in top_features[i+1:]:
                # 乘法交互
                all_features.append(
                    (df[col1] * df[col2]).rename(f'{col1}_x_{col2}')
                )
                # 除法交互（避免除零）
                all_features.append(
                    (df[col1] / (df[col2].replace(0, np.nan) + 1e-8)).rename(f'{col1}_div_{col2}')
                )

        # 6. Rolling Z-score (50 × 2 = 100)
        if self.verbose:
            print("  创建Rolling Z-score特征...")
        zscore_windows = [14, 30]
        for col in feature_cols:
            for window in zscore_windows:
                rolling_mean = df[col].rolling(window=window, min_periods=1).mean()
                rolling_std = df[col].rolling(window=window, min_periods=1).std()
                zscore = ((df[col] - rolling_mean) / (rolling_std + 1e-8)).rename(f'{col}_zscore{window}')
                all_features.append(zscore)

        # 合并所有特征
        if self.verbose:
            print("  合并特征...")
        feature_df = pd.concat(all_features, axis=1)

        # 添加保留列
        for col in preserve_cols:
            feature_df[col] = df[col].values

        # 缺失值处理
        if self.verbose:
            print("  处理缺失值...")
        feature_df.ffill(inplace=True)
        feature_df.bfill(inplace=True)

        # 中位数填充剩余缺失值
        for c in feature_df.columns:
            if feature_df[c].isnull().any():
                feature_df[c].fillna(feature_df[c].median(), inplace=True)

        if self.verbose:
            print(f"【精简特征工程】输出: {feature_df.shape}")
            print(f"  生成特征数: {feature_df.shape[1] - len(preserve_cols)}")

        return feature_df
