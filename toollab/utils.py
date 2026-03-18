"""
【工具函数库】Utils
包含数据分析、可视化、计时等辅助函数
"""

import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from typing import Optional, List


# ==================== 数据分析函数 ====================

def describe_dataset(df: pd.DataFrame):
    """
    【功能】生成数据集的详细描述报告
    【参数】
        df (pd.DataFrame): 要分析的数据集
    【输出】
        打印数据集的统计摘要，包括：
        - 数据集大小 (行数×列数)
        - 内存占用
        - 数据类型分布
        - 数值型特征的统计量 (均值、标准差、分位数等)
    【用途】
        快速了解数据集的基本情况，用于EDA探索性数据分析
    """
    print("=" * 80)
    print("📋 Dataset Overview")
    print("=" * 80)
    print("\n--- Basic Dimensions & Memory ---")

    # shape 形状
    num_rows, num_cols = df.shape
    print(f"**Shape (Rows, Columns):** ({num_rows:,}, {num_cols:,})")

    # memory 内存
    mem_usage = df.memory_usage(deep=True).sum()
    mem_mbs = mem_usage / (1024**2)
    print(f"**Total Memory Usage:** {mem_mbs:.2f} MB")

    print("\n--- Feature Data Types and Counts ---")

    # data types 数据类型
    dtype_counts = df.dtypes.astype(str).value_counts().reset_index()
    dtype_counts.columns = ['Data_Type', 'Count']
    print(dtype_counts.to_string(index=False))

    # stats 统计信息
    print("\n" + "=" * 80)
    print("📊 Descriptive Statistics")
    print("=" * 80)

    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()

    # numerical 数字
    if numerical_cols:
        print("\n### Numerical Features")
        # Use transpose for better readability when many features exist
        num_desc = df[numerical_cols].describe().T
        # Add IQR for a more detailed statistical view
        num_desc['IQR'] = num_desc['75%'] - num_desc['25%']
        print(num_desc.to_string())
        print(f"\nFound {len(numerical_cols)} numerical features.")

    # categorical
    if categorical_cols:
        print("\n### Categorical / Object Features")
        # Include top, frequency, and unique count
        cat_desc = df[categorical_cols].describe().T
        print(cat_desc.to_string())
        print(f"\nFound {len(categorical_cols)} categorical/object features.")

    # datetime
    if datetime_cols:
        print("\n### Datetime Features")
        dt_desc = df[datetime_cols].describe().T
        print(dt_desc.to_string())
        print(f"\nFound {len(datetime_cols)} datetime features.")


def missing_duplicates_analysis(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    【功能】分析数据集的缺失值和重复行
    【参数】
        df (pd.DataFrame): 要分析的数据集
        top_n (int): 显示前N个缺失最多的特征 (默认20)
    【返回】
        pd.DataFrame: 缺失值汇总表，包含:
                      - Missing_Count: 缺失值数量
                      - Missing_Percent: 缺失百分比
    【原理】
        1. 统计每列的缺失值数量和占比
        2. 检测重复行 (完全相同的样本)
    【缺失值处理策略】
        检测 → 报告 → 后续由特征工程函数处理:
        - ffill: 前向填充 (用前一个值)
        - bfill: 后向填充 (用后一个值)
        - fillna(median): 剩余的用中位数
    """
    print("--- 📊 Missing Data and Duplicates Analysis ---")

    # ========== 1. 统计缺失值 ==========
    # df.isnull().sum(): 统计每列的True(缺失)数量
    missing_counts = df.isnull().sum()

    # 创建缺失值汇总表
    missing_summary = pd.DataFrame({
        'Missing_Count': missing_counts,           # 缺失数量
        'Missing_Percent': 100 * missing_counts / len(df)  # 缺失百分比
    })

    # 只保留有缺失的列 (Missing_Count > 0)
    missing_summary = missing_summary[missing_summary['Missing_Count'] > 0]

    # 按缺失数量降序排序 (缺失最多的排前面)
    missing_summary = missing_summary.sort_values(by='Missing_Count', ascending=False)

    # ========== 2. 检测重复行 ==========
    # df.duplicated(): 返回布尔Series，True表示重复行
    # .sum(): 统计重复行数量
    num_duplicates = df.duplicated().sum()
    print(f"**Duplicate rows found:** {num_duplicates}")

    # ========== 3. 打印摘要 ==========
    if missing_summary.empty:
        print("✅ **No missing values found** in the dataset.")
        return pd.DataFrame()

    print(f"\n**Total features with missing values:** {len(missing_summary)}")

    print("\n**Missing Data Summary Table (Top 10):**")
    print(missing_summary.head(top_n).to_string())

    return missing_summary


def detect_outliers(df, method='iqr', threshold=2.5, z_threshold=3.0, cols=None, summary=True):
    """
    【功能】检测离群值
    【参数】
        df: DataFrame
        method: 'iqr' 或 'zscore'
        threshold: IQR方法的阈值倍数
        z_threshold: Z-score方法的阈值
        cols: 要检测的列（默认所有数值列）
        summary: 是否打印摘要
    """
    from scipy import stats

    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if method not in ['iqr', 'zscore']:
        raise ValueError("method must be 'iqr' or 'zscore'")

    outlier_flags = pd.DataFrame(False, index=df.index, columns=cols)

    for col in cols:
        series = df[col].dropna()

        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outlier_flags[col] = (df[col] < lower_bound) | (df[col] > upper_bound)

        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(series))
            outlier_flags[col] = z_scores > z_threshold

    if summary:
        summary_df = pd.DataFrame({
            'outlier_count': outlier_flags.sum(),
            'percent_outliers': 100 * outlier_flags.sum() / len(df)
        }).sort_values('percent_outliers', ascending=False)

        print("📊 Outlier Detection Summary:")
        print(summary_df.round(2))
        return outlier_flags, summary_df

    return outlier_flags


# ==================== 时间和性能工具 ====================

def now_str() -> str:
    """返回当前时间字符串"""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def timer(func):
    """
    【功能】函数计时器装饰器

    【用法】
        @timer
        def my_function():
            ...
    【效果】
        自动打印函数的开始时间和运行时长
    【输出示例】
        [2025-11-27 10:00:00] START tune_lightgbm
        [2025-11-27 10:12:34] DONE  tune_lightgbm (elapsed 754.2s)
    """
    def wrapper(*args, **kwargs):
        start = time.time()  # 起点精确时间
        print(f"[{now_str()}] START {func.__name__}")  # now_str 当前时间字符串 START 函数
        result = func(*args, **kwargs)  # 传递参数
        elapsed = time.time() - start  # 运行花费总时长
        print(f"[{now_str()}] DONE  {func.__name__} (elapsed {elapsed:.1f}s)")  # 函数结束以及已经运行了多久
        return result
    return wrapper


# ==================== 评估指标 ====================

def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    【功能】计算Spearman相关系数（排序相关）
    【参数】
        y_true (np.ndarray): 真实值数组
        y_pred (np.ndarray): 预测值数组
    【返回】
        float: Spearman相关系数（-1到1之间）

    【注意】
        如果计算失败（NaN），返回0.0（中性分数）而非-1.0
        NaN通常表示无法计算相关性（如方差为0），应视为无信息而非负相关
    """
    corr, _ = spearmanr(y_true, y_pred)
    if np.isnan(corr):
        return 0.0  # 修正：NaN视为无相关性（0.0）而非最差分数（-1.0）
    return float(corr)


# ==================== 交叉验证工具 ====================

def time_series_cv_splits(X: pd.DataFrame, n_splits: int = 4):
    """
    【功能】时间序列交叉验证分割
    【参数】
        X: 特征矩阵
        n_splits: 分割数
    【返回】
        生成器，每次yield (train_idx, val_idx)
    """
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return tscv.split(X)


def sliding_window_cv_splits(X, train_size=90, n_splits=4):
    """
    【已废弃】建议使用 true_online_cv_splits

    【功能】滑动窗口交叉验证 - 每折固定使用train_size条数据

    【参数】
        X: 特征矩阵（DataFrame或数组）
        train_size: 训练窗口大小
        n_splits: 分割数

    【返回】
        生成器，每次yield (train_idx, val_idx)
    """
    n_samples = len(X)
    val_size = (n_samples - train_size) // n_splits

    for i in range(n_splits):
        val_start = train_size + i * val_size
        val_end = min(val_start + val_size, n_samples)
        train_start = max(0, val_start - train_size)
        train_end = val_start
        train_idx = np.arange(train_start, train_end)
        val_idx = np.arange(val_start, val_end)
        yield train_idx, val_idx


def true_online_cv_splits(X, train_size=90, step=10):
    """
    【真正的在线学习CV】Walk-Forward滑动窗口验证

    完全模拟推理场景：每step天重新训练一次模型

    【参数】
        X: 特征矩阵（DataFrame或数组）
        train_size: 训练窗口大小（天数）
        step: 滑动步长（天数），即多久重新训练一次

    【返回】
        生成器，每次yield (train_idx, val_idx)

    【示例】
        9000条数据, train_size=90, step=10:
        Window 1:  Train[0:90]      → Val[90:100]
        Window 2:  Train[10:100]    → Val[100:110]
        Window 3:  Train[20:110]    → Val[110:120]
        ...
        Window 890: Train[8900:8990] → Val[8990:9000]

        总计：890个窗口（完全模拟在线学习场景）

    【真实场景对应】
        - Train[0:90]   → 用前90天数据训练
        - Val[90:100]   → 预测接下来10天
        - Train[10:100] → 滑动10天，用新的90天数据重新训练
        - Val[100:110]  → 预测接下来10天
        - ...

    【性能】
        单次训练时间（90样本，2000特征）：0.05-0.1秒
        总时间（890窗口）：约90秒/trial
    """
    n_samples = len(X)
    val_size = step

    for start in range(0, n_samples - train_size - val_size, step):
        train_end = start + train_size
        val_start = train_end
        val_end = val_start + val_size

        if val_end > n_samples:
            break

        train_idx = np.arange(start, train_end)
        val_idx = np.arange(val_start, val_end)
        yield train_idx, val_idx


# ==================== 可视化占位函数 ====================

def plot_feature_importance_lgbm(model, feature_names: List[str], top_n: int = 20,
                                 figsize=(10, 8), save_path: Optional[str] = None):
    """【功能】绘制LightGBM特征重要性（本地环境跳过）"""
    print("(绘图已跳过 - 本地环境无matplotlib)")
    return


def plot_feature_importance_catboost(model, feature_names: List[str], top_n: int = 20,
                                     figsize=(10, 8), save_path: Optional[str] = None):
    """【功能】绘制CatBoost特征重要性（本地环境跳过）"""
    print("(绘图已跳过 - 本地环境无matplotlib)")
    return


# ==================== 因子重要性分析 ====================

def analyze_factor_importance_from_model(
    model,
    feature_names: List[str],
    original_factors: List[str],
    top_n: int = 50,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    【功能】从训练好的模型中提取原始因子的重要性

    【参数】
        model: 训练好的模型（LightGBM或CatBoost）
        feature_names: 模型训练时使用的特征名列表（~2000个衍生特征）
        original_factors: 原始因子列表（50个）
        top_n: 显示Top N个因子
        save_path: 保存路径（可选）

    【返回】
        pd.DataFrame: 因子重要性汇总表

    【原理】
        1. 获取模型的特征重要性（2000个衍生特征）
        2. 将衍生特征重要性聚合到原始因子
           例如：v042_lag_1, v042_roll_mean_10, v042_pct_change_3
                都聚合到 v042
        3. 按重要性排序

    【使用场景】
        训练完成后，分析哪些原始因子对模型贡献最大
    """
    # 获取模型特征重要性
    try:
        # LightGBM
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        # CatBoost
        elif hasattr(model, 'get_feature_importance'):
            importances = model.get_feature_importance()
        else:
            raise ValueError("模型不支持特征重要性提取")
    except Exception as e:
        print(f"❌ 无法提取特征重要性: {e}")
        return pd.DataFrame()

    # 归一化重要性（总和=100）
    importance_sum = importances.sum()
    if importance_sum > 0:
        importances_pct = (importances / importance_sum) * 100
    else:
        importances_pct = importances

    # 创建特征重要性映射
    feature_importance_map = dict(zip(feature_names, importances_pct))

    # 聚合到原始因子
    factor_importance = {}
    factor_feature_count = {}  # 每个因子生成了多少衍生特征
    factor_top_features = {}   # 每个因子最重要的3个衍生特征

    for factor in original_factors:
        # 找到所有包含该因子名的衍生特征
        # 例如：v042 → v042_lag_1, v042_roll_mean_10, etc.
        derived_features = [
            feat for feat in feature_names
            if feat.startswith(factor + '_') or feat == factor
        ]

        if len(derived_features) == 0:
            factor_importance[factor] = 0.0
            factor_feature_count[factor] = 0
            factor_top_features[factor] = []
            continue

        # 聚合重要性（求和）
        total_importance = sum([feature_importance_map.get(feat, 0.0) for feat in derived_features])
        factor_importance[factor] = total_importance
        factor_feature_count[factor] = len(derived_features)

        # 找Top 3衍生特征
        derived_importance = [(feat, feature_importance_map.get(feat, 0.0)) for feat in derived_features]
        derived_importance.sort(key=lambda x: x[1], reverse=True)
        factor_top_features[factor] = derived_importance[:3]

    # 创建DataFrame
    importance_df = pd.DataFrame({
        'factor': list(factor_importance.keys()),
        'importance': list(factor_importance.values()),
        'n_derived_features': [factor_feature_count[f] for f in factor_importance.keys()],
        'avg_importance': [
            factor_importance[f] / factor_feature_count[f] if factor_feature_count[f] > 0 else 0.0
            for f in factor_importance.keys()
        ]
    })

    # 添加Top衍生特征信息
    importance_df['top_3_features'] = [
        ', '.join([f"{feat}({imp:.2f}%)" for feat, imp in factor_top_features[f]])
        for f in importance_df['factor']
    ]

    # 排序
    importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)

    # 添加排名和累计重要性
    importance_df['rank'] = range(1, len(importance_df) + 1)
    importance_df['cumulative_importance'] = importance_df['importance'].cumsum()

    # 打印报告
    print("="*100)
    print("【因子重要性分析报告】")
    print("="*100)
    print(f"\n模型类型: {type(model).__name__}")
    print(f"原始因子数: {len(original_factors)}")
    print(f"衍生特征数: {len(feature_names)}")
    print(f"总重要性: 100.00%")

    print(f"\n【Top {min(top_n, len(importance_df))} 重要因子】")
    print("-"*100)
    display_cols = ['rank', 'factor', 'importance', 'n_derived_features', 'avg_importance', 'cumulative_importance']
    print(importance_df.head(top_n)[display_cols].to_string(index=False))

    # 统计
    top_10_importance = importance_df.head(10)['importance'].sum()
    top_20_importance = importance_df.head(20)['importance'].sum()

    print(f"\n【重要性分布】")
    print(f"  Top 10 因子贡献: {top_10_importance:.2f}%")
    print(f"  Top 20 因子贡献: {top_20_importance:.2f}%")
    print(f"  Top 30 因子贡献: {importance_df.head(30)['importance'].sum():.2f}%")

    # D类因子统计
    d_factors = importance_df[importance_df['factor'].str.startswith('d')]
    if len(d_factors) > 0:
        print(f"\n【D类因子表现】")
        print(f"  D类因子数: {len(d_factors)}")
        print(f"  D类总贡献: {d_factors['importance'].sum():.2f}%")
        print(f"  D类平均贡献: {d_factors['importance'].mean():.2f}%")
        print(f"\n  D类因子 Top 5:")
        print(d_factors.head(5)[['rank', 'factor', 'importance']].to_string(index=False))

    # 保存
    if save_path:
        importance_df.to_csv(save_path, index=False)
        print(f"\n✅ 因子重要性报告已保存: {save_path}")

    print("="*100)

    return importance_df


# ==================== 模型诊断函数 ====================

def analyze_lgbm_tree_complexity(model):
    """
    【功能】统计 LightGBM 模型的树复杂度（树数量、平均深度、平均叶子数等）

    Args:
        model: 已训练好的 LGBMRegressor（带 booster_ 属性）

    Returns:
        dict: {
            'n_trees': int,
            'avg_depth': float,
            'max_depth': int,
            'avg_leaves': float,
            'max_leaves': int,
        }
    """
    booster = getattr(model, "booster_", None)
    if booster is None:
        return {
            "n_trees": 0,
            "avg_depth": 0.0,
            "max_depth": 0,
            "avg_leaves": 0.0,
            "max_leaves": 0,
        }

    dump = booster.dump_model()
    tree_infos = dump.get("tree_info", [])
    if not tree_infos:
        return {
            "n_trees": 0,
            "avg_depth": 0.0,
            "max_depth": 0,
            "avg_leaves": 0.0,
            "max_leaves": 0,
        }

    depths = []
    leaves = []

    def _depth(node):
        # 递归计算一棵树的最大深度
        if "left_child" not in node and "right_child" not in node:
            return 1
        left = _depth(node["left_child"]) if "left_child" in node else 0
        right = _depth(node["right_child"]) if "right_child" in node else 0
        return 1 + max(left, right)

    for t in tree_infos:
        tree_struct = t.get("tree_structure", {})
        if tree_struct:
            depths.append(_depth(tree_struct))
        leaves.append(t.get("num_leaves", 0))

    if not depths:
        avg_depth = 0.0
        max_depth = 0
    else:
        avg_depth = float(np.mean(depths))
        max_depth = int(max(depths))

    if not leaves:
        avg_leaves = 0.0
        max_leaves = 0
    else:
        avg_leaves = float(np.mean(leaves))
        max_leaves = int(max(leaves))

    return {
        "n_trees": len(tree_infos),
        "avg_depth": avg_depth,
        "max_depth": max_depth,
        "avg_leaves": avg_leaves,
        "max_leaves": max_leaves,
    }


def apply_preprocessing_rules(df, preprocessing_rules):
    """
    对新数据应用已保存的预处理规则

    Args:
        df (pd.DataFrame): 新数据
        preprocessing_rules (dict): 预处理规则字典，格式：
            {
                'flip_rules': ['A1', 'B2'],  # 需要翻转的特征
                'transform_rules': {
                    'A1': 'log',       # 应用Log变换
                    'B2': 'rank',      # 应用Rank变换
                    'C3': 'winsorize'  # 应用3-Sigma裁剪
                }
            }

    Returns:
        pd.DataFrame: 应用变换后的DataFrame

    Example:
        >>> import json
        >>> with open('preprocessing_rules.json') as f:
        ...     rules = json.load(f)
        >>> new_data_transformed = apply_preprocessing_rules(new_data, rules)
    """
    df = df.copy()

    # Step 1: IC负值翻转
    flip_rules = preprocessing_rules.get('flip_rules', [])
    flip_count = 0
    for col in flip_rules:
        if col in df.columns:
            df[col] = df[col] * (-1)
            flip_count += 1

    # Step 2: 应用变换（内联变换函数，避免依赖）
    def winsorize_3sigma(series):
        """3-Sigma去极值"""
        mean_val = series.mean()
        std_val = series.std()
        upper = mean_val + 3 * std_val
        lower = mean_val - 3 * std_val
        return series.clip(lower, upper)

    def log_transform(series):
        """对数变换（处理偏态分布）"""
        min_val = series.min()
        if min_val <= 0:
            series = series - min_val + 1e-8
        return np.log(series)

    def rank_transform(series):
        """排序变换（转为百分位）"""
        return series.rank(pct=True)

    transforms = {
        'winsorize': winsorize_3sigma,
        'log': log_transform,
        'rank': rank_transform
    }

    transform_rules = preprocessing_rules.get('transform_rules', {})
    transform_count = 0
    for col, transform_name in transform_rules.items():
        if col in df.columns and transform_name != 'none' and transform_name in transforms:
            try:
                df[col] = transforms[transform_name](df[col])
                transform_count += 1
            except Exception as e:
                print(f"⚠️ 警告: 特征 {col} 变换失败: {e}")

    print(f"✅ 预处理规则已应用")
    print(f"   翻转特征: {flip_count}个")
    print(f"   变换特征: {transform_count}个")

    return df
