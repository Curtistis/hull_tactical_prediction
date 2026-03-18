"""
【特征预处理器】Feature Preprocessor
处理流程: IC负值翻转 → 三轮变换优化 → 基于IC排序筛选
"""

import numpy as np
import pandas as pd


class FeaturePreprocessor:
    """特征预处理器 - 在特征工程前优化原始特征"""

    def __init__(self, analyzer, target_col='market_forward_excess_returns', verbose=True):
        """
        Args:
            analyzer: FactorICAnalyzer实例
            target_col: 目标列名
            verbose: 是否打印详细信息
        """
        self.analyzer = analyzer
        self.target_col = target_col
        self.verbose = verbose
        self.exclude_cols = {target_col, 'date_id', 'forward_returns', 'risk_free_rate'}

    def flip_negative_ic_features(self, train_data, test_data, factor_df):
        """
        对IC为负的特征进行翻转(乘以-1)

        Args:
            train_data: 训练数据（polars或pandas DataFrame）
            test_data: 测试数据（polars或pandas DataFrame）
            factor_df: IC分析结果DataFrame（包含'特征'和'IC'列）

        Returns:
            train_flipped: 翻转后的训练集
            test_flipped: 翻转后的测试集
        """
        if self.verbose:
            print("\n开始IC负值特征翻转...")

        # 1. 识别IC为负的特征
        negative_ic_features = factor_df[factor_df['IC'] < 0]['特征'].tolist()

        if self.verbose:
            print(f"发现 {len(negative_ic_features)} 个IC为负的特征")
            if len(negative_ic_features) > 0:
                print(f"前10个需要翻转的特征: {negative_ic_features[:10]}")

        # 2. 转换为pandas进行处理
        train_pd = self._to_pandas(train_data)
        test_pd = self._to_pandas(test_data)

        # 3. 对训练集进行翻转
        if self.verbose:
            print("\n处理训练集...")

        flipped_train_count = 0
        for feature in negative_ic_features:
            if feature in train_pd.columns:
                train_pd[feature] = train_pd[feature] * (-1)
                flipped_train_count += 1

        if self.verbose:
            print(f"✓ 训练集中 {flipped_train_count} 个特征已翻转")

        # 4. 对测试集进行相同的翻转
        if self.verbose:
            print("\n处理测试集...")

        flipped_test_count = 0
        for feature in negative_ic_features:
            if feature in test_pd.columns:
                test_pd[feature] = test_pd[feature] * (-1)
                flipped_test_count += 1

        if self.verbose:
            print(f"✓ 测试集中 {flipped_test_count} 个特征已翻转")

        # 5. 转回原始格式
        train_flipped = self._from_pandas(train_pd, train_data)
        test_flipped = self._from_pandas(test_pd, test_data)

        if self.verbose:
            print("\n" + "="*80)
            print("【IC负值特征翻转完成】")
            print("="*80)
            print(f"训练集形状: {train_flipped.shape}")
            print(f"测试集形状: {test_flipped.shape}")

        return train_flipped, test_flipped

    def apply_transformation_if_better(self, train_data, test_data, transform_func,
                                      transform_name='变换'):
        """
        对所有特征尝试应用变换，只保留改进的特征

        Args:
            train_data: 训练数据
            test_data: 测试数据
            transform_func: 变换函数
            transform_name: 变换名称（用于打印）

        Returns:
            train_optimized: 优化后的训练集
            test_optimized: 优化后的测试集
            report_df: 变换报告DataFrame
        """
        if self.verbose:
            print(f"\n开始应用【{transform_name}】...")

        # 转换为pandas
        train_pd = self._to_pandas(train_data)
        test_pd = self._to_pandas(test_data)

        # 获取目标值
        target = pd.to_numeric(train_pd[self.target_col], errors='coerce').values

        # 获取特征列
        feature_cols = [col for col in train_pd.columns if col not in self.exclude_cols]

        if self.verbose:
            print(f"共有 {len(feature_cols)} 个特征需要评估...")

        # 评估每个特征
        improvements = []
        improved_count = 0

        for idx, col in enumerate(feature_cols, 1):
            if self.verbose and idx % 50 == 0:
                print(f"  已完成 {idx}/{len(feature_cols)} 个特征...")

            # 计算原始IC
            original_values = train_pd[col].values
            original_metrics = self.analyzer.calculate_single_factor_metrics(original_values, target)
            original_fitness = abs(original_metrics['Fitness'])

            # 应用变换
            try:
                transformed_series = transform_func(train_pd[col])
                transformed_values = transformed_series.values

                # 检查变换后是否有效
                if np.isnan(transformed_values).all() or np.isinf(transformed_values).any():
                    is_improved = False
                    new_fitness = original_fitness
                else:
                    # 计算变换后的IC
                    transformed_metrics = self.analyzer.calculate_single_factor_metrics(transformed_values, target)
                    new_fitness = abs(transformed_metrics['Fitness'])

                    # 判断是否改进
                    is_improved = new_fitness > original_fitness

                    if is_improved:
                        # 更新训练集和测试集
                        train_pd[col] = transformed_series
                        if col in test_pd.columns:
                            try:
                                test_pd[col] = transform_func(test_pd[col])
                            except:
                                pass
                        improved_count += 1
            except Exception as e:
                is_improved = False
                new_fitness = original_fitness

            improvements.append({
                '特征': col,
                '原始Fitness': original_fitness,
                f'{transform_name}后Fitness': new_fitness,
                '是否改进': is_improved,
                '改进幅度': new_fitness - original_fitness if is_improved else 0
            })

        # 生成报告
        report_df = pd.DataFrame(improvements)
        report_df = report_df.sort_values('改进幅度', ascending=False)

        if self.verbose:
            print(f"\n✅ 【{transform_name}】完成！")
            print(f"   改进特征数: {improved_count}/{len(feature_cols)} ({100*improved_count/len(feature_cols):.1f}%)")
            if improved_count > 0:
                print(f"   平均改进幅度: {report_df[report_df['是否改进']]['改进幅度'].mean():.4f}")
                print(f"\n   Top 10 改进最大的特征:")
                top_improved = report_df[report_df['是否改进']].head(10)
                for _, row in top_improved.iterrows():
                    print(f"     {row['特征']}: {row['原始Fitness']:.2f} → {row[f'{transform_name}后Fitness']:.2f} (+{row['改进幅度']:.2f})")

        # 转回原始格式
        train_optimized = self._from_pandas(train_pd, train_data)
        test_optimized = self._from_pandas(test_pd, test_data)

        return train_optimized, test_optimized, report_df

    def select_features_by_ic(self, train_data, test_data, factor_df,
                             ic_threshold=0.01, top_n=None, min_fitness=None):
        """
        根据IC值筛选特征（支持动态因子池管理）

        Args:
            train_data: 训练数据
            test_data: 测试数据
            factor_df: IC分析结果DataFrame（需包含'特征'、'IC'、'Fitness'列）
            ic_threshold: |IC| 的最小阈值
            top_n: 如果指定，只保留Top N个特征（按|IC|排序）
            min_fitness: Fitness的最小阈值，用于退场条件（例如0.01）

        Returns:
            train_selected: 筛选后的训练集
            test_selected: 筛选后的测试集
            selected_features: 选中的特征列表
        """
        if self.verbose:
            print("\n" + "="*80)
            print("【基于IC进行特征筛选】（动态因子池管理）")
            print("="*80)

        # 转换为pandas
        train_pd = self._to_pandas(train_data)
        test_pd = self._to_pandas(test_data)

        # 按|IC|排序
        factor_df_sorted = factor_df.copy()
        factor_df_sorted['IC_abs'] = factor_df_sorted['IC'].abs()
        factor_df_sorted = factor_df_sorted.sort_values('IC_abs', ascending=False)

        # 筛选特征 - 支持多重条件
        if top_n is not None:
            # 优先按top_n筛选
            selected_factor_df = factor_df_sorted.head(top_n)
            if self.verbose:
                print(f"\n策略: 保留Top {top_n}个特征（按|IC|排序）")
        else:
            # 按IC阈值筛选
            selected_factor_df = factor_df_sorted[factor_df_sorted['IC_abs'] >= ic_threshold]
            if self.verbose:
                print(f"\n策略: 保留|IC| >= {ic_threshold}的特征")

        # 应用Fitness退场条件（如果指定）
        if min_fitness is not None:
            original_count = len(selected_factor_df)
            selected_factor_df = selected_factor_df[selected_factor_df['Fitness'] >= min_fitness]
            removed_count = original_count - len(selected_factor_df)

            if self.verbose:
                print(f"   + 额外退场条件: Fitness >= {min_fitness:.4f}")
                if removed_count > 0:
                    print(f"   剔除 {removed_count} 个低Fitness因子")

        selected_features = selected_factor_df['特征'].tolist()

        # 保留必要的列
        keep_cols = [col for col in self.exclude_cols if col in train_pd.columns]
        final_cols = keep_cols + selected_features

        # 筛选列
        train_selected = train_pd[final_cols]
        test_cols = [col for col in final_cols if col in test_pd.columns]
        test_selected = test_pd[test_cols]

        if self.verbose:
            print(f"\n✅ 特征筛选完成！")
            print(f"   原始特征数: {len(train_pd.columns) - len(keep_cols)}")
            print(f"   筛选后特征数: {len(selected_features)}")
            print(f"   保留比例: {100*len(selected_features)/(len(train_pd.columns)-len(keep_cols)):.1f}%")

            print(f"\n   IC统计:")
            print(f"     平均|IC|: {selected_factor_df['IC_abs'].mean():.4f}")
            print(f"     最大|IC|: {selected_factor_df['IC_abs'].max():.4f}")
            print(f"     最小|IC|: {selected_factor_df['IC_abs'].min():.4f}")

            print(f"\n   Fitness统计:")
            print(f"     平均Fitness: {selected_factor_df['Fitness'].mean():.4f}")
            print(f"     最大Fitness: {selected_factor_df['Fitness'].max():.4f}")
            print(f"     最小Fitness: {selected_factor_df['Fitness'].min():.4f}")

            print(f"\n   Top 10 特征:")
            for idx, row in selected_factor_df.head(10).iterrows():
                print(f"     {row['特征']}: IC={row['IC']:.4f}, Fitness={row['Fitness']:.4f}")

        # 转回原始格式
        train_selected = self._from_pandas(train_selected, train_data)
        test_selected = self._from_pandas(test_selected, test_data)

        return train_selected, test_selected, selected_features

    # ==================== 变换函数 ====================

    @staticmethod
    def winsorize_3sigma(series):
        """3-Sigma去极值"""
        mean_val = series.mean()
        std_val = series.std()
        upper = mean_val + 3 * std_val
        lower = mean_val - 3 * std_val
        return series.clip(lower, upper)

    @staticmethod
    def log_transform(series):
        """对数变换（处理偏态分布）"""
        min_val = series.min()
        if min_val <= 0:
            series = series - min_val + 1e-8
        return np.log(series)

    @staticmethod
    def rank_transform(series):
        """排序变换（转为百分位）"""
        return series.rank(pct=True)

    # ==================== 工具函数 ====================

    @staticmethod
    def _to_pandas(data):
        """转换为pandas DataFrame"""
        if hasattr(data, 'to_pandas'):
            return data.to_pandas()
        else:
            return data.copy()

    @staticmethod
    def _from_pandas(pd_data, original_data):
        """转回原始格式"""
        if hasattr(original_data, 'to_pandas'):
            import polars as pl
            return pl.from_pandas(pd_data)
        else:
            return pd_data
