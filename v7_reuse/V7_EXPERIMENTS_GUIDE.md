# V7 Notebook 四核心实验指南

## 📋 概述

本文档详细说明V7 Notebook中四个核心实验的实现方案，这些实验旨在系统性优化模型性能并验证三模型集成架构（LightGBM + CatBoost + MLP）的有效性。

---

## ✅ ModelTuner 实验函数修改完成状态

### 实验1: 树结构超参数网格搜索 ✅
**文件**: `toollab/model_tuner.py`
**函数**: `grid_search_lgbm_tree_structure`
**状态**: 已按文档要求完全重写

#### 关键修改：
- ✅ 函数签名改为 `X_train, y_train, X_val, y_val` 分离参数
- ✅ 新增 `base_params` 参数（基础LightGBM参数）
- ✅ 新增 `min_split_gain_list` 参数（扫描 min_split_gain）
- ✅ 使用 `itertools.product` 进行4维网格搜索
- ✅ 输出包含树复杂度统计（avg_depth, max_depth_observed, avg_leaves, max_leaves_observed）

#### 函数签名：
```python
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
) -> pd.DataFrame
```

---

### 实验2: 训练窗口大小实验 ✅
**文件**: `toollab/model_tuner.py`
**函数**: `experiment_lgbm_window_sizes`
**状态**: 已增强，添加树复杂度统计

#### 关键修改：
- ✅ 函数签名改为必需参数（去除默认值）
- ✅ 新增 `step` 参数（控制滑动步长）
- ✅ 每个窗口记录树复杂度（avg_tree_depth, avg_tree_leaves）
- ✅ 输出包含 stability_score = mean_spearman - 2 * std_spearman
- ✅ 使用 try-finally 保证恢复原始参数

#### 函数签名：
```python
def experiment_lgbm_window_sizes(
    self,
    X: pd.DataFrame,
    y: pd.Series,
    best_params: Dict[str, Any],
    window_sizes: List[int],
    step: Optional[int] = None,
) -> pd.DataFrame
```

---

### 实验3: 因子集合对比实验 ✅
**文件**: `toollab/model_tuner.py`
**函数**: `compare_factor_sets_lgbm`
**状态**: 已重命名参数，统一CV逻辑

#### 关键修改：
- ✅ 参数重命名：`factor_sets` → `factor_strategies`
- ✅ 统一使用同一套CV切分（保证可比较性）
- ✅ 输出包含 stability_score
- ✅ 支持滑动窗口和传统TimeSeriesSplit两种模式

#### 函数签名：
```python
def compare_factor_sets_lgbm(
    self,
    factor_strategies: Dict[str, pd.DataFrame],
    y: pd.Series,
    best_params: Dict[str, Any],
) -> pd.DataFrame
```

---

### 实验4: 三模型集成实验 ✅
**文件**: `toollab/model_tuner.py`
**函数**: `experiment_lgbm_vs_mlp_ensemble`
**状态**: 已完全重写（删除LSTM，改为LightGBM+CatBoost+MLP+Ridge）

#### 关键修改：
- ✅ 完全删除PyTorch LSTM相关代码
- ✅ 新增CatBoost模型
- ✅ 使用sklearn MLPRegressor（Tabular MLP）
- ✅ 实现简单平均集成（三模型均权）
- ✅ 实现Ridge线性stacking（三模型）
- ✅ 函数签名改为 `X_train, y_train, X_val, y_val` 分离参数

#### 函数签名：
```python
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
) -> pd.DataFrame
```

#### 输出格式：
```
| model                          | spearman |
|--------------------------------|----------|
| stacking_ridge(lgbm+cat+mlp)   | 0.0456   |
| mean_ensemble(lgbm+cat+mlp)    | 0.0453   |
| lgbm                           | 0.0450   |
| catboost                       | 0.0448   |
| mlp                            | 0.0420   |
```

---

## 🔬 V7 Notebook 使用示例

### 实验1使用示例：
```python
# 准备固定的train/val切分
split_date = df_feat['date_id'].quantile(0.7)
train_mask = df_feat['date_id'] < split_date
X_train = df_feat[train_mask][FEATURES]
y_train = df_feat[train_mask][TARGET]
X_val = df_feat[~train_mask][FEATURES]
y_val = df_feat[~train_mask][TARGET]

# 基础参数（来自之前Optuna调参）
base_params = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "reg_lambda": 0.01,
    "reg_alpha": 0.01,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
}

# 执行网格搜索
tree_results = tuner.grid_search_lgbm_tree_structure(
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    base_params=base_params,
    max_depth_list=[3, 4, 5, 6],
    num_leaves_list=[8, 16, 31, 64],
    min_child_samples_list=[1, 3, 5, 10],
    min_split_gain_list=[0.0, 0.001, 0.01, 0.1]
)

print(tree_results.head(10))
```

### 实验2使用示例：
```python
window_results = tuner.experiment_lgbm_window_sizes(
    X=df_feat[FEATURES],
    y=df_feat[TARGET],
    best_params=BEST_TREE_PARAMS,  # 使用实验1的最优参数
    window_sizes=[60, 90, 120, 180],
    step=10  # 每10天重训一次
)

print(window_results)
```

### 实验3使用示例：
```python
# 准备不同因子策略
factor_strategies = {
    'fixed_all_history': df_feat[TOP_50_FACTORS + LAGGED_FEATURES],
    'top30_recent_ic': df_feat[TOP_30_FACTORS + LAGGED_FEATURES],
    'top100_all': df_feat[TOP_100_FACTORS + LAGGED_FEATURES]
}

factor_results = tuner.compare_factor_sets_lgbm(
    factor_strategies=factor_strategies,
    y=df_feat[TARGET],
    best_params=BEST_TREE_PARAMS
)

print(factor_results)
```

### 实验4使用示例：
```python
ensemble_results = tuner.experiment_lgbm_vs_mlp_ensemble(
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    lgbm_params=LGBM_PARAMS,
    catboost_params=CATBOOST_PARAMS,
    mlp_hidden_layer_sizes=(64, 32),
    mlp_alpha=1e-4,
    mlp_learning_rate_init=1e-3
)

print("\n模型性能对比:")
print(ensemble_results)
```

---

## 📊 输出示例

### 实验1输出：
```
| max_depth | num_leaves | min_child_samples | min_split_gain | spearman | avg_depth | avg_leaves |
|-----------|------------|-------------------|----------------|----------|-----------|------------|
| 5         | 31         | 3                 | 0.001          | 0.0452   | 4.23      | 28.5       |
| 6         | 31         | 3                 | 0.001          | 0.0450   | 4.67      | 29.2       |
| 5         | 16         | 3                 | 0.01           | 0.0445   | 3.89      | 15.1       |
```

### 实验2输出：
```
| train_window_size | n_windows | mean_spearman | std_spearman | stability_score | avg_tree_depth | avg_tree_leaves |
|-------------------|-----------|---------------|--------------|-----------------|----------------|-----------------|
| 90                | 850       | 0.0453        | 0.0120       | 0.0213          | 4.12           | 27.3            |
| 120               | 780       | 0.0450        | 0.0115       | 0.0220          | 4.34           | 29.1            |
| 60                | 920       | 0.0445        | 0.0135       | 0.0175          | 3.78           | 23.5            |
```

### 实验3输出：
```
| strategy            | n_windows | mean_spearman | std_spearman | stability_score |
|---------------------|-----------|---------------|--------------|-----------------|
| top100_all          | 850       | 0.0465        | 0.0118       | 0.0229          |
| fixed_all_history   | 850       | 0.0453        | 0.0120       | 0.0213          |
| top30_recent_ic     | 850       | 0.0440        | 0.0125       | 0.0190          |
```

### 实验4输出：
```
| model                          | spearman |
|--------------------------------|----------|
| stacking_ridge(lgbm+cat+mlp)   | 0.0456   |
| mean_ensemble(lgbm+cat+mlp)    | 0.0453   |
| lgbm                           | 0.0450   |
| catboost                       | 0.0448   |
| mlp                            | 0.0420   |
```

---

## ⚙️ 技术细节

### 实验1：树复杂度诊断
**目标**: 发现"树无法分裂"问题
**关键指标**:
- `avg_depth / max_depth` - 深度利用率
- `avg_leaves / num_leaves` - 叶子利用率
- 当利用率 < 0.5 时，表示树结构过于简单

### 实验2：窗口大小优化
**目标**: 找到最优训练窗口
**关键指标**:
- `mean_spearman` - 平均表现
- `stability_score` - 稳定性（mean - 2*std）
- `avg_tree_depth / avg_tree_leaves` - 不同窗口下的树复杂度变化

### 实验3：因子策略对比
**目标**: 评估因子管理策略
**典型策略**:
- **固定因子**: 使用全历史Fitness选因子
- **动态因子**: 每月用最近N天Fitness重选
- **时间加权**: 使用指数加权Fitness

### 实验4：三模型集成
**目标**: 验证集成学习价值
**模型对比**:
1. LightGBM (树模型主力)
2. CatBoost (补充树模型)
3. MLP (神经网络补充)
4. 简单平均集成 (三模型均权)
5. Ridge stacking (线性meta模型)

---

## 🚀 下一步计划

1. ✅ **ModelTuner修改完成**
2. ⏳ **创建V7 Notebook实验模块**
3. ⏳ **删除无用cells**
4. ⏳ **改造推理Pipeline为三模型集成**

---

## 📌 注意事项

1. **实验1**: train/val必须是固定时间切分，不是CV
2. **实验2**: step参数控制重训频率，建议10天
3. **实验3**: 所有策略使用相同CV切分，保证可比性
4. **实验4**: Ridge stacking是简化版，生产环境需用OOF预测

---

**文档版本**: v1.0
**创建日期**: 2025-01-XX
**状态**: ✅ ModelTuner修改完成，待Notebook集成
