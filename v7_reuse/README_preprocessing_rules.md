# 因子预处理规则使用指南

## 📦 保存的文件

- `preprocessing_rules.json` - 因子变换规则（IC翻转 + 三轮变换）
- `factor_ic_analysis.csv` - 因子IC分析结果

## 🎯 功能说明

因子预处理包含两个核心步骤：

1. **IC负值翻转** - 对IC<0的因子乘以-1，统一方向性
2. **三轮变换优化** - 依次尝试Log/Rank/3-Sigma变换，仅保留能提升Fitness的变换

预处理规则保存为JSON格式，可应用于验证集，确保训练集和验证集使用相同的变换方式。

## 📁 JSON格式说明

```json
{
  "flip_rules": [
    "A12",
    "B34",
    "C56"
  ],
  "transform_rules": {
    "A1": "log",
    "A2": "rank",
    "A3": "winsorize",
    "A4": "none"
  }
}
```

**字段说明**:
- `flip_rules`: 需要翻转的因子列表（IC<0）
- `transform_rules`: 每个因子的变换类型
  - `log`: Log1p变换（符号保持）
  - `rank`: Rank标准化（百分位）
  - `winsorize`: 3-Sigma裁剪
  - `none`: 无变换

## 🚀 使用方法

### 方法1: 加载并应用规则（推荐）

```python
import json
from toollab import apply_preprocessing_rules

# Step 1: 加载规则
with open('v7_reuse/preprocessing_rules.json', 'r') as f:
    preprocessing_rules = json.load(f)

# Step 2: 应用到验证集
val_df_transformed = apply_preprocessing_rules(val_df, preprocessing_rules)
```

### 方法2: 手动分步应用

```python
import json
import pandas as pd
from toollab.feature_preprocessor import FeaturePreprocessor

# 加载规则
with open('v7_reuse/preprocessing_rules.json', 'r') as f:
    rules = json.load(f)

val_df = val_df.copy()

# Step 1: IC翻转
for col in rules['flip_rules']:
    if col in val_df.columns:
        val_df[col] = val_df[col] * (-1)

# Step 2: 应用变换
transforms = {
    'winsorize': FeaturePreprocessor.winsorize_3sigma,
    'log': FeaturePreprocessor.log_transform,
    'rank': FeaturePreprocessor.rank_transform
}

for col, transform_name in rules['transform_rules'].items():
    if col in val_df.columns and transform_name != 'none':
        val_df[col] = transforms[transform_name](val_df[col])
```

## 📊 完整workflow示例

### 训练阶段（保存规则）

```python
# 【Notebook Cell: 因子预处理】
# 执行IC翻转 + 三轮变换，收集规则
preprocessing_rules = {
    'flip_rules': negative_ic_features,
    'transform_rules': {...}
}

# 【Notebook Cell: 保存规则】
import json
with open('v7_reuse/preprocessing_rules.json', 'w') as f:
    json.dump(preprocessing_rules, f, indent=2)
```

### 推理阶段（应用规则）

```python
# 【在线推理代码】
import json
from toollab import apply_preprocessing_rules

# 加载规则（一次性）
with open('v7_reuse/preprocessing_rules.json', 'r') as f:
    preprocessing_rules = json.load(f)

# 每个新batch应用相同规则
for new_batch in test_batches:
    # 应用预处理
    new_batch_transformed = apply_preprocessing_rules(
        new_batch,
        preprocessing_rules
    )

    # 继续特征工程 + 预测
    # ...
```

## ⚙️ 函数API

### `apply_preprocessing_rules(df, preprocessing_rules)`

对新数据应用已保存的预处理规则

**参数**:
- `df` (pd.DataFrame): 需要处理的数据
- `preprocessing_rules` (dict): 规则字典，包含 `flip_rules` 和 `transform_rules`

**返回**:
- `pd.DataFrame`: 应用变换后的数据

**输出**:
```
✅ 预处理规则已应用
   翻转特征: 23个
   变换特征: 87个
```

## 🔍 常见问题

### Q1: 为什么要保存规则而不是保存数据？
**A**: 因为验证集大小不固定，在线推理时需要动态应用变换。保存规则更灵活。

### Q2: 如果验证集缺少某些因子怎么办？
**A**: `apply_preprocessing_rules()` 会自动跳过不存在的列，不会报错。

### Q3: 三轮变换的顺序重要吗？
**A**: 是的！顺序为 Log → Rank → 3-Sigma，每一轮都基于上一轮的结果。保存规则后按相同顺序应用即可。

### Q4: 如何验证规则应用是否正确？
**A**: 可以比较训练集和验证集的因子分布：

```python
# 检查某个因子的分布
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
train['A1'].hist(bins=50, ax=ax1, alpha=0.7)
val_transformed['A1'].hist(bins=50, ax=ax2, alpha=0.7)
ax1.set_title('Train')
ax2.set_title('Validation')
plt.show()
```

## 📌 注意事项

1. **一致性**: 训练集和验证集必须使用相同的规则
2. **顺序**: 特征工程必须在应用预处理规则之后
3. **更新**: 如果重新运行因子预处理，需重新保存规则
4. **版本管理**: 建议为不同实验保存不同版本的规则文件

## 🔗 相关文件

- `toollab/utils.py:583` - `apply_preprocessing_rules()` 函数实现
- `toollab/feature_preprocessor.py` - 原始预处理类
- `hull_submission_v7_online_learning_local.ipynb` - 主Notebook
  - Cell: `factor_preprocess_cell` - 收集规则
  - Cell: `save_preprocessed_data` - 保存规则

---

**版本**: v1.0
**创建日期**: 2025-01-XX
**状态**: ✅ 已完成
