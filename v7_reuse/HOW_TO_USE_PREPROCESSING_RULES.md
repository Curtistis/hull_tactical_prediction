# 如何使用预处理规则

## 📋 概述

本文档说明如何在 Notebook 中使用已保存的预处理规则对新数据（验证集/测试集）进行预处理。

---

## 📂 已保存的文件

在 `v7_reuse/` 目录下已生成以下文件：

```
v7_reuse/
├── preprocessing_rules.json      # 预处理规则（翻转+变换）
├── factor_ic_analysis.csv        # IC分析结果
└── train_preprocessed.parquet    # 预处理后的训练数据
```

---

## 🔧 使用方法

### 方法1: 使用 toollab 函数（推荐）

```python
import json
from toollab.utils import apply_preprocessing_rules

# 1. 加载预处理规则
with open('v7_reuse/preprocessing_rules.json', 'r') as f:
    preprocessing_rules = json.load(f)

# 2. 对新数据应用规则
df_val_preprocessed = apply_preprocessing_rules(df_val, preprocessing_rules)
df_test_preprocessed = apply_preprocessing_rules(df_test, preprocessing_rules)
```

### 方法2: 手动应用规则

如果不想导入 toollab，可以直接使用以下代码：

```python
import json
import numpy as np

# 1. 加载规则
with open('v7_reuse/preprocessing_rules.json', 'r') as f:
    rules = json.load(f)

# 2. 复制数据
df = df.copy()

# 3. 应用翻转
for col in rules['flip_rules']:
    if col in df.columns:
        df[col] = df[col] * (-1)

# 4. 定义变换函数
def winsorize_3sigma(series):
    mean_val = series.mean()
    std_val = series.std()
    upper = mean_val + 3 * std_val
    lower = mean_val - 3 * std_val
    return series.clip(lower, upper)

def log_transform(series):
    min_val = series.min()
    if min_val <= 0:
        series = series - min_val + 1e-8
    return np.log(series)

def rank_transform(series):
    return series.rank(pct=True)

transforms = {
    'winsorize': winsorize_3sigma,
    'log': log_transform,
    'rank': rank_transform
}

# 5. 应用变换
for col, transform_name in rules['transform_rules'].items():
    if col in df.columns and transform_name != 'none':
        try:
            df[col] = transforms[transform_name](df[col])
        except Exception as e:
            print(f"警告: {col} 变换失败: {e}")
```

---

## 📊 预处理规则详情

### 规则统计

- **翻转规则**: 44个特征（IC为负的因子）
- **变换规则**: 98个特征
  - `log`: 54个
  - `none`: 27个
  - `rank`: 17个

### 翻转规则示例

需要翻转的前10个特征：
```
M4, lagged_market_forward_excess_returns, lagged_forward_returns,
S2, E7, E11, E12, P8, I2, M12
```

### 变换规则分配策略

根据IC绝对值强度分配变换：

| IC绝对值范围 | 变换类型 | 说明 |
|-------------|---------|------|
| > 0.02      | none    | 高IC因子，保持原样 |
| 0.01-0.02   | rank    | 中IC因子，Rank标准化 |
| < 0.01      | log     | 低IC因子，Log变换 |

---

## ✅ 验证示例

```python
import json
from toollab.utils import apply_preprocessing_rules

# 加载规则
with open('v7_reuse/preprocessing_rules.json', 'r') as f:
    rules = json.load(f)

print(f"翻转规则: {len(rules['flip_rules'])}个")
print(f"变换规则: {len(rules['transform_rules'])}个")

# 应用到验证集
df_val_processed = apply_preprocessing_rules(df_val.copy(), rules)

# 检查翻转
print("\n验证翻转 (M4应该符号相反):")
print(f"原始: {df_val['M4'].iloc[0]:.4f}")
print(f"处理后: {df_val_processed['M4'].iloc[0]:.4f}")
```

---

## ⚠️ 注意事项

1. **保持一致性**: 训练集和验证集/测试集必须使用相同的预处理规则
2. **顺序重要**: 先翻转，后变换
3. **缺失特征**: 如果某个特征不存在于新数据中，会自动跳过
4. **变换失败**: 如果变换失败（如log变换遇到负值），会打印警告并跳过

---

## 📝 Notebook 集成示例

### 在实验流程中使用

```python
# ============================================================
# 【验证集预处理】应用训练集规则
# ============================================================
import json
from toollab.utils import apply_preprocessing_rules

# 加载规则
with open('v7_reuse/preprocessing_rules.json', 'r') as f:
    preprocessing_rules = json.load(f)

print("对验证集应用预处理规则...")
df_val = apply_preprocessing_rules(df_val, preprocessing_rules)

# ============================================================
# 【实验1】树结构网格搜索
# ============================================================
# 使用预处理后的数据进行实验...
```

---

**创建日期**: 2025-12-15
**状态**: ✅ 规则已保存，可直接使用
