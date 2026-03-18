# 预处理数据复用指南

## 📦 保存的文件

- `train_preprocessed.parquet` - IC翻转 + 三轮变换后的训练集
- `test_preprocessed.parquet` - 同样处理的测试集（如有）
- `factor_ic_analysis.csv` - 因子IC分析结果

## 🚀 快速加载（跳过预处理）

### 方法1: 直接加载Parquet

```python
import pandas as pd

REUSE_DIR = '/Users/curtis/Desktop/hull_tactical_prediction/v7_reuse'

# 加载预处理后的数据
train = pd.read_parquet(f'{REUSE_DIR}/train_preprocessed.parquet')
test = pd.read_parquet(f'{REUSE_DIR}/test_preprocessed.parquet')

print(f"✅ 训练集: {train.shape}")
print(f"✅ 测试集: {test.shape}")
```

### 方法2: 使用辅助函数

```python
def load_preprocessed_data(reuse_dir='/Users/curtis/Desktop/hull_tactical_prediction/v7_reuse'):
    """
    加载预处理后的数据

    Returns:
        train, test, factor_df
    """
    import pandas as pd
    import os

    train_path = os.path.join(reuse_dir, 'train_preprocessed.parquet')
    test_path = os.path.join(reuse_dir, 'test_preprocessed.parquet')
    ic_path = os.path.join(reuse_dir, 'factor_ic_analysis.csv')

    # 加载数据
    train = pd.read_parquet(train_path)

    # 测试集（如果存在）
    if os.path.exists(test_path):
        test = pd.read_parquet(test_path)
    else:
        test = pd.DataFrame(columns=train.columns)

    # IC分析结果（如果存在）
    factor_df = None
    if os.path.exists(ic_path):
        factor_df = pd.read_csv(ic_path)

    print(f"✅ 已加载预处理数据")
    print(f"   训练集: {train.shape}")
    print(f"   测试集: {test.shape}")
    if factor_df is not None:
        print(f"   IC分析: {len(factor_df)}个因子")

    return train, test, factor_df

# 使用
train, test, factor_df = load_preprocessed_data()
```

## ⚡ 性能对比

| 操作 | 耗时 |
|------|------|
| 从CSV加载+预处理 | ~5-10分钟 |
| 从Parquet直接加载 | ~5-10秒 |

**提速约60倍！**

## 📌 注意事项

1. **数据一致性**: 确保使用相同的CONFIG配置
2. **更新时机**: 如果修改了因子预处理逻辑，需重新运行保存
3. **存储空间**: Parquet格式已压缩，约为CSV的1/3大小

## 🔄 完整workflow

```python
# 第一次运行：预处理 + 保存
# [运行因子预处理cell]
# [运行保存cell]

# 后续运行：直接加载
train, test, factor_df = load_preprocessed_data()

# 继续后续流程（因子池实验、特征工程等）
```
