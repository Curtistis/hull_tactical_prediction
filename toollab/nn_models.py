"""
神经网络模型封装 - Tabular MLP
在现有树模型框架基础上，提供一个轻量级的全连接神经网络作为补充模型
支持sklearn和PyTorch两个版本
"""

from typing import Tuple, Optional, List
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


def create_tabular_mlp(
    hidden_layer_sizes: Tuple[int, ...] = (64, 32),
    alpha: float = 1e-4,
    learning_rate_init: float = 1e-3,
    random_state: int = 42,
) -> Pipeline:
    """
    【功能】创建一个用于表格数据的 MLP 回归模型（带标准化）

    说明：
        - 输入：与 LightGBM / CatBoost 相同的特征矩阵（横截面 + 时间混合）
        - 结构：StandardScaler + 两层全连接 ReLU
        - 使用 early_stopping 和 adaptive learning rate，减轻过拟合
        - 这个 MLP 主打"低频更新补充模型"，例如每月重训一次，再与树模型做集成

    Args:
        hidden_layer_sizes: 隐藏层维度，例如 (64, 32)
        alpha: L2 正则项，越大越保守
        learning_rate_init: 初始学习率
        random_state: 随机种子，保证结果可复现

    Returns:
        sklearn Pipeline 对象，可直接 .fit(X, y) / .predict(X)
    """
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=alpha,
        batch_size="auto",
        learning_rate="adaptive",
        learning_rate_init=learning_rate_init,
        max_iter=200,
        early_stopping=True,
        n_iter_no_change=10,
        validation_fraction=0.1,
        random_state=random_state,
        verbose=False,
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", mlp),
    ])

    return model


# ============================================================
# PyTorch MLP for GPU acceleration
# ============================================================

class TabularMLP(nn.Module):
    """
    【PyTorch版MLP】支持GPU加速的表格数据回归模型

    结构：
        Input → Linear → ReLU → Dropout → ... → Linear → Output

    特点：
        - 支持GPU训练
        - Early stopping
        - L2正则化（通过weight_decay）
        - 自动标准化（需外部StandardScaler）
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32],
        dropout: float = 0.2
    ):
        super(TabularMLP, self).__init__()

        layers = []
        prev_dim = input_dim

        # 隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze()


def train_pytorch_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    hidden_dims: List[int] = [64, 32],
    dropout: float = 0.2,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    epochs: int = 200,
    early_stopping_patience: int = 10,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    verbose: bool = False
) -> Tuple[TabularMLP, StandardScaler]:
    """
    【训练PyTorch MLP】支持GPU加速和early stopping

    Args:
        X_train: 训练特征 (n_samples, n_features)
        y_train: 训练目标 (n_samples,)
        X_val: 验证特征（可选，用于early stopping）
        y_val: 验证目标（可选）
        hidden_dims: 隐藏层维度列表
        dropout: Dropout比例
        lr: 学习率
        weight_decay: L2正则化系数
        batch_size: 批次大小
        epochs: 最大训练轮数
        early_stopping_patience: Early stopping耐心值
        device: 'cuda' 或 'cpu'
        verbose: 是否打印训练信息

    Returns:
        (model, scaler): 训练好的模型和标准化器
    """
    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 转换为Tensor
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)

    # 创建DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 验证集（如果提供）
    if X_val is not None and y_val is not None:
        X_val_scaled = scaler.transform(X_val)
        X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
        y_val_tensor = torch.FloatTensor(y_val).to(device)
    else:
        X_val_tensor = None
        y_val_tensor = None

    # 创建模型
    input_dim = X_train.shape[1]
    model = TabularMLP(input_dim, hidden_dims, dropout).to(device)

    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # 训练循环
    for epoch in range(epochs):
        # 训练模式
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_batch)

        train_loss /= len(X_train)

        # 验证（如果有验证集）
        if X_val_tensor is not None:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()

            # Early stopping检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1

            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            # Early stopping
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        else:
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}")

    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, scaler


def predict_pytorch_mlp(
    model: TabularMLP,
    scaler: StandardScaler,
    X: np.ndarray,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> np.ndarray:
    """
    【PyTorch MLP预测】

    Args:
        model: 训练好的模型
        scaler: 标准化器
        X: 输入特征
        device: 设备

    Returns:
        预测结果
    """
    model.eval()
    X_scaled = scaler.transform(X)
    X_tensor = torch.FloatTensor(X_scaled).to(device)

    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy()

    return predictions
