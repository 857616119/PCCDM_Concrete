# 文件：train_diffusion_without_physics.py
"""
训练无物理约束的扩散模型（对照组）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import math
import os

# ==========================================
# 基础组件（与原有代码相同）
# ==========================================

def timestep_embedding(timesteps, dim, max_period=10000):
    """创建正弦时间步嵌入"""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class ResNetBlock(nn.Module):
    """简单的 ResNet 块"""
    def __init__(self, d_main, d_hidden, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_main)
        self.linear1 = nn.Linear(d_main, d_hidden)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_hidden, d_main)
        
    def forward(self, x):
        h = self.norm1(x)
        h = self.linear1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.linear2(h)
        return x + h

class ResNet(nn.Module):
    """ResNet 骨干网络"""
    def __init__(self, d_in, n_blocks, d_main, d_hidden, dropout_first, dropout_second, d_out):
        super().__init__()
        self.input_proj = nn.Linear(d_in, d_main)
        self.blocks = nn.ModuleList([
            ResNetBlock(d_main, d_hidden, dropout_second)
            for _ in range(n_blocks)
        ])
        self.norm_out = nn.LayerNorm(d_main)
        self.output_proj = nn.Linear(d_main, d_out)
        self.dropout_first = nn.Dropout(dropout_first)

    @classmethod
    def make_baseline(cls, d_in, n_blocks, d_main, d_hidden, dropout_first, dropout_second, d_out):
        return cls(d_in, n_blocks, d_main, d_hidden, dropout_first, dropout_second, d_out)

    def forward(self, x):
        h = self.input_proj(x)
        h = self.dropout_first(h)
        for block in self.blocks:
            h = block(h)
        h = self.norm_out(h)
        return self.output_proj(h)

# ==========================================
# 无物理约束的扩散模型
# ==========================================

class DiffusionModelWithoutPhysics(nn.Module):
    """无物理约束的扩散模型"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_main = config.d_main
        self.d_hidden = config.d_hidden
        self.n_blocks = config.n_blocks
        self.dropout = config.dropout
        
        # 时间步嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(self.d_main, self.d_main),
            nn.SiLU(),
            nn.Linear(self.d_main, self.d_main)
        )
        
        # 条件嵌入
        self.condition_encoder = nn.Sequential(
            nn.Linear(config.condition_dim, self.d_main),
            nn.SiLU(),
            nn.Linear(self.d_main, self.d_main)
        )
        
        # 骨干网络（无物理编码器）
        self.backbone = ResNet.make_baseline(
            d_in=self.d_main,
            n_blocks=self.n_blocks,
            d_main=self.d_main,
            d_hidden=self.d_hidden,
            dropout_first=self.dropout,
            dropout_second=self.dropout,
            d_out=config.input_dim
        )
        
        self.input_proj = nn.Linear(config.input_dim, self.d_main)
        
    def forward(self, x_noisy, t, conditions, physics_constraints=None):
        # 时间步嵌入
        t_emb = timestep_embedding(t, self.d_main)
        t_emb = self.time_embed(t_emb)
        
        # 条件嵌入
        c_emb = self.condition_encoder(conditions)
        
        # 融合上下文（无物理嵌入）
        context = t_emb + c_emb
        
        # 处理输入
        h = self.input_proj(x_noisy)
        h = h + context
        
        return self.backbone(h)

# ==========================================
# 扩散过程（无物理损失）
# ==========================================

class DiffusionProcessWithoutPhysics:
    def __init__(self, config):
        self.config = config
        self.timesteps = config.timesteps
        self.betas = torch.linspace(config.beta_start, config.beta_end, self.timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
    def p_losses(self, model, x_start, t, conditions, physics_constraints=None):
        noise = torch.randn_like(x_start)
        
        # 加噪
        sqrt_alpha = self.sqrt_alphas_cumprod.to(t.device)[t].reshape(-1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod.to(t.device)[t].reshape(-1, 1)
        x_noisy = sqrt_alpha * x_start + sqrt_one_minus_alpha * noise
        
        # 预测噪声（无物理约束）
        pred_noise = model(x_noisy, t, conditions, physics_constraints)
        
        # 仅计算MSE损失
        loss = F.mse_loss(pred_noise, noise)
        
        return loss

# ==========================================
# 数据加载与配置（与之前相同）
# ==========================================

class ConcreteDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        
        self.target_col = ['相对抗压强度']
        self.cond_cols = [
            '快中子辐照注量 n/cm2', 
            '辐照温度', 
            '水灰比', 
            '骨料水泥比_numeric', 
            'concrete_type_encoded'
        ]
        
        # 提取数据
        self.Y = self.df[self.target_col].values.astype(np.float32)
        self.X = self.df[self.cond_cols].copy()
        
        # 预处理
        self.X['快中子辐照注量 n/cm2'] = np.log1p(self.X['快中子辐照注量 n/cm2'])
        self.X = self.X.values.astype(np.float32)
        
        # 标准化
        self.y_mean = self.Y.mean(axis=0)
        self.y_std = self.Y.std(axis=0)
        self.x_mean = self.X.mean(axis=0)
        self.x_std = self.X.std(axis=0)
        
        self.Y_scaled = (self.Y - self.y_mean) / (self.y_std + 1e-8)
        self.X_scaled = (self.X - self.x_mean) / (self.x_std + 1e-8)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return self.Y_scaled[idx], self.X_scaled[idx]

class Config:
    def __init__(self):
        # 数据维度
        self.input_dim = 1      # 目标维度 (强度)
        self.condition_dim = 5  # 条件维度 (5个特征)
        
        # 网络架构
        self.d_main = 128
        self.d_hidden = 256
        self.hidden_dim = 256
        self.n_blocks = 3
        self.dropout = 0.1
        
        # 扩散参数
        self.timesteps = 500
        self.beta_start = 0.0001
        self.beta_end = 0.02
        
        # 训练参数
        self.lr = 1e-3
        self.batch_size = 32
        self.epochs = 1000

# ==========================================
# 训练主程序
# ==========================================

def train():
    # 1. 准备配置和数据
    config = Config()
    dataset = ConcreteDataset('data/processed/concrete_augmented.csv')
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    print(f"数据集大小: {len(dataset)}")
    print(f"训练无物理约束扩散模型")
    
    # 2. 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = DiffusionModelWithoutPhysics(config).to(device)
    diffusion = DiffusionProcessWithoutPhysics(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-5
    )
    
    # 3. 训练循环
    model.train()
    print("开始训练...")
    
    for epoch in range(config.epochs):
        epoch_loss = 0
        batch_count = 0
        
        for target_batch, cond_batch in dataloader:
            target_batch = target_batch.to(device)
            cond_batch = cond_batch.to(device)
            
            # 随机时间步
            t = torch.randint(0, config.timesteps, (target_batch.shape[0],), device=device).long()
            
            optimizer.zero_grad()
            
            # 计算损失（无物理约束）
            loss = diffusion.p_losses(model, target_batch, t, cond_batch)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            
        # 更新学习率
        scheduler.step()
            
        # 打印日志
        if (epoch + 1) % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            avg_loss = epoch_loss / batch_count
            print(f"Epoch {epoch+1}/{config.epochs}, Loss: {avg_loss:.6f}, LR: {current_lr:.6f}")

    # 4. 保存模型
    os.makedirs('results/models', exist_ok=True)
    torch.save(model.state_dict(), 'results/models/diffusion_without_physics.pth')
    print("模型已保存: diffusion_without_physics.pth")

if __name__ == '__main__':
    if os.path.exists('data/processed/concrete_augmented.csv'):
        train()
    else:
        print("请先生成 concrete_augmented.csv 文件")