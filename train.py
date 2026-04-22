# -*- coding: utf-8 -*-
"""
PCCDM Training Script
包含无约束扩散模型和物理约束扩散模型的完整训练流程
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import os

# ==========================================
# 1. 基础配置与网络结构 (对齐验证代码)
# ==========================================
class Config:
    def __init__(self):
        self.input_dim = 1
        self.condition_dim = 5
        self.d_main = 64      # 轻量化：防过拟合
        self.d_hidden = 128
        self.hidden_dim = 128
        self.n_blocks = 2     # 轻量化：防过拟合
        self.dropout = 0.2
        self.timesteps = 100  # 表格数据 100 步足够
        self.beta_start = 0.0001
        self.beta_end = 0.02
        
        # 训练超参数
        self.epochs = 500
        self.batch_size = 128
        self.lr = 1e-3
        self.physics_weight = 0.1 # 物理损失的权重 λ

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class ResNetBlock(nn.Module):
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
    def __init__(self, d_in, n_blocks, d_main, d_hidden, dropout_first, dropout_second, d_out):
        super().__init__()
        self.input_proj = nn.Linear(d_in, d_main)
        self.blocks = nn.ModuleList([ResNetBlock(d_main, d_hidden, dropout_second) for _ in range(n_blocks)])
        self.norm_out = nn.LayerNorm(d_main)
        self.output_proj = nn.Linear(d_main, d_out)
        self.dropout_first = nn.Dropout(dropout_first)

    @classmethod
    def make_baseline(cls, d_in, n_blocks, d_main, d_hidden, dropout_first, dropout_second, d_out):
        return cls(d_in, n_blocks, d_main, d_hidden, dropout_first, dropout_second, d_out)

    def forward(self, x):
        h = self.input_proj(x)
        h = self.dropout_first(h)
        for block in self.blocks: h = block(h)
        h = self.norm_out(h)
        return self.output_proj(h)

class PhysicsAwareResNetModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_main = config.d_main
        self.time_embed = nn.Sequential(nn.Linear(self.d_main, self.d_main), nn.SiLU(), nn.Linear(self.d_main, self.d_main))
        self.condition_encoder = nn.Sequential(nn.Linear(config.condition_dim, self.d_main), nn.SiLU(), nn.Linear(self.d_main, self.d_main))
        self.backbone = ResNet.make_baseline(self.d_main, config.n_blocks, self.d_main, config.d_hidden, config.dropout, config.dropout, config.input_dim)
        self.input_proj = nn.Linear(config.input_dim, self.d_main)
        
    def forward(self, x_noisy, t, conditions):
        t_emb = self.time_embed(timestep_embedding(t, self.d_main))
        c_emb = self.condition_encoder(conditions)
        return self.backbone(self.input_proj(x_noisy) + t_emb + c_emb)

# ==========================================
# 2. 数据处理与加载
# ==========================================
class ConcreteDataset(Dataset):
    def __init__(self, data_path):
        df = pd.read_csv(data_path)
        
        # 1. 基础清理
        target_col = '相对抗压强度'
        cond_cols = ['快中子辐照注量 n/cm2', '辐照温度', '水灰比', '骨料水泥比_numeric', 'concrete_type_encoded']
        
        for col in [target_col, '水灰比', '骨料水泥比_numeric']:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]
            
        self.Y_raw = df[[target_col]].values.astype(np.float32)
        X_df = df[cond_cols].copy()
        X_df['快中子辐照注量 n/cm2'] = np.log1p(X_df['快中子辐照注量 n/cm2'])
        self.X_raw = X_df.values.astype(np.float32)
        
        self.y_mean, self.y_std = self.Y_raw.mean(axis=0), self.Y_raw.std(axis=0)
        self.x_mean, self.x_std = self.X_raw.mean(axis=0), self.X_raw.std(axis=0)
        
        self.Y_scaled = (self.Y_raw - self.y_mean) / (self.y_std + 1e-8)
        self.X_scaled = (self.X_raw - self.x_mean) / (self.x_std + 1e-8)

    def __len__(self): return len(self.Y_scaled)
    def __getitem__(self, idx): return torch.tensor(self.Y_scaled[idx]), torch.tensor(self.X_scaled[idx])

# ==========================================
# 3. 物理约束损失函数 (PyTorch 可微版)
# ==========================================
def compute_physics_loss(x0_hat, cond, y_mean, y_std, x_mean, x_std):
    """可反向传播的物理残差函数"""
    device = x0_hat.device
    y_m, y_s = torch.tensor(y_mean, device=device), torch.tensor(y_std, device=device)
    x_m, x_s = torch.tensor(x_mean, device=device), torch.tensor(x_std, device=device)
    
    # 反归一化到物理量纲
    strength = (x0_hat * y_s + y_m).squeeze()
    wc_ratio = cond[:, 2] * x_s[2] + x_m[2]
    fluence = cond[:, 0] * x_s[0] + x_m[0]
    
    # 1. 绝对边界约束
    bounds_loss = torch.mean(torch.relu(0.1 - strength)**2 + torch.relu(strength - 2.0)**2)
    
    # 2. 单调性物理对抗约束 (使用 Batch 内随机配对比较)
    batch_size = strength.shape[0]
    if batch_size > 1:
        idx = torch.randperm(batch_size) # 打乱得到配对样本
        dw = wc_ratio - wc_ratio[idx]
        ds = strength - strength[idx]
        df = fluence - fluence[idx]
        
        # 惩罚：W/C 增加但强度反常增加
        wc_violation = torch.where((dw > 0.05) & (ds > 0), ds**2, torch.zeros_like(ds))
        # 惩罚：注量增加但强度反常增加
        fluence_violation = torch.where((df > 0.5) & (ds > 0), ds**2, torch.zeros_like(ds))
        
        monotone_loss = torch.mean(wc_violation) + torch.mean(fluence_violation)
    else:
        monotone_loss = 0.0
        
    return bounds_loss + monotone_loss

# ==========================================
# 4. 训练主循环
# ==========================================
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    config = Config()
    dataset = ConcreteDataset('data/processed/concrete_augmented.csv')
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    
    # 预计算扩散常数
    betas = torch.linspace(config.beta_start, config.beta_end, config.timesteps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    os.makedirs('results/models', exist_ok=True)
    
    # ----------------------------------------
    # 任务 A: 训练无物理约束的基线扩散模型
    # ----------------------------------------
    print("\n" + "="*50)
    print("⏳ 正在训练：无约束扩散模型 (Unconstrained DDPM)...")
    model_no_phys = PhysicsAwareResNetModel(config).to(device)
    optimizer_no_phys = optim.Adam(model_no_phys.parameters(), lr=config.lr)
    
    for epoch in range(config.epochs):
        total_loss = 0
        for y, x_cond in dataloader:
            y, x_cond = y.to(device), x_cond.to(device)
            optimizer_no_phys.zero_grad()
            
            t = torch.randint(0, config.timesteps, (y.shape[0],), device=device).long()
            noise = torch.randn_like(y)
            alpha_t = alphas_cumprod[t].view(-1, 1)
            
            # 前向加噪
            y_noisy = torch.sqrt(alpha_t) * y + torch.sqrt(1 - alpha_t) * noise
            # 预测噪声
            noise_pred = model_no_phys(y_noisy, t, x_cond)
            
            loss = nn.MSELoss()(noise_pred, noise)
            loss.backward()
            optimizer_no_phys.step()
            total_loss += loss.item()
            
        if (epoch+1) % 100 == 0:
            print(f"  Epoch {epoch+1}/{config.epochs} | MSE Loss: {total_loss/len(dataloader):.4f}")
            
    torch.save(model_no_phys.state_dict(), 'results/models/diffusion_without_physics.pth')
    print("✅ 无约束模型保存完毕！")

    # ----------------------------------------
    # 任务 B: 训练 PCCDM (含物理约束损失)
    # ----------------------------------------
    print("\n" + "="*50)
    print("⏳ 正在训练：物理约束扩散模型 (PCCDM)...")
    model_pccdm = PhysicsAwareResNetModel(config).to(device)
    optimizer_pccdm = optim.Adam(model_pccdm.parameters(), lr=config.lr)
    
    for epoch in range(config.epochs):
        total_mse, total_phys = 0, 0
        for y, x_cond in dataloader:
            y, x_cond = y.to(device), x_cond.to(device)
            optimizer_pccdm.zero_grad()
            
            t = torch.randint(0, config.timesteps, (y.shape[0],), device=device).long()
            noise = torch.randn_like(y)
            alpha_t = alphas_cumprod[t].view(-1, 1)
            
            y_noisy = torch.sqrt(alpha_t) * y + torch.sqrt(1 - alpha_t) * noise
            noise_pred = model_pccdm(y_noisy, t, x_cond)
            
            # 1. 统计保真度损失 (生成质量)
            mse_loss = nn.MSELoss()(noise_pred, noise)
            
            # 2. 核心：估计干净数据 x0_hat 并计算物理损失 (消除詹森间隙的关键)
            x0_hat = (y_noisy - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            phys_loss = compute_physics_loss(x0_hat, x_cond, dataset.y_mean, dataset.y_std, dataset.x_mean, dataset.x_std)
            
            # 联合损失函数
            loss = mse_loss + config.physics_weight * phys_loss
            loss.backward()
            optimizer_pccdm.step()
            
            total_mse += mse_loss.item()
            total_phys += phys_loss.item()
            
        if (epoch+1) % 100 == 0:
            print(f"  Epoch {epoch+1}/{config.epochs} | MSE: {total_mse/len(dataloader):.4f} | Phys_Loss: {total_phys/len(dataloader):.4f}")
            
    torch.save(model_pccdm.state_dict(), 'results/models/pccdm_augmented_model.pth')
    print("✅ PCCDM 模型保存完毕！")

if __name__ == '__main__':
    train()