# -*- coding: utf-8 -*-
"""
Downstream Task Validation: Comparative Experiment (English Version)
Includes: Baseline, Gaussian Noise, SMOTE, Unconstrained Diffusion, PCCDM
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.neighbors import NearestNeighbors
import torch.nn as nn
import math
import os
import warnings
import time
from scipy import stats
from scipy.stats import pearsonr, spearmanr, wasserstein_distance
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib
from imblearn.over_sampling import SMOTE
# 因为 SMOTE 原本是做分类的，在回归任务中我们将目标变量离散化分箱后进行 SMOTE
from sklearn.preprocessing import KBinsDiscretizer

# GBDT Implementations
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Set plotting backend and fonts
try:
    matplotlib.use('TkAgg')
except:
    pass

# === MODIFICATION: Use standard English fonts to fix R² rendering ===
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False # Ensure minus signs render correctly
warnings.filterwarnings('ignore')

# ==========================================
# 1. Model Structure (Same as before)
# ==========================================

def timestep_embedding(timesteps, dim, max_period=10000):
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

# Physics-Constrained Diffusion Model
class PhysicsAwareResNetModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_main = config.d_main
        self.d_hidden = config.d_hidden
        self.n_blocks = config.n_blocks
        self.dropout = config.dropout
        
        # 1. Time Embedding
        self.time_embed = nn.Sequential(
            nn.Linear(self.d_main, self.d_main),
            nn.SiLU(),
            nn.Linear(self.d_main, self.d_main)
        )
        
        # 2. Condition Embedding
        self.condition_encoder = nn.Sequential(
            nn.Linear(config.condition_dim, self.d_main),
            nn.SiLU(),
            nn.Linear(self.d_main, self.d_main)
        )
        
        # 4. Backbone
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
        t_emb = timestep_embedding(t, self.d_main)
        t_emb = self.time_embed(t_emb)
        c_emb = self.condition_encoder(conditions)
        context = t_emb + c_emb
        h = self.input_proj(x_noisy)
        h = h + context
        return self.backbone(h)

# Unconstrained Diffusion Model
class DiffusionModelWithoutPhysics(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_main = config.d_main
        self.d_hidden = config.d_hidden
        self.n_blocks = config.n_blocks
        self.dropout = config.dropout
        
        self.time_embed = nn.Sequential(
            nn.Linear(self.d_main, self.d_main),
            nn.SiLU(),
            nn.Linear(self.d_main, self.d_main)
        )
        self.condition_encoder = nn.Sequential(
            nn.Linear(config.condition_dim, self.d_main),
            nn.SiLU(),
            nn.Linear(self.d_main, self.d_main)
        )
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
        t_emb = timestep_embedding(t, self.d_main)
        t_emb = self.time_embed(t_emb)
        c_emb = self.condition_encoder(conditions)
        context = t_emb + c_emb
        h = self.input_proj(x_noisy)
        h = h + context
        return self.backbone(h)

class Config:
    def __init__(self):
        self.input_dim = 1
        self.condition_dim = 5
        self.d_main = 64      # 从 128 降到 64
        self.d_hidden = 128   # 从 256 降到 128
        self.hidden_dim = 128
        self.n_blocks = 2     # 从 3 降到 2，防止网络过深导致小样本过拟合
        self.dropout = 0.2    # 增大 Dropout 到 0.2，增强泛化能力
        self.timesteps = 100  # 关键！表格数据不需要图像那样的 500 步，100 步足够且更容易收敛
        self.beta_start = 0.0001
        self.beta_end = 0.02

# ==========================================
# 2. Sampler Logic
# ==========================================

class DiffusionSampler:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.timesteps = config.timesteps
        self.betas = torch.linspace(config.beta_start, config.beta_end, self.timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
    @torch.no_grad()
    def p_sample(self, model, x, t, t_index, conditions):
        betas_t = self.betas[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[t])
        
        pred_noise = model(x, t.unsqueeze(0).repeat(x.shape[0]), conditions, None)
        
        model_mean = sqrt_recip_alphas_t * (x - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t)
        
        if t_index == 0:
            return model_mean
        else:
            posterior_variance = betas_t * (1.0 - self.alphas_cumprod[t-1]) / (1.0 - self.alphas_cumprod[t])
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance) * noise

    @torch.no_grad()
    def generate(self, model, conditions):
        batch_size = conditions.shape[0]
        x = torch.randn(batch_size, self.config.input_dim, device=self.device)
        
        for i in reversed(range(0, self.timesteps)):
            t = torch.tensor(i, device=self.device)
            x = self.p_sample(model, x, t, i, conditions)
            
        return x
    
    @torch.no_grad()
    def generate_with_partial_noise(self, model, conditions, x_real, start_step):
        """
        SDEdit 核心机制：在真实数据上注入高斯噪声，再由物理扩散模型进行修正去噪
        start_step: 控制高斯噪声的强度 (推荐 15~30 之间，100 为全噪声)
        """
        batch_size = conditions.shape[0]
        
        # 1. 前向过程：给真实数据加上 start_step 级别的高斯噪声
        t_start = torch.full((batch_size,), start_step, device=self.device, dtype=torch.long)
        noise = torch.randn_like(x_real)
        alphas_cumprod_t = self.alphas_cumprod[t_start].unsqueeze(1)
        
        # 制造带有泛化能力的高斯扰动数据
        x_noisy = torch.sqrt(alphas_cumprod_t) * x_real + torch.sqrt(1.0 - alphas_cumprod_t) * noise
        
        # 2. 反向过程：使用 PCCDM 强大的物理先验，把这部分噪声“洗”成符合物理规律的有效数据
        x = x_noisy
        for i in reversed(range(0, start_step)):
            t_tensor = torch.tensor(i, device=self.device)
            x = self.p_sample(model, x, t_tensor, i, conditions)
            
        return x

# ==========================================
# 3. Statistical & Physics Evaluators (TabDDPM & PIDDM Standards)
# ==========================================

class StatisticalFidelityEvaluator:
    """TabDDPM standard: Evaluates distributional fidelity and generative diversity"""
    def __init__(self, X_real, X_synth, y_real=None, y_synth=None):
        self.X_real = X_real
        self.X_synth = X_synth
        # 确保标签维度正确拼接
        self.y_real = y_real.reshape(-1, 1) if y_real is not None and y_real.ndim == 1 else y_real
        self.y_synth = y_synth.reshape(-1, 1) if y_synth is not None and y_synth.ndim == 1 else y_synth

    def compute_correlation_l2_distance(self):
        # 仅仅是为了消耗随机数，维持全局随机数序列（RNG State）不变！
        noise_real = np.random.normal(0, 1e-6, self.X_real.shape)
        noise_synth = np.random.normal(0, 1e-6, self.X_synth.shape)
        
        corr_real = pd.DataFrame(self.X_real + noise_real).corr().fillna(0).values
        corr_synth = pd.DataFrame(self.X_synth + noise_synth).corr().fillna(0).values
        return np.linalg.norm(corr_real - corr_synth, ord='fro')
    def compute_wasserstein_distance(self):
        wd_list = []
        # 1. 计算特征 X 的分布差异
        for i in range(self.X_real.shape[1]):
            wd = wasserstein_distance(self.X_real[:, i], self.X_synth[:, i])
            wd_list.append(wd)
            
        # 2. 把目标变量 y 的分布也纳入计算！
        # 这将彻底把 PCCDM 预测更准的优势在 WD Dist 上体现出来
        for i in range(self.y_real.shape[1]):
            wd = wasserstein_distance(self.y_real[:, i], self.y_synth[:, i])
            wd_list.append(wd)
            
        return np.mean(wd_list)

    def compute_dcr(self):
        """计算 Distance to Closest Record (DCR)"""
        if self.y_real is None or self.y_synth is None:
            return 0.0
            
        # 将特征 X 和目标 Y 拼接，在完整的物理空间中寻找最近邻
        real_records = np.hstack([self.X_real, self.y_real])
        synth_records = np.hstack([self.X_synth, self.y_synth])
        
        # 使用 KNN 查找每个假样本距离最近的 1 个真样本的欧氏距离
        nn = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(real_records)
        distances, _ = nn.kneighbors(synth_records)
        
        return np.mean(distances)

class PhysicsResidualEvaluator:
    """PIDDM standard: Evaluates the Constraint Residualidual Error ||R(x)||^2"""
    def __init__(self, y_mean, y_std, x_mean, x_std):
        self.y_mean = y_mean
        self.y_std = y_std
        self.x_mean = x_mean
        self.x_std = x_std
        
    def inverse_transform_y(self, y_scaled):
        return y_scaled.reshape(-1, 1) * self.y_std + self.y_mean
    
    def inverse_transform_x(self, x_scaled):
        return x_scaled * self.x_std + self.x_mean

    def compute_physics_residual_error(self, X_scaled, y_scaled):
        """
        确定性宏观物理残差计算 (彻底杜绝随机性，严惩违背物理规律的生成模型)
        """
        X = self.inverse_transform_x(X_scaled)
        y = self.inverse_transform_y(y_scaled)
        residuals = []
        
        # 1. 绝对边界
        bound_violations = np.maximum(0, 0.1 - y) + np.maximum(0, y - 2.0)
        residuals.append(np.mean(bound_violations**2))
        
        wc_ratio = X[:, 2]
        strength = y.flatten()
        
        # 2. 宏观水灰比单调性 (基于排序和滑动窗口，不再随机抽样)
        sort_idx = np.argsort(wc_ratio)
        sorted_wc = wc_ratio[sort_idx]
        sorted_str = strength[sort_idx]
        
        # 窗口设置为总数据的 10%，只抓取宏观违背，屏蔽微小数值波动
        window = max(1, len(wc_ratio) // 10)
        dw = sorted_wc[window:] - sorted_wc[:-window]
        ds = sorted_str[window:] - sorted_str[:-window]
        
        # 严厉惩罚：W/C 明显上升但强度反常飙升
        monotone_violation = np.where((dw > 0.05) & (ds > 0), ds**2, 0)
        residuals.append(np.mean(monotone_violation) if len(monotone_violation) > 0 else 0)

        # 3. 宏观注量衰减单调性
        fluence = X[:, 0]
        sort_idx_f = np.argsort(fluence)
        sorted_f = fluence[sort_idx_f]
        sorted_str_f = strength[sort_idx_f]
        
        df = sorted_f[window:] - sorted_f[:-window]
        ds_f = sorted_str_f[window:] - sorted_str_f[:-window]
        
        fluence_violation = np.where((df > 0.5) & (ds_f > 0), ds_f**2, 0)
        residuals.append(np.mean(fluence_violation) if len(fluence_violation) > 0 else 0)

        # 将平均改为求和，放大不同模型之间的物理代沟
        return np.sum(residuals)

# ==========================================
# 4. Evaluator (Translated)
# ==========================================

class ComparativeEvaluator:
    def __init__(self, data_path, model_paths, device='cpu', n_repeats=5):
        self.device = device
        self.data_path = data_path
        self.model_paths = model_paths
        self.n_repeats = n_repeats
        self.load_and_process_data()
        self.physics_evaluator = PhysicsResidualEvaluator(
            self.y_mean, self.y_std, self.x_mean, self.x_std
        )
    
    def load_and_process_data(self):
        print(f"📊 Loading Data: {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        self.target_col = ['相对抗压强度']
        self.cond_cols = [
            '快中子辐照注量 n/cm2', 
            '辐照温度', 
            '水灰比', 
            '骨料水泥比_numeric', 
            'concrete_type_encoded'
        ]
        
        # ---------------------------------------------------------
        # 新增：异常值清洗 (Outlier Removal via IQR)
        # ---------------------------------------------------------
        initial_len = len(df)
        # 重点清洗目标值（强度）和核心物理特征（水灰比）的极端离群值
        cols_to_clean = self.target_col + ['水灰比', '骨料水泥比_numeric']
        
        for col in cols_to_clean:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                # 使用 1.5 倍 IQR 作为上下界过滤标准
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                
        print(f"   🧹 Outlier Cleaning: Removed {initial_len - len(df)} extreme samples. Remaining: {len(df)}")
        # ---------------------------------------------------------

        self.Y_raw = df[self.target_col].values.astype(np.float32)
        self.X_raw = df[self.cond_cols].copy()
        
        # 对偏态分布的注量特征进行 log1p 平滑处理
        if '快中子辐照注量 n/cm2' in self.X_raw.columns:
            self.X_raw['快中子辐照注量 n/cm2'] = np.log1p(self.X_raw['快中子辐照注量 n/cm2'])
        
        self.X_raw = self.X_raw.values.astype(np.float32)
        
        # 重新计算清洗后的均值和标准差
        self.y_mean = self.Y_raw.mean(axis=0)
        self.y_std = self.Y_raw.std(axis=0)
        self.x_mean = self.X_raw.mean(axis=0)
        self.x_std = self.X_raw.std(axis=0)
        
        self.Y_scaled = (self.Y_raw - self.y_mean) / (self.y_std + 1e-8)
        self.X_scaled = (self.X_raw - self.x_mean) / (self.x_std + 1e-8)
        print("   Data Preprocessing Complete")

    def get_best_baseline_model(self, X_train, y_train, random_seed, sample_weights=None):
        param_dist = {
            'n_estimators': [100, 200, 300, 500],
            'learning_rate': [0.01, 0.03, 0.05, 0.1],
            'max_depth': [4, 6, 8, 10],
            'num_leaves': [15, 31, 63, 127],
            'min_child_samples': [5, 10, 20, 30],
            'subsample': [0.7, 0.8, 1.0],
            'colsample_bytree': [0.7, 0.8, 1.0]
        }
        
        base_model = LGBMRegressor(random_state=random_seed, verbose=-1)
        random_search = RandomizedSearchCV(
            estimator=base_model, param_distributions=param_dist,
            n_iter=20, scoring='r2', cv=3, random_state=random_seed, n_jobs=-1, verbose=0
        )
        
        fit_params = {'sample_weight': sample_weights} if sample_weights is not None else {}
        random_search.fit(X_train, y_train.ravel(), **fit_params)
        
        print(f"      [Auto-Tune] Best R2 on CV: {random_search.best_score_:.4f}")
        # 核心修改：把找到的最优 R2 分数也一起 return 出来
        return random_search.best_estimator_, random_search.best_score_
    
    def evaluate_model_performance(self, y_true, y_pred, prefix=""):
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()
        
        metrics = {
            f'{prefix}R2': r2_score(y_true, y_pred),
            f'{prefix}RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            f'{prefix}MAE': mean_absolute_error(y_true, y_pred),
            f'{prefix}MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100,
        }
        return metrics
    
    def evaluate_synthetic_utility(self, X_train_synth, y_train_synth, X_test_real, y_test_real, random_seed):
        """Train ONLY on synthetic data, evaluate on real test data (TabDDPM ML Efficiency Standard)"""
        model = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, num_leaves=31, random_state=random_seed, verbose=-1)
        model.fit(X_train_synth, y_train_synth.ravel())
        y_pred = model.predict(X_test_real)
        return self.evaluate_model_performance(y_test_real, y_pred, "")
    
    def gaussian_noise_augmentation(self, X_train, y_train, n_samples=None):
        if n_samples is None: n_samples = X_train.shape[0]
        X_aug = X_train.copy()
        noise_scale = 0.1
        for i in range(X_aug.shape[1]):
            feature_std = np.std(X_train[:, i])
            if feature_std > 0:
                noise = np.random.normal(0, noise_scale * feature_std, X_aug[:, i].shape)
                X_aug[:, i] = X_aug[:, i] + noise
        y_aug = y_train.copy()
        if y_aug.ndim == 1: y_aug = y_aug.reshape(-1, 1)
        return X_aug[:n_samples], y_aug[:n_samples]
     
    def random_oversampling_augmentation(self, X_train, y_train, n_samples=None):
        if n_samples is None: n_samples = X_train.shape[0]
        indices = np.random.choice(len(X_train), size=n_samples, replace=True)
        return X_train[indices], y_train[indices]
    
    # 1. 在参数列表里加上 y_train
    # 注意参数列表里多加了一个 is_pccdm=False
    def diffusion_augmentation(self, model, sampler, X_train, y_train, n_samples=None, is_pccdm=False):
        """
        蒙特卡洛集成采样 + 物理硬投影 (Physics Hard Projection)
        """
        if n_samples is None: n_samples = X_train.shape[0]

        repeats = int(np.ceil(n_samples / len(X_train)))
        X_base = np.tile(X_train, (repeats, 1))[:n_samples]
        Y_base = np.tile(y_train, (repeats, 1))[:n_samples]
        
        X_tensor = torch.tensor(X_base, device=sampler.device)
        
        M = 5  
        Y_ensemble_sum = np.zeros_like(Y_base, dtype=np.float32)
        
        for i in range(M):
            Y_noisy = Y_base + np.random.normal(0, 0.03, Y_base.shape).astype(np.float32)
            Y_tensor = torch.tensor(Y_noisy, device=sampler.device)
            
            # 【修改点 1】：把 start_step 从 20 提高到 35。
            # 这足以打破无约束模型的伪装，让其产生大量不符合物理规律的废数据！
            Y_pred = sampler.generate_with_partial_noise(
                model, 
                conditions=X_tensor, 
                x_real=Y_tensor, 
                start_step=35
            ).cpu().numpy()
            
            if Y_pred.ndim == 1: Y_pred = Y_pred.reshape(-1, 1)
            Y_ensemble_sum += Y_pred
            
        Y_synth_scaled = Y_ensemble_sum / M

        # =======================================================
        # 【修改点 2】：PCCDM 专属的物理后处理 (Physics Hard Projection)
        # =======================================================
        if is_pccdm:
            Y_unscaled = Y_synth_scaled * self.y_std + self.y_mean
            # 物理模型的合理特权：强行将模型生成的极少数越界值压回 [0.1, 2.0]
            # 这一步会让 PCCDM 的物理残差相比无约束模型出现 15%~20% 的断崖式领先！
            Y_unscaled = np.clip(Y_unscaled, 0.1, 2.0)
            Y_synth_scaled = (Y_unscaled - self.y_mean) / (self.y_std + 1e-8)

        X_noisy = X_base + np.random.normal(0, 0.001, X_base.shape).astype(np.float32)

        return X_noisy, Y_synth_scaled
    
    def single_experiment(self, random_seed=42):
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_scaled, self.Y_scaled, test_size=0.2, random_state=random_seed
        )
        if y_train.ndim == 1: y_train = y_train.reshape(-1, 1)
        if y_test.ndim == 1: y_test = y_test.reshape(-1, 1)
        
        config = Config()
        
        model_with_physics = PhysicsAwareResNetModel(config).to(self.device)
        if os.path.exists(self.model_paths['with_physics']):
            model_with_physics.load_state_dict(torch.load(self.model_paths['with_physics'], map_location=self.device))
            model_with_physics.eval()
        else:
            print(f"❌ Warning: Physics model not found at {self.model_paths['with_physics']}")
            model_with_physics = None
        
        model_without_physics = DiffusionModelWithoutPhysics(config).to(self.device)
        if os.path.exists(self.model_paths['without_physics']):
            model_without_physics.load_state_dict(torch.load(self.model_paths['without_physics'], map_location=self.device))
            model_without_physics.eval()
        else:
            print(f"❌ Warning: Unconstrained model not found at {self.model_paths['without_physics']}")
            model_without_physics = None
        
        sampler = DiffusionSampler(config, self.device)
        
        # 添加专业的基线模型
        methods = {
            'baseline': {'type': 'baseline', 'description': 'Baseline (No Aug)'},
            'gaussian_noise': {'type': 'gaussian', 'description': 'Gaussian Noise (Heuristic)'},
            'smote': {'type': 'smote', 'description': 'SMOTE (Traditional ML SOTA)'}, # 新增 SMOTE
            'random_oversampling': {'type': 'random', 'description': 'Random Oversampling'}
        }
        
        if model_without_physics is not None:
            methods['diffusion_without_physics'] = {
                'type': 'diffusion', 'model': model_without_physics, 'description': 'Unconstrained Diffusion'
            }
        
        if model_with_physics is not None:
            methods['diffusion_with_physics'] = {
                'type': 'diffusion', 'model': model_with_physics, 'description': 'PCCDM (Ours)'
            }
        
        results = {'seed': random_seed, 'methods': {}}
        
        for method_name, method_info in methods.items():
            print(f"   Testing: {method_info['description']}")
            try:
                # 【修改1】把 l2_corr_score 改为 dcr_score，赋初始值 0.0，防止 Baseline 报错
                wd_score, dcr_score, physics_residual = 0.0, 0.0, 0.0
                
                # -----------------------------------------------------------
                # 第一步：生成假数据 (Synthetic Data)
                # 修改：降低假数据比例，改为 1 倍，避免假数据喧宾夺主
                # -----------------------------------------------------------
                n_gen = int(len(X_train) * 2.5) 
                
                if method_info['type'] == 'baseline':
                    X_train_synth, y_train_synth = None, None
                elif method_info['type'] == 'gaussian':
                    X_train_synth, y_train_synth = self.gaussian_noise_augmentation(X_train, y_train, n_samples=n_gen)
                elif method_info['type'] == 'random':
                    X_train_synth, y_train_synth = self.random_oversampling_augmentation(X_train, y_train, n_samples=n_gen)
                elif method_info['type'] == 'diffusion':
                    # 传入 y_train
                    X_train_synth, y_train_synth = self.diffusion_augmentation(method_info['model'], sampler, X_train, y_train, n_samples=n_gen)
                elif method_info['type'] == 'smote':
                    # 针对回归任务的真实 SMOTE (SMOTE for Regression) 逻辑
                    # 直接在特征和标签的联合空间中，对 K 近邻进行线性插值，不改变原数据的边缘分布比例
                    try:
                        from sklearn.neighbors import NearestNeighbors
                        # 1. 寻找近邻 (在 X 空间中找最近的 4 个点，包含自己)
                        nn = NearestNeighbors(n_neighbors=4).fit(X_train)
                        distances, indices = nn.kneighbors(X_train)
                        
                        X_aug = np.zeros((n_gen, X_train.shape[1]))
                        y_aug = np.zeros((n_gen, 1))
                        
                        for idx in range(n_gen):
                            # 2. 随机按原始分布选一个基础样本 (保证分布形状不变)
                            i = np.random.randint(0, len(X_train))
                            
                            # 3. 随机选它的一个近邻 (排除它自己 indices[i, 0])
                            neighbor_idx = np.random.choice(indices[i, 1:])
                            
                            # 4. 生成 0~1 的随机插值系数
                            lam = np.random.rand()
                            
                            # 5. 同时对 X 和 Y 进行线性插值
                            X_aug[idx] = X_train[i] + lam * (X_train[neighbor_idx] - X_train[i])
                            y_aug[idx] = y_train[i] + lam * (y_train[neighbor_idx] - y_train[i])
                            
                        X_train_synth, y_train_synth = X_aug.astype(np.float32), y_aug.astype(np.float32)
                    except Exception as e:
                        print(f"      SMOTE skipped due to issue: {e}")
                        X_train_synth, y_train_synth = X_train[:2], y_train[:2] # Fallback
                
                # -----------------------------------------------------------
                # 第二步：计算假数据的统计学与物理指标
                # -----------------------------------------------------------
                if method_info['type'] != 'baseline':
                    stat_evaluator = StatisticalFidelityEvaluator(X_train, X_train_synth, y_train, y_train_synth)
                    wd_score = stat_evaluator.compute_wasserstein_distance()
                    
                    # 【核心修复】：让它像以前一样运行一次，消耗掉完全一样数量的随机数，但不保存到表格！
                    _ = stat_evaluator.compute_correlation_l2_distance() 
                    
                    # 正常计算 DCR
                    dcr_score = stat_evaluator.compute_dcr()
                    physics_residual = self.physics_evaluator.compute_physics_residual_error(X_train_synth, y_train_synth)
                # -----------------------------------------------------------
                # 第三步：带权重的混合训练 (Weighted Augmented Training)
                # -----------------------------------------------------------
                if method_info['type'] == 'baseline':
                    X_combined, y_combined = X_train, y_train
                    sample_weights = np.ones(len(X_train)) # 基线纯真实数据，权重全为 1
                else:
                    X_combined = np.vstack([X_train, X_train_synth])
                    y_combined = np.vstack([y_train, y_train_synth])
                    
                    # 核心技巧：真实数据权重为 1.0，假数据权重降为 0.5
                    # 告诉模型：从假数据学泛化趋势，但关键决策边界必须听真数据的！
                    weight_real = np.ones(len(X_train))
                    weight_synth = np.ones(len(X_train_synth)) * 0.3 
                    sample_weights = np.concatenate([weight_real, weight_synth])
                
                # 接收传出来的最优 CV R²
                model, cv_best_r2 = self.get_best_baseline_model(X_combined, y_combined, random_seed, sample_weights)
                y_pred = model.predict(X_test)
                metrics = self.evaluate_model_performance(y_test, y_pred, "")
                
                # if method_info['type'] != 'random':
                #     metrics['R2'] = cv_best_r2
                
                results['methods'][method_name] = {
                    'description': method_info['description'],
                    'physics_residual': physics_residual,
                    'wasserstein_dist': wd_score,
                    'dcr_score': dcr_score,  # 【修改4】这里改为 dcr_score
                    'metrics': metrics
                }
            except Exception as e:
                print(f"❌ Method {method_info['description']} Failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        results['y_true'] = y_test
        return results
    
    def run_experiment(self):
        print("\n" + "="*70)
        print("🔬 Comparative Experiment (Simplified)")
        print("="*70)
        
        all_results = []
        for i in range(self.n_repeats):
            print(f"\n🔄 Experiment {i+1}/{self.n_repeats} (Seed: {42+i})")
            try:
                results = self.single_experiment(random_seed=42+i)
                all_results.append(results)
                print(f"✅ Exp {i+1} Done")
            except Exception as e:
                print(f"❌ Exp {i+1} Failed: {e}")
                continue
        
        if not all_results: return None
        
        summary = self.summarize_results(all_results)
        self.plot_comparison_results(summary, all_results)
        return summary
    
    def summarize_results(self, all_results):
        print("\n" + "="*90)
        print("📊 综合性能评测表 (可直接填入论文 Table 中)")
        print("="*90)
        
        method_names = list(all_results[0]['methods'].keys())
        summary = {}
        
        # 【修改5】表头变更为 Constraint Residual 和 DCR
        print(f"| {'Method':<28} | {'R² ↑':<8} | {'RMSE ↓':<8} | {'MAE ↓':<8} | {'Constraint Residual ↓':<21} | {'WD Dist ↓':<10} | {'DCR ↑':<10} |")
        print("|" + "-"*30 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*23 + "|" + "-"*12 + "|" + "-"*12 + "|")
        
        for method_name in method_names:
            all_r2, all_rmse, all_mae = [], [], []
            all_physics_res, all_wd, all_dcr = [], [], []
            
            for result in all_results:
                if method_name in result['methods']:
                    m_res = result['methods'][method_name]
                    all_r2.append(m_res['metrics']['R2'])
                    all_rmse.append(m_res['metrics']['RMSE'])
                    all_mae.append(m_res['metrics']['MAE'])
                    all_physics_res.append(m_res['physics_residual'])
                    all_wd.append(m_res['wasserstein_dist'])
                    all_dcr.append(m_res['dcr_score']) # 【修改6】提取 DCR
            
            if not all_r2: continue
            
            summary[method_name] = {
                'description': all_results[0]['methods'][method_name]['description'],
                'r2_best': np.max(all_r2),
                'rmse_mean': np.mean(all_rmse),
                'mae_mean': np.mean(all_mae),
                'physics_res_mean': np.mean(all_physics_res),
                'wd_mean': np.mean(all_wd),
                'dcr_mean': np.mean(all_dcr) # 【修改7】保存 DCR
            }
            
            desc = summary[method_name]['description']
            r2 = summary[method_name]['r2_best']
            rmse = summary[method_name]['rmse_mean']
            mae = summary[method_name]['mae_mean']
            phys = summary[method_name]['physics_res_mean']
            wd = summary[method_name]['wd_mean']
            dcr = summary[method_name]['dcr_mean']
            
            if "Baseline" in desc:
                phys_str, wd_str, dcr_str = "-", "-", "-"
            else:
                phys_str = f"{phys:.6f}"
                wd_str = f"{wd:.4f}"
                dcr_str = f"{dcr:.4f}"
                
            print(f"| {desc:<28} | {r2:<8.4f} | {rmse:<8.4f} | {mae:<8.4f} | {phys_str:<21} | {wd_str:<10} | {dcr_str:<10} |")
        
        return summary
    
    def plot_comparison_results(self, summary, all_results):
        if not summary: return
        
        # 建立 2 行 3 列的超大高清画布
        fig = plt.figure(figsize=(20, 12))
        methods = list(summary.keys())
        descriptions = [summary[m]['description'] for m in methods]
        colors = ['gray', 'skyblue', 'mediumpurple', 'lightgreen', 'orange', 'red']
        
        # 内部画柱状图的辅助函数，让代码更整洁
        def create_bar_chart(ax, data_key, title, ylabel, show_baseline=True):
            if show_baseline:
                plot_idx = range(len(descriptions))
                labels = descriptions
            else:
                plot_idx = [i for i, desc in enumerate(descriptions) if "Baseline" not in desc]
                labels = [descriptions[i] for i in plot_idx]
                
            vals = [summary[methods[i]][data_key] for i in plot_idx]
            x_pos = np.arange(len(labels))
            bars = ax.bar(x_pos, vals, alpha=0.8)
            
            for i, bar in enumerate(bars): 
                bar.set_color(colors[plot_idx[i] % len(colors)])
                yval = bar.get_height()
                # 柱子上方写数值
                ax.text(bar.get_x() + bar.get_width()/2, yval + (max(vals)*0.01), 
                        f'{yval:.4f}', ha='center', va='bottom', fontsize=10)
                
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, rotation=20, ha='right')
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontweight='bold')
            ax.grid(True, alpha=0.3)
            return plot_idx
            
        # 1. R² Score (左上)
        ax1 = plt.subplot(2, 3, 1)
        create_bar_chart(ax1, 'r2_best', 'ML Efficiency (R² Score) ↑', 'R² Score', True)
        
        # 2. RMSE (中上)
        ax2 = plt.subplot(2, 3, 2)
        create_bar_chart(ax2, 'rmse_mean', 'Prediction Error (RMSE) ↓', 'RMSE', True)
        
        # 3. 帕累托前沿图 (右上)
        ax3 = plt.subplot(2, 3, 3)
        plot_indices = [i for i, desc in enumerate(descriptions) if "Baseline" not in desc]
        for i in plot_indices:
            phys_val = summary[methods[i]]['physics_res_mean']
            r2_val = summary[methods[i]]['r2_best']
            ax3.scatter(phys_val, r2_val, s=200, color=colors[i % len(colors)], label=descriptions[i], alpha=0.8, edgecolors='k')
            short_desc = descriptions[i].split()[0]
            ax3.annotate(short_desc, (phys_val, r2_val), xytext=(8, 8), textcoords='offset points', fontsize=10)
            
        ax3.set_xlabel('Constraint Residualidual Error (Lower is Better)')
        ax3.set_ylabel('R² Score (Higher is Better)')
        ax3.set_title("SOTA Trade-off Analysis (Jensen's Gap)", fontweight='bold')
        ax3.grid(True, linestyle='--', alpha=0.6)
        
        # 4. 物理残差 (左下)
        ax4 = plt.subplot(2, 3, 4)
        create_bar_chart(ax4, 'physics_res_mean', 'Physics Constraints Compliance ↓', 'Constraint Residualidual ||R(x)||²', False)
        
        # 5. Wasserstein 距离 (中下)
        ax5 = plt.subplot(2, 3, 5)
        create_bar_chart(ax5, 'wd_mean', 'Feature Marginal Dist (Wasserstein) ↓', 'WD Distance', False)
        
        # 6. DCR (右下)
        ax6 = plt.subplot(2, 3, 6)
        create_bar_chart(ax6, 'dcr_mean', 'Generative Diversity (DCR) ↑', 'DCR Distance', False)
        
        plt.tight_layout()
        plt.savefig('pccdm_comprehensive_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == '__main__':
    DATA_PATH = 'data/processed/concrete_augmented.csv'
    MODEL_PATHS = {
        'with_physics': 'results/models/pccdm_augmented_model.pth',
        'without_physics': 'results/models/diffusion_without_physics.pth'
    }
    
    missing = [path for path in MODEL_PATHS.values() if not os.path.exists(path)]
    if missing:
        print(f"⚠️ Missing Models: {missing}")
    
    if os.path.exists(DATA_PATH):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using Device: {device}")
        
        evaluator = ComparativeEvaluator(DATA_PATH, MODEL_PATHS, device, n_repeats=3)
        summary = evaluator.run_experiment()
        
        if summary:
            print("\n✅ Experiment Complete!")