# -*- coding: utf-8 -*-
"""
下游任务验证：多对照组比较实验（修复版）
包括：原始数据、高斯噪声、SMOTE、无物理约束扩散、物理约束扩散
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import math
import os
import warnings
import time
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib

# GBDT实现
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# 设置绘图后端和字体
try:
    matplotlib.use('TkAgg')
except:
    pass
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# ==========================================
# 1. 模型结构定义 (包含物理约束和无物理约束)
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



# 物理约束扩散模型
class PhysicsAwareResNetModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_main = config.d_main
        self.d_hidden = config.d_hidden
        self.n_blocks = config.n_blocks
        self.dropout = config.dropout
        
        # 1. 时间步嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(self.d_main, self.d_main),
            nn.SiLU(),
            nn.Linear(self.d_main, self.d_main)
        )
        
        # 2. 条件嵌入
        self.condition_encoder = nn.Sequential(
            nn.Linear(config.condition_dim, self.d_main),
            nn.SiLU(),
            nn.Linear(self.d_main, self.d_main)
        )
        
        # 3. 【已删除】物理约束编码器
        # self.physics_encoder = PhysicsConstraintEncoder(...) # 这一行被删除了
        
        # 4. 骨干网络
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
        # 即使删除了 encoder，为了接口兼容，我们保留 physics_constraints 参数，但不使用它
        
        t_emb = timestep_embedding(t, self.d_main)
        t_emb = self.time_embed(t_emb)
        
        c_emb = self.condition_encoder(conditions)
        
        # 融合上下文 (不再包含 p_emb)
        context = t_emb + c_emb
        
        h = self.input_proj(x_noisy)
        h = h + context
        return self.backbone(h)

# 无物理约束扩散模型
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
        self.d_main = 128
        self.d_hidden = 256
        self.hidden_dim = 256
        self.n_blocks = 3
        self.dropout = 0.1
        self.timesteps = 500
        self.beta_start = 0.0001
        self.beta_end = 0.02

# ==========================================
# 2. 采样逻辑
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

# ==========================================
# 3. 物理一致性检查器
# ==========================================

class PhysicsConsistencyChecker:
    """检查生成数据是否符合已知物理规律"""
    
    def __init__(self, y_mean, y_std, x_mean, x_std):
        """
        参数：
            y_mean, y_std: 目标变量的均值和标准差（用于反标准化）
            x_mean, x_std: 特征变量的均值和标准差（用于反标准化）
        """
        self.y_mean = y_mean
        self.y_std = y_std
        self.x_mean = x_mean
        self.x_std = x_std
        
    def inverse_transform_y(self, y_scaled):
        """反标准化目标变量"""
        if y_scaled.ndim == 1:
            y_scaled = y_scaled.reshape(-1, 1)
        return y_scaled * self.y_std + self.y_mean
    
    def inverse_transform_x(self, x_scaled):
        """反标准化特征变量"""
        return x_scaled * self.x_std + self.x_mean
    
    def check_monotonicity(self, X, y, feature_idx=0):
        """
        检查单调性：强度应随注量增加而降低
        参数：
            X: 特征矩阵
            y: 目标向量（相对抗压强度）
            feature_idx: 中子注量特征的索引（默认为0）
        """
        # 反标准化
        X_original = self.inverse_transform_x(X)
        y_original = self.inverse_transform_y(y)
        
        # 提取注量特征（需要逆log变换）
        fluence = np.expm1(X_original[:, feature_idx])  # 逆log(1+x)变换
        
        # 计算Pearson相关系数
        corr, p_value = pearsonr(fluence, y_original.ravel())
        
        # 计算Spearman秩相关系数（对单调性更敏感）
        spearman_corr, spearman_p = spearmanr(fluence, y_original.ravel())
        
        return {
            'pearson_correlation': corr,
            'pearson_p_value': p_value,
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'is_monotonic': corr < 0 and p_value < 0.05,  # 显著负相关
            'fluence_strength_relationship': f"注量-强度相关系数: {corr:.3f} (p={p_value:.3f})"
        }
    
    def check_temperature_effect(self, X, y, temp_idx=1):
        """
        检查温度效应：强度应随温度升高而降低
        参数：
            X: 特征矩阵
            y: 目标向量
            temp_idx: 温度特征的索引（默认为1）
        """
        # 反标准化
        X_original = self.inverse_transform_x(X)
        y_original = self.inverse_transform_y(y)
        
        # 提取温度特征
        temperature = X_original[:, temp_idx]
        
        # 计算相关系数
        corr, p_value = pearsonr(temperature, y_original.ravel())
        
        return {
            'temperature_correlation': corr,
            'temperature_p_value': p_value,
            'has_temperature_effect': corr < 0 and p_value < 0.05,
            'temperature_strength_relationship': f"温度-强度相关系数: {corr:.3f} (p={p_value:.3f})"
        }
    
    def check_bounds(self, y):
        """
        检查边界：相对抗压强度应在合理范围内
        混凝土相对抗压强度通常在0.2-1.5之间
        """
        # 反标准化
        y_original = self.inverse_transform_y(y)
        
        lower_bound = 0.2
        upper_bound = 1.5
        
        within_bounds = np.all((y_original >= lower_bound) & (y_original <= upper_bound))
        min_value = np.min(y_original)
        max_value = np.max(y_original)
        percentage_within = np.mean((y_original >= lower_bound) & (y_original <= upper_bound)) * 100
        
        return {
            'min_strength': min_value,
            'max_strength': max_value,
            'within_bounds': within_bounds,
            'percentage_within_bounds': percentage_within,
            'bounds_info': f"强度范围: [{min_value:.3f}, {max_value:.3f}], {percentage_within:.1f}%在合理范围内"
        }
    
    def check_wc_ratio_effect(self, X, y, wc_idx=2):
        """
        检查水灰比效应：强度应随水灰比增加而降低
        """
        # 反标准化
        X_original = self.inverse_transform_x(X)
        y_original = self.inverse_transform_y(y)
        
        # 提取水灰比特征
        wc_ratio = X_original[:, wc_idx]
        
        # 计算相关系数
        corr, p_value = pearsonr(wc_ratio, y_original.ravel())
        
        return {
            'wc_ratio_correlation': corr,
            'wc_ratio_p_value': p_value,
            'has_wc_ratio_effect': corr < 0 and p_value < 0.05,
            'wc_ratio_relationship': f"水灰比-强度相关系数: {corr:.3f} (p={p_value:.3f})"
        }
    
    def comprehensive_check(self, X, y, dataset_name=""):
        """
        综合物理一致性检查
        """
        print(f"\n🔬 {dataset_name}物理一致性检查:")
        print("="*50)
        
        results = {}
        
        # 1. 单调性检查
        mono_result = self.check_monotonicity(X, y)
        results.update(mono_result)
        print(f"1. 单调性（注量-强度）: {mono_result['fluence_strength_relationship']}")
        print(f"   是否显著单调递减: {'是' if mono_result['is_monotonic'] else '否'}")
        
        # 2. 温度效应检查
        temp_result = self.check_temperature_effect(X, y)
        results.update(temp_result)
        print(f"2. 温度效应: {temp_result['temperature_strength_relationship']}")
        print(f"   是否显著温度负效应: {'是' if temp_result['has_temperature_effect'] else '否'}")
        
        # 3. 水灰比效应检查
        wc_result = self.check_wc_ratio_effect(X, y)
        results.update(wc_result)
        print(f"3. 水灰比效应: {wc_result['wc_ratio_relationship']}")
        print(f"   是否显著水灰比负效应: {'是' if wc_result['has_wc_ratio_effect'] else '否'}")
        
        # 4. 边界检查
        bounds_result = self.check_bounds(y)
        results.update(bounds_result)
        print(f"4. 边界检查: {bounds_result['bounds_info']}")
        print(f"   是否全部在合理范围内: {'是' if bounds_result['within_bounds'] else '否'}")
        
        # 5. 计算物理一致性分数 (0-100分)
        consistency_score = 0
        if mono_result['is_monotonic']:
            consistency_score += 25
        if temp_result['has_temperature_effect']:
            consistency_score += 25
        if wc_result['has_wc_ratio_effect']:
            consistency_score += 25
        if bounds_result['percentage_within_bounds'] > 95:
            consistency_score += 25
        
        results['physics_consistency_score'] = consistency_score
        print(f"\n📊 物理一致性总分: {consistency_score}/100")
        
        return results

# ==========================================
# 4. 实验评估类（简化版，专注于关键比较）
# ==========================================

class ComparativeEvaluator:
    def __init__(self, data_path, model_paths, device='cpu', n_repeats=5):
        self.device = device
        self.data_path = data_path
        self.model_paths = model_paths
        self.n_repeats = n_repeats
        
        # 加载并预处理数据
        self.load_and_process_data()
        
        # 初始化物理一致性检查器
        self.physics_checker = PhysicsConsistencyChecker(
            self.y_mean, self.y_std, self.x_mean, self.x_std
        )
    
    def load_and_process_data(self):
        print(f"📊 正在加载数据: {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        # 特征选择
        self.target_col = ['相对抗压强度']
        self.cond_cols = [
            '快中子辐照注量 n/cm2', 
            '辐照温度', 
            '水灰比', 
            '骨料水泥比_numeric', 
            'concrete_type_encoded'
        ]
        
        # 提取数据
        self.Y_raw = df[self.target_col].values.astype(np.float32)
        self.X_raw = df[self.cond_cols].copy()
        
        # 预处理：Log 变换注量
        if '快中子辐照注量 n/cm2' in self.X_raw.columns:
            self.X_raw['快中子辐照注量 n/cm2'] = np.log1p(self.X_raw['快中子辐照注量 n/cm2'])
        
        self.X_raw = self.X_raw.values.astype(np.float32)
        
        # 记录统计量用于标准化
        self.y_mean = self.Y_raw.mean(axis=0)
        self.y_std = self.Y_raw.std(axis=0)
        self.x_mean = self.X_raw.mean(axis=0)
        self.x_std = self.X_raw.std(axis=0)
        
        # 标准化
        self.Y_scaled = (self.Y_raw - self.y_mean) / (self.y_std + 1e-8)
        self.X_scaled = (self.X_raw - self.x_mean) / (self.x_std + 1e-8)
        
        print("   数据预处理完成")
    
    def evaluate_model_performance(self, y_true, y_pred, prefix=""):
        """计算多个评估指标"""
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()
        
        metrics = {
            f'{prefix}R2': r2_score(y_true, y_pred),
            f'{prefix}RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            f'{prefix}MAE': mean_absolute_error(y_true, y_pred),
            f'{prefix}MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100,
        }
        
        return metrics
    
    def gaussian_noise_augmentation(self, X_train, y_train, n_samples=None):
        """
        高斯噪声增强：对条件添加高斯噪声，目标值保持原样
        """
        if n_samples is None:
            n_samples = X_train.shape[0]
        
        # 复制并添加噪声
        X_aug = X_train.copy()
        noise_scale = 0.1  # 噪声标准差
        
        # 对每个特征添加不同尺度的噪声
        for i in range(X_aug.shape[1]):
            feature_std = np.std(X_train[:, i])
            if feature_std > 0:
                noise = np.random.normal(0, noise_scale * feature_std, X_aug[:, i].shape)
                X_aug[:, i] = X_aug[:, i] + noise
        
        # 目标值保持不变
        y_aug = y_train.copy()
        
        # 确保维度一致
        if y_aug.ndim == 1:
            y_aug = y_aug.reshape(-1, 1)
        
        return X_aug[:n_samples], y_aug[:n_samples]
     
    def random_oversampling_augmentation(self, X_train, y_train, n_samples=None):
        """
        随机过采样增强：简单复制现有样本
        """
        if n_samples is None:
            n_samples = X_train.shape[0]
        
        # 随机选择样本进行复制
        indices = np.random.choice(len(X_train), size=n_samples, replace=True)
        X_aug = X_train[indices]
        y_aug = y_train[indices]
        
        return X_aug, y_aug
    
    def diffusion_augmentation(self, model, sampler, X_train, n_samples=None):
        """
        扩散模型增强：使用扩散模型生成新的目标值
        """
        if n_samples is None:
            n_samples = X_train.shape[0]
        
        # 对条件添加噪声
        X_synth_cond = X_train[:n_samples].copy()
        noise = np.random.normal(0, 0.08, X_synth_cond.shape).astype(np.float32)
        X_synth_cond = X_synth_cond + noise
        
        # 使用扩散模型生成目标值
        X_synth_cond_tensor = torch.tensor(X_synth_cond, device=sampler.device)
        Y_synth_scaled = sampler.generate(model, X_synth_cond_tensor).cpu().numpy()
        
        # 确保维度一致
        if Y_synth_scaled.ndim == 1:
            Y_synth_scaled = Y_synth_scaled.reshape(-1, 1)
        
        return X_synth_cond, Y_synth_scaled
    
    def single_experiment(self, random_seed=42):
        """单次实验，测试所有方法"""
        # 设置随机种子
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # 划分训练测试集
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_scaled, self.Y_scaled, test_size=0.2, random_state=random_seed
        )
        
        # 确保维度正确
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
        if y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)
        
        # 加载扩散模型
        config = Config()
        
        # 加载物理约束扩散模型
        model_with_physics = PhysicsAwareResNetModel(config).to(self.device)
        if os.path.exists(self.model_paths['with_physics']):
            model_with_physics.load_state_dict(
                torch.load(self.model_paths['with_physics'], map_location=self.device)
            )
            model_with_physics.eval()
        else:
            print(f"❌ 警告：未找到物理约束模型 {self.model_paths['with_physics']}")
            model_with_physics = None
        
        # 加载无物理约束扩散模型
        model_without_physics = DiffusionModelWithoutPhysics(config).to(self.device)
        if os.path.exists(self.model_paths['without_physics']):
            model_without_physics.load_state_dict(
                torch.load(self.model_paths['without_physics'], map_location=self.device)
            )
            model_without_physics.eval()
        else:
            print(f"❌ 警告：未找到无物理约束模型 {self.model_paths['without_physics']}")
            model_without_physics = None
        
        sampler = DiffusionSampler(config, self.device)
        
        # 定义所有方法
        methods = {
            'baseline': {
                'type': 'baseline',
                'description': '原始数据（无增强）'
            },
            'gaussian_noise': {
                'type': 'gaussian',
                'description': '高斯噪声增强'
            },
            'random_oversampling': {
                'type': 'random',
                'description': '随机过采样'
            }
        }
        
        # 添加扩散模型方法（如果可用）
        if model_without_physics is not None:
            methods['diffusion_without_physics'] = {
                'type': 'diffusion',
                'model': model_without_physics,
                'description': '无物理约束扩散模型'
            }
        
        if model_with_physics is not None:
            methods['diffusion_with_physics'] = {
                'type': 'diffusion',
                'model': model_with_physics,
                'description': '物理约束扩散模型'
            }
        
        # 存储结果
        results = {
            'seed': random_seed,
            'methods': {},
            'physics_checks': {}  # 新增：存储物理检查结果
        }
        
        # 对每种方法进行实验
        for method_name, method_info in methods.items():
            print(f"   正在测试：{method_info['description']}")
            
            try:
                if method_info['type'] == 'baseline':
                    # 基准方法：仅使用原始数据
                    X_train_aug = X_train
                    y_train_aug = y_train
                    
                    # 没有合成数据，跳过物理检查
                    physics_score = None
                    
                elif method_info['type'] == 'gaussian':
                    # 高斯噪声增强
                    X_train_aug, y_train_aug = self.gaussian_noise_augmentation(
                        X_train, y_train, n_samples=X_train.shape[0]
                    )
                    
                    # 检查增强数据的物理一致性
                    physics_check = self.physics_checker.comprehensive_check(
                        X_train_aug, y_train_aug.ravel(), f"{method_name}_synthetic"
                    )
                    physics_score = physics_check.get('physics_consistency_score', 0)
                    results['physics_checks'][method_name] = physics_check  # 存储详细结果
                    
                elif method_info['type'] == 'random':
                    # 随机过采样增强
                    X_train_aug, y_train_aug = self.random_oversampling_augmentation(
                        X_train, y_train, n_samples=X_train.shape[0]
                    )
                    
                    # 随机过采样不改变数据分布，物理一致性应该与原始数据类似
                    physics_score = None
                    
                elif method_info['type'] == 'diffusion':
                    # 扩散模型增强
                    X_train_aug, y_train_aug = self.diffusion_augmentation(
                        method_info['model'], sampler, X_train, n_samples=X_train.shape[0]
                    )
                    
                    # 检查增强数据的物理一致性
                    physics_check = self.physics_checker.comprehensive_check(
                        X_train_aug, y_train_aug.ravel(), f"{method_name}_synthetic"
                    )
                    physics_score = physics_check.get('physics_consistency_score', 0)
                    results['physics_checks'][method_name] = physics_check  # 存储详细结果
                
                # 训练下游模型（使用 LightGBM）
                model = LGBMRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=6,
                    num_leaves=31,
                    random_state=random_seed,
                    verbose=-1
                )
                
                # 合并原始和增强数据
                X_combined = np.vstack([X_train, X_train_aug])
                y_combined = np.vstack([y_train, y_train_aug])
                
                model.fit(X_combined, y_combined.ravel())
                y_pred = model.predict(X_test)
                
                # 评估性能
                metrics = self.evaluate_model_performance(y_test, y_pred, "")
                
                # 存储结果
                results['methods'][method_name] = {
                    'description': method_info['description'],
                    'physics_score': physics_score,
                    'metrics': metrics,
                    'y_pred': y_pred
                }
                
            except Exception as e:
                print(f"❌ 方法 {method_info['description']} 失败：{e}")
                continue
        
        results['y_true'] = y_test
        return results
    
    def run_experiment(self):
        """运行多次实验并汇总结果"""
        print("\n" + "="*70)
        print("🔬 多对照组比较实验（简化版）")
        print("="*70)
        
        all_results = []
        
        for i in range(self.n_repeats):
            print(f"\n🔄 实验 {i+1}/{self.n_repeats} (随机种子: {42+i})")
            
            try:
                results = self.single_experiment(random_seed=42+i)
                all_results.append(results)
                print(f"✅ 实验 {i+1} 完成")
            except Exception as e:
                print(f"❌ 实验 {i+1} 失败: {e}")
                continue
        
        if not all_results:
            print("\n❌ 所有实验都失败了！")
            return None
        
        # 汇总结果
        summary = self.summarize_results(all_results)
        
        # 可视化结果
        self.plot_comparison_results(summary, all_results)
        
        return summary
    
    def summarize_results(self, all_results):
        """汇总多次实验的结果"""
        print("\n" + "="*70)
        print("📊 多方法比较汇总结果")
        print("="*70)
        
        # 获取所有方法名称
        method_names = list(all_results[0]['methods'].keys())
        
        summary = {}
        
        for method_name in method_names:
            # 收集所有实验的指标
            all_r2 = []
            all_physics_scores = []
            
            for result in all_results:
                if method_name in result['methods']:
                    method_result = result['methods'][method_name]
                    all_r2.append(method_result['metrics']['R2'])
                    
                    if method_result['physics_score'] is not None:
                        all_physics_scores.append(method_result['physics_score'])
            
            if not all_r2:
                continue
            
            # 计算统计量
            r2_mean = np.mean(all_r2)
            r2_std = np.std(all_r2)
            
            if all_physics_scores:
                physics_mean = np.mean(all_physics_scores)
                physics_std = np.std(all_physics_scores)
            else:
                physics_mean = None
                physics_std = None
            
            summary[method_name] = {
                'description': all_results[0]['methods'][method_name]['description'],
                'r2_mean': r2_mean,
                'r2_std': r2_std,
                'physics_mean': physics_mean,
                'physics_std': physics_std,
                'n_experiments': len(all_r2)
            }
            
            print(f"\n📋 方法: {summary[method_name]['description']}")
            print(f"  平均R²: {r2_mean:.4f} (±{r2_std:.4f})")
            if physics_mean is not None:
                print(f"  平均物理一致性分数: {physics_mean:.1f} (±{physics_std:.1f})/100")
        
        # 计算相对于基线的改进
        if 'baseline' in summary:
            baseline_r2 = summary['baseline']['r2_mean']
            print(f"\n📈 相对于基线的R²改进:")
            for method_name, method_summary in summary.items():
                if method_name != 'baseline':
                    improvement = method_summary['r2_mean'] - baseline_r2
                    improvement_pct = (improvement / baseline_r2) * 100 if baseline_r2 != 0 else 0
                    print(f"  {method_summary['description']}: {improvement:+.4f} ({improvement_pct:+.1f}%)")
        
        return summary
    
    def plot_comparison_results(self, summary, all_results):
        """绘制比较结果图"""
        if not summary:
            return
        
        # ========== 图 1：主要性能指标对比 ==========
        fig1 = plt.figure(figsize=(15, 10))
        
        # 1. R²对比条形图
        ax1 = plt.subplot(2, 2, 1)
        method_descriptions = [summary[m]['description'] for m in summary.keys()]
        r2_means = [summary[m]['r2_mean'] for m in summary.keys()]
        r2_stds = [summary[m]['r2_std'] for m in summary.keys()]
        
        x_pos = np.arange(len(method_descriptions))
        
        bars = ax1.bar(x_pos, r2_means, yerr=r2_stds, capsize=5, alpha=0.8)
        
        # 为不同方法设置不同颜色
        colors = ['gray', 'skyblue', 'lightgreen', 'orange', 'red'][:len(bars)]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(method_descriptions, rotation=15, ha='right')
        ax1.set_ylabel('R²分数')
        ax1.set_title('各方法 R²性能对比 (±标准差)')
        ax1.grid(True, alpha=0.3)
        
        # 2. 物理一致性分数对比
        ax2 = plt.subplot(2, 2, 2)
        physics_means = []
        physics_stds = []
        physics_labels = []
        
        for method_name, method_summary in summary.items():
            if method_summary['physics_mean'] is not None:
                physics_means.append(method_summary['physics_mean'])
                physics_stds.append(method_summary['physics_std'])
                physics_labels.append(method_summary['description'])
        
        if physics_means:
            x_pos2 = np.arange(len(physics_labels))
            bars2 = ax2.bar(x_pos2, physics_means, yerr=physics_stds, capsize=5, alpha=0.8)
            
            # 设置颜色
            physics_colors = ['skyblue', 'orange', 'red'][:len(bars2)]
            for bar, color in zip(bars2, physics_colors):
                bar.set_color(color)
            
            ax2.set_xticks(x_pos2)
            ax2.set_xticklabels(physics_labels, rotation=15, ha='right')
            ax2.set_ylabel('物理一致性分数')
            ax2.set_title('生成数据物理一致性对比')
            ax2.set_ylim([0, 100])
            ax2.grid(True, alpha=0.3)
        
        # 3. R²改进幅度（相对于基线）
        ax3 = plt.subplot(2, 2, 3)
        if 'baseline' in summary:
            baseline_r2 = summary['baseline']['r2_mean']
            improvements = []
            improvement_labels = []
            
            for method_name, method_summary in summary.items():
                if method_name != 'baseline':
                    improvement = method_summary['r2_mean'] - baseline_r2
                    improvements.append(improvement)
                    improvement_labels.append(method_summary['description'])
            
            if improvements:
                x_pos3 = np.arange(len(improvements))
                bars3 = ax3.bar(x_pos3, improvements, alpha=0.8)
                
                # 设置颜色：正值绿色，负值红色
                for i, (bar, improvement) in enumerate(zip(bars3, improvements)):
                    bar.set_color('green' if improvement > 0 else 'red')
                    
                    # 标注数值
                    ax3.text(i, improvement + (0.01 if improvement >= 0 else -0.02), 
                            f'{improvement:+.3f}', ha='center', va='bottom' if improvement >= 0 else 'top')
                
                ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
                ax3.set_xticks(x_pos3)
                ax3.set_xticklabels(improvement_labels, rotation=15, ha='right')
                ax3.set_ylabel('R²改进（相对于基线）')
                ax3.set_title('各方法相对于基线的改进')
                ax3.grid(True, alpha=0.3)
        
        # 4. 方法比较总结表格
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis('tight')
        ax4.axis('off')
        
        # 创建总结表格
        table_data = []
        for method_name, method_summary in summary.items():
            if method_summary['physics_mean'] is not None:
                table_data.append([
                    method_summary['description'],
                    f"{method_summary['r2_mean']:.3f}±{method_summary['r2_std']:.3f}",
                    f"{method_summary['physics_mean']:.1f}±{method_summary['physics_std']:.1f}"
                ])
            else:
                table_data.append([
                    method_summary['description'],
                    f"{method_summary['r2_mean']:.3f}±{method_summary['r2_std']:.3f}",
                    "N/A"
                ])
        
        table = ax4.table(cellText=table_data,
                         colLabels=['方法', 'R²分数', '物理一致性分数'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.4, 0.3, 0.3])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        plt.suptitle('图 1：主要性能指标对比', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('figure1_performance_comparison.png', dpi=300, bbox_inches='tight')
        print("\n📊 图 1 已保存：figure1_performance_comparison.png")
        plt.show()
        
        # ========== 图 2：预测效果可视化 ==========
        fig2 = plt.figure(figsize=(15, 5))
        
        last_result = all_results[-1]
        y_true = last_result['y_true'].ravel()
        
        # 选择关键方法进行对比
        key_methods = ['baseline', 'gaussian_noise', 'random_oversampling']
        if 'diffusion_without_physics' in last_result['methods']:
            key_methods.append('diffusion_without_physics')
        if 'diffusion_with_physics' in last_result['methods']:
            key_methods.append('diffusion_with_physics')
        
        markers = ['o', 's', '^', 'v', 'D']
        
        # 5. 预测散点图对比
        ax5 = plt.subplot(1, 3, 1)
        
        for i, method_name in enumerate(key_methods):
            if method_name in last_result['methods']:
                method_result = last_result['methods'][method_name]
                y_pred = method_result['y_pred'].ravel()
                r2 = method_result['metrics']['R2']
                
                # 绘制散点图
                ax5.scatter(y_true, y_pred, alpha=0.6, marker=markers[i], 
                           label=f"{method_result['description']} (R²={r2:.3f})")
        
        ax5.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
        ax5.set_xlabel('真实值 (标准化)')
        ax5.set_ylabel('预测值 (标准化)')
        ax5.set_title('预测效果散点图对比')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)
        
        # 6. 误差分布对比
        ax6 = plt.subplot(1, 3, 2)
        
        for i, method_name in enumerate(key_methods):
            if method_name in last_result['methods']:
                method_result = last_result['methods'][method_name]
                y_pred = method_result['y_pred'].ravel()
                errors = y_true - y_pred
                
                # 绘制核密度估计
                sns.kdeplot(errors, fill=True, alpha=0.5, label=method_result['description'], ax=ax6)
        
        ax6.axvline(0, color='k', linestyle='--', linewidth=1)
        ax6.set_xlabel('预测误差')
        ax6.set_ylabel('密度')
        ax6.set_title('误差分布密度曲线')
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)
        
        # 7. 误差箱线图
        ax7 = plt.subplot(1, 3, 3)
        
        error_data = []
        error_labels = []
        for method_name in key_methods:
            if method_name in last_result['methods']:
                method_result = last_result['methods'][method_name]
                y_pred = method_result['y_pred'].ravel()
                errors = y_true - y_pred
                error_data.append(errors)
                error_labels.append(method_result['description'])
        
        bp = ax7.boxplot(error_data, labels=error_labels, patch_artist=True)
        
        # 设置颜色
        colors_box = ['gray', 'skyblue', 'lightgreen', 'orange', 'red'][:len(error_data)]
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax7.set_ylabel('预测误差')
        ax7.set_title('误差分布箱线图')
        ax7.grid(True, alpha=0.3)
        plt.xticks(rotation=15, ha='right')
        
        plt.suptitle('图 2：预测效果可视化', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('figure2_prediction_visualization.png', dpi=300, bbox_inches='tight')
        print("📊 图 2 已保存：figure2_prediction_visualization.png")
        plt.show()
        
        # ========== 图 3：物理一致性详细分析 ==========
        fig3 = plt.figure(figsize=(15, 5))
        
        # 从最后一次实验的物理检查中提取数据
        physics_checks = last_result.get('physics_checks', {})
        
        if physics_checks:
            correlation_data = {}
            correlation_labels = {
                'fluence': '注量 - 强度',
                'temperature': '温度 - 强度',
                'wc_ratio': '水灰比 - 强度'
            }
            
            for method_name, check_result in physics_checks.items():
                correlation_data[method_name] = {
                    'fluence': check_result.get('pearson_correlation', 0),
                    'temperature': check_result.get('temperature_correlation', 0),
                    'wc_ratio': check_result.get('wc_ratio_correlation', 0)
                }
            
            # 8. 物理相关性系数热力图
            ax8 = plt.subplot(1, 3, 1)
            
            methods_list = list(correlation_data.keys())
            correlations = ['fluence', 'temperature', 'wc_ratio']
            
            data_matrix = np.array([[correlation_data[m][c] for c in correlations] for m in methods_list])
            
            im = ax8.imshow(data_matrix, cmap='RdYlGn_r', aspect='auto', vmin=-1, vmax=0)
            
            # 设置刻度和标签
            ax8.set_xticks(np.arange(len(correlations)))
            ax8.set_yticks(np.arange(len(methods_list)))
            ax8.set_xticklabels([correlation_labels[c] for c in correlations], rotation=45, ha='right', fontsize=9)
            ax8.set_yticklabels([summary[m]['description'] for m in methods_list], fontsize=8)
            
            # 在每个单元格中显示数值
            for i in range(len(methods_list)):
                for j in range(len(correlations)):
                    text = ax8.text(j, i, f'{data_matrix[i, j]:.3f}',
                                   ha="center", va="center", color="black", fontsize=9)
            
            ax8.set_title('物理相关性系数热力图')
            plt.colorbar(im, ax=ax8, label='Pearson 相关系数')
            
            # 9. 相关性系数对比条形图
            ax9 = plt.subplot(1, 3, 2)
            
            x_pos = np.arange(len(correlations))
            width = 0.2
            
            for i, method_name in enumerate(methods_list):
                corr_values = [correlation_data[method_name][c] for c in correlations]
                ax9.bar(x_pos + i*width, corr_values, width, label=summary[method_name]['description'], alpha=0.8)
            
            ax9.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            ax9.set_ylabel('Pearson 相关系数')
            ax9.set_title('物理相关性系数对比')
            ax9.set_xticks(x_pos + width * (len(methods_list)-1)/2)
            ax9.set_xticklabels(['注量 - 强度', '温度 - 强度', '水灰比 - 强度'], rotation=15, ha='right')
            ax9.legend(fontsize=7, loc='upper right')
            ax9.grid(True, alpha=0.3)
            
            # 10. 物理一致性得分详情（堆叠条形图）
            ax10 = plt.subplot(1, 3, 3)
            
            methods_with_scores = []
            scores_list = []
            
            for method_name, check_result in physics_checks.items():
                # 计算各维度得分
                score_components = {
                    '注量效应': 25 if check_result.get('is_monotonic', False) else 0,
                    '温度效应': 25 if check_result.get('has_temperature_effect', False) else 0,
                    '水灰比效应': 25 if check_result.get('has_wc_ratio_effect', False) else 0,
                    '边界合理性': min(25, check_result.get('percentage_within_bounds', 0) / 95 * 25)
                }
                
                methods_with_scores.append(method_name)
                scores_list.append(list(score_components.values()))
            
            # 绘制堆叠条形图
            x_pos = np.arange(len(methods_with_scores))
            bottom = np.zeros(len(methods_with_scores))
            
            categories = ['注量效应', '温度效应', '水灰比效应', '边界合理性']
            category_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            
            for i, (category, color) in enumerate(zip(categories, category_colors)):
                scores = [scores_list[j][i] for j in range(len(methods_with_scores))]
                ax10.bar(x_pos, scores, bottom=bottom, label=category, color=color, alpha=0.8)
                bottom += scores
            
            ax10.set_xticks(x_pos)
            ax10.set_xticklabels([summary[m]['description'] for m in methods_with_scores], rotation=15, ha='right')
            ax10.set_ylabel('比率')
            ax10.set_title('物理合理性比率')
            ax10.legend(loc='upper right', fontsize=8)
            ax10.set_ylim([0, 100])
            ax10.grid(True, alpha=0.3)
        
        plt.suptitle('图 3：物理合理性比率详细分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('figure3_physics_analysis.png', dpi=300, bbox_inches='tight')
        print("📊 图 3 已保存：figure3_physics_analysis.png")
        plt.show()

# ==========================================
# 主程序
# ==========================================

if __name__ == '__main__':
    # 文件路径
    DATA_PATH = 'data/processed/concrete_augmented.csv'
    MODEL_PATHS = {
        'with_physics': 'results/models/pccdm_augmented_model.pth',
        'without_physics': 'results/models/diffusion_without_physics.pth'
    }
    
    # 检查文件是否存在
    missing_files = []
    for key, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            missing_files.append((key, path))
            print(f"⚠️ 警告: 未找到模型文件 {path}")
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ 错误: 未找到数据文件 {DATA_PATH}")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        # 运行比较实验
        experiment = ComparativeEvaluator(
            DATA_PATH, MODEL_PATHS, device, n_repeats=3  # 先运行3次测试
        )
        
        summary = experiment.run_experiment()
        
        if summary:
            print("\n✅ 比较实验完成！")
            print("\n🎯 关键发现:")
            
            # 找出最佳方法
            best_method = max(summary.items(), key=lambda x: x[1]['r2_mean'])
            print(f"1. 最佳预测性能: {best_method[1]['description']} (R²={best_method[1]['r2_mean']:.3f})")
            
            # 找出最佳物理一致性
            methods_with_physics = {k: v for k, v in summary.items() if v['physics_mean'] is not None}
            if methods_with_physics:
                best_physics = max(methods_with_physics.items(), key=lambda x: x[1]['physics_mean'])
                print(f"2. 最佳物理一致性: {best_physics[1]['description']} ({best_physics[1]['physics_mean']:.1f}/100)")
            
            # 计算物理约束扩散模型的改进
            if 'diffusion_with_physics' in summary and 'diffusion_without_physics' in summary:
                improvement = (summary['diffusion_with_physics']['r2_mean'] - 
                              summary['diffusion_without_physics']['r2_mean'])
                print(f"3. 物理约束带来的R²额外提升: {improvement:+.4f}")