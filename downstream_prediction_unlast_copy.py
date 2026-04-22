import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import math
import os

# 设置绘图风格
plt.style.use('default')
# 尝试设置中文字体，如果失败则回退到英文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_context("paper", font_scale=1.2)

# ==========================================
# 1. 模型定义 (必须与训练时一致)
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
        return x + self.linear2(self.dropout(self.act(self.linear1(self.norm1(x)))))

class ResNet(nn.Module):
    def __init__(self, d_in, n_blocks, d_main, d_hidden, dropout_first, dropout_second, d_out):
        super().__init__()
        self.input_proj = nn.Linear(d_in, d_main)
        self.blocks = nn.ModuleList([
            ResNetBlock(d_main, d_hidden, dropout_second) for _ in range(n_blocks)
        ])
        self.norm_out = nn.LayerNorm(d_main)
        self.output_proj = nn.Linear(d_main, d_out)
        self.dropout_first = nn.Dropout(dropout_first)
    def forward(self, x):
        h = self.input_proj(x)
        h = self.dropout_first(h)
        for block in self.blocks: h = block(h)
        return self.output_proj(self.norm_out(h))

class DiffusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_main = config.d_main
        self.time_embed = nn.Sequential(
            nn.Linear(self.d_main, self.d_main), nn.SiLU(), nn.Linear(self.d_main, self.d_main)
        )
        self.condition_encoder = nn.Sequential(
            nn.Linear(config.condition_dim, self.d_main), nn.SiLU(), nn.Linear(self.d_main, self.d_main)
        )
        self.backbone = ResNet(
            self.d_main, config.n_blocks, self.d_main, config.d_hidden, config.dropout, config.dropout, config.input_dim
        )
        self.input_proj = nn.Linear(config.input_dim, self.d_main)
        
    def forward(self, x_noisy, t, conditions, physics_constraints=None):
        t_emb = self.time_embed(timestep_embedding(t, self.d_main))
        c_emb = self.condition_encoder(conditions)
        return self.backbone(self.input_proj(x_noisy) + t_emb + c_emb)

# 兼容旧代码的类名
PhysicsAwareResNetModel = DiffusionModel
DiffusionModelWithoutPhysics = DiffusionModel

class Config:
    def __init__(self):
        self.input_dim = 1
        self.condition_dim = 5
        self.d_main = 128
        self.d_hidden = 256
        self.n_blocks = 3
        self.dropout = 0.1
        self.timesteps = 500
        self.beta_start = 0.0001
        self.beta_end = 0.02

# ==========================================
# 2. 采样器
# ==========================================

class DiffusionSampler:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.timesteps = config.timesteps
        self.betas = torch.linspace(config.beta_start, config.beta_end, self.timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
    @torch.no_grad()
    def generate(self, model, conditions):
        batch_size = conditions.shape[0]
        x = torch.randn(batch_size, self.config.input_dim, device=self.device)
        for i in reversed(range(0, self.timesteps)):
            t = torch.tensor([i] * batch_size, device=self.device)
            pred_noise = model(x, t, conditions)
            model_mean = self.sqrt_recip_alphas[i] * (x - self.betas[i] * pred_noise / self.sqrt_one_minus_alphas_cumprod[i])
            if i > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(self.betas[i] * (1. - self.alphas_cumprod[i-1]) / (1. - self.alphas_cumprod[i]))
                x = model_mean + sigma * noise
            else:
                x = model_mean
        return x

# ==========================================
# 3. 核心绘图逻辑
# ==========================================

def plot_physics_correlations():
    # 路径配置
    data_path = 'data/processed/concrete_augmented.csv'
    model_paths = {
        'Without Physics': 'results/models/diffusion_without_physics.pth',
        'With Physics (PCCDM)': 'results/models/pccdm_augmented_model.pth'
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载数据
    df = pd.read_csv(data_path)
    cond_cols = ['快中子辐照注量 n/cm2', '辐照温度', '水灰比', '骨料水泥比_numeric', 'concrete_type_encoded']
    X_raw = df[cond_cols].copy()
    if '快中子辐照注量 n/cm2' in X_raw.columns:
        X_raw['快中子辐照注量 n/cm2'] = np.log1p(X_raw['快中子辐照注量 n/cm2'])
    
    X_mean = X_raw.mean(axis=0)
    X_std = X_raw.std(axis=0)
    X_scaled = (X_raw - X_mean) / (X_std + 1e-8)
    X_tensor = torch.tensor(X_scaled.values, dtype=torch.float32).to(device)
    
    # === 修改 A: 增加绘图样本量，确保 P 值稳定 ===
    # 如果数据量够大，尽量用 1000 个点，这样 weak correlation 也能显现出显著性
    sample_size = min(1000, len(X_tensor))
    indices = torch.randperm(len(X_tensor))[:sample_size]
    
    X_sample = X_tensor[indices]
    X_raw_sample = X_raw.iloc[indices.cpu().numpy()].reset_index(drop=True)
    
    # 2. 生成数据 (代码不变)
    generated_data = {}
    config = Config()
    sampler = DiffusionSampler(config, device)
    
    for name, path in model_paths.items():
        if not os.path.exists(path):
            continue
        model = DiffusionModel(config).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        generated_data[name] = sampler.generate(model, X_sample).cpu().numpy()
    
    # 3. 绘制对比图
    features_to_plot = [
        ('Fluence (Log scale)', '快中子辐照注量 n/cm2', 0),
        ('Water-to-Cement Ratio', '水灰比', 2)
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True) # 稍微调宽一点
    
    for i, (feat_name_en, feat_col, feat_idx) in enumerate(features_to_plot):
        x_val = X_raw_sample[feat_col].values
        
        for j, (model_name, y_val) in enumerate(generated_data.items()):
            ax = axes[i, j]
            
            # 绘制散点
            sns.regplot(x=x_val, y=y_val, ax=ax, 
                        scatter_kws={'alpha':0.4, 's':20}, 
                        line_kws={'color':'red', 'linewidth': 2})
            
            # 计算相关性
            corr, p_val = stats.pearsonr(x_val, y_val.ravel())
            
            # 设置标题
            if i == 0:
                ax.set_title(f"{model_name}", fontsize=14, fontweight='bold', pad=15)
            if j == 0:
                ax.set_ylabel("Predicted Strength (Norm)", fontsize=12)
            else:
                ax.set_ylabel("")
            ax.set_xlabel(feat_name_en, fontsize=12)
            
            # === 修改 B: 修复文本显示问题 ===
            # 判定物理一致性
            is_consistent = (corr < 0) and (p_val < 0.05)
            
            # 颜色和状态文本 (放弃 Emoji，使用纯文本)
            color = 'green' if is_consistent else 'red'
            status_text = "Consistent: YES" if is_consistent else "Consistent: NO"
            
            # 构建统计信息字符串
            stats_text = (f"Pearson r: {corr:.3f}\n"
                          f"P-value: {p_val:.3e}\n"
                          f"{status_text}")
            
            # 文本框
            props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor=color, linewidth=1.5)
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                    fontsize=11, verticalalignment='top', bbox=props, color='black', fontfamily='sans-serif')
            
            ax.grid(True, alpha=0.3, linestyle='--')

    plt.suptitle(f"Physics Consistency Comparison (N={sample_size})", fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig('physics_evidence_plot_fixed.png', dpi=300, bbox_inches='tight')
    print("图表已保存为 physics_evidence_plot_fixed.png")
    plt.show()

# ==========================================
# 4. 稳定性验证模块 (新增代码)
# ==========================================

def verify_improvement_stability(n_rounds=20, sample_size=800):
    print(f"🚀 开始稳定性验证 (共 {n_rounds} 轮, 每轮样本 N={sample_size})...")
    
    # --- 1. 准备工作 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = Config()
    
    # 加载数据
    data_path = 'data/processed/concrete_augmented.csv'
    df = pd.read_csv(data_path)
    cond_cols = ['快中子辐照注量 n/cm2', '辐照温度', '水灰比', '骨料水泥比_numeric', 'concrete_type_encoded']
    X_raw_full = df[cond_cols].copy()
    if '快中子辐照注量 n/cm2' in X_raw_full.columns:
        X_raw_full['快中子辐照注量 n/cm2'] = np.log1p(X_raw_full['快中子辐照注量 n/cm2'])
    
    # 标准化
    X_mean = X_raw_full.mean(axis=0)
    X_std = X_raw_full.std(axis=0)
    X_scaled_full = (X_raw_full - X_mean) / (X_std + 1e-8)
    X_tensor_full = torch.tensor(X_scaled_full.values, dtype=torch.float32).to(device)

    # 加载模型 (只加载一次以节省时间)
    model_paths = {
        'Without Physics': 'results/models/diffusion_without_physics.pth',
        'With Physics (PCCDM)': 'results/models/pccdm_augmented_model.pth'
    }
    models = {}
    for name, path in model_paths.items():
        if os.path.exists(path):
            m = DiffusionModel(config).to(device)
            m.load_state_dict(torch.load(path, map_location=device))
            m.eval()
            models[name] = m
        else:
            print(f"⚠️ Warning: Model not found at {path}")
            return

    sampler = DiffusionSampler(config, device)
    
    # 记录结果的容器
    results = []
    
    # --- 2. 循环验证 ---
    features_to_test = [
        ('Fluence', '快中子辐照注量 n/cm2'),
        ('W/C Ratio', '水灰比')
    ]
    
    for i in range(n_rounds):
        print(f"  [Round {i+1}/{n_rounds}] Sampling & Generating...", end='\r')
        
        # 随机重采样
        indices = torch.randperm(len(X_tensor_full))[:sample_size]
        X_subset = X_tensor_full[indices]
        X_raw_subset = X_raw_full.iloc[indices.cpu().numpy()].reset_index(drop=True)
        
        for model_name, model in models.items():
            # 生成预测
            y_pred = sampler.generate(model, X_subset).cpu().numpy().ravel()
            
            for feat_name, feat_col in features_to_test:
                x_val = X_raw_subset[feat_col].values
                
                # 计算统计量
                r, p = stats.pearsonr(x_val, y_pred)
                
                # 判断是否符合物理一致性 (负相关且显著)
                is_pass = (r < 0) and (p < 0.05)
                
                results.append({
                    'Round': i,
                    'Model': model_name,
                    'Feature': feat_name,
                    'Pearson_r': r,
                    'P_value': p,
                    'Pass': is_pass
                })  
    
    print(f"\n✅ 验证完成! 正在生成分析报告...")
    df_res = pd.DataFrame(results)
    
    # --- 3. 生成统计表格 ---
    print("\n" + "="*60)
    print(" >>> 物理一致性稳定性报告 (Stability Report) <<< ")
    print("="*60)
    
    summary = df_res.groupby(['Feature', 'Model']).agg(
        Mean_R=('Pearson_r', 'mean'),
        Std_R=('Pearson_r', 'std'),
        Max_P=('P_value', 'max'),  # 看看最差的一次P值是多少
        Pass_Rate=('Pass', lambda x: f"{sum(x)/len(x):.1%}") # 成功率
    )
    print(summary)
    
    # 保存 CSV
    df_res.to_csv('physics_stability_results.csv', index=False)
    
    # --- 4. 绘制箱线图 (核心证据) ---
    plt.figure(figsize=(12, 6))
    
    # 子图 1: Pearson r 分布
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df_res, x='Feature', y='Pearson_r', hue='Model', palette=['#ff9999', '#99ff99'])
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.title("Distribution of Pearson Correlation (r)\n(Lower is Better)", fontsize=12)
    plt.ylabel("Pearson r")
    
    # 子图 2: P-value 分布 (取对数显示以便观察极小值)
    plt.subplot(1, 2, 2)
    df_res['log_p'] = np.log10(df_res['P_value'] + 1e-50) # 避免log(0)
    sns.boxplot(data=df_res, x='Feature', y='log_p', hue='Model', palette=['#ff9999', '#99ff99'])
    plt.axhline(np.log10(0.05), color='red', linestyle='--', label='p=0.05 Threshold')
    plt.title("Distribution of Log P-values\n(Lower is More Significant)", fontsize=12)
    plt.ylabel("Log10(P-value)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('physics_stability_boxplot.png', dpi=300)
    print("\n📊 统计图表已保存为: physics_stability_boxplot.png")
    plt.show()

if __name__ == '__main__':
    verify_improvement_stability(n_rounds=20, sample_size=800)
    