# -*- coding: utf-8 -*-
"""
最终全能版评估脚本 (Final Integrated Evaluation Framework)
包含：课程学习、数据增强、统计评估、以及深度物理约束验证
"""

import matplotlib
matplotlib.use('TkAgg')
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # 新增: 用于绘制趋势线
import torch
import torch.nn.functional as F
import sys
import os
import warnings
import json
from scipy.stats import wasserstein_distance, spearmanr # 新增: 用于相关性检查
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

# 导入简化扩散模型中的关键组件
sys.path.append('.')
try:
    from pccdm_model_last import (
        ConcreteDataPreprocessor,
        ConcreteDataset,
        SimpleConditionalDiffusion,
        DiffusionManager,
        PhysicsAwareLoss,
        DiffusionTrainer,
        ConcreteGenerator
    )
    print("✅ 成功导入扩散模型组件")
except ImportError as e:
    print(f"❌ 无法导入扩散模型组件: {e}")
    sys.exit(1)

# 创建目录
os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/models', exist_ok=True)

# ==================== 1. 物理约束损失 (保持不变) ====================
class ImprovedPhysicsAwareLoss(PhysicsAwareLoss):
    def __init__(self, alpha=0.1, wcr_weight=1.0):
        super().__init__(alpha=alpha)
        self.wcr_weight = wcr_weight
    
    def compute_physics_loss(self, x_pred, physics_conditions):
        loss = 0.0
        # 约束1: 水灰比范围 (放宽到 ±2.2 以允许边缘分布)
        wcr = x_pred[:, 0]
        wcr_loss = F.relu(wcr - 2.2) + F.relu(-2.2 - wcr)
        loss += torch.mean(wcr_loss) * self.wcr_weight
        
        # 约束2: 辐照注量非负
        if x_pred.shape[1] >= 2:
            fluence = x_pred[:, 1]
            fluence_loss = F.relu(-fluence - 0.5)
            loss += torch.mean(fluence_loss) * 0.5
        return loss

# ==================== 2. 训练器 (保持不变) ====================
class ImprovedDiffusionTrainer(DiffusionTrainer):
    def __init__(self, model, diffusion, loss_fn, device='cpu'):
        super().__init__(model, diffusion, loss_fn, device=device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        self.scheduler = None 
    
    def train_epoch(self, dataloader, epoch, total_epochs):
        self.model.train()
        total_loss = 0
        
        # 课程学习：动态调整物理权重
        if epoch < total_epochs * 0.3:
            physics_weight_factor = 0.0
        elif epoch < total_epochs * 0.7:
            physics_weight_factor = (epoch - total_epochs * 0.3) / (total_epochs * 0.4)
        else:
            physics_weight_factor = 1.0
            
        for batch_idx, batch in enumerate(dataloader):
            x_start = batch['input'].to(self.device)
            physics_conditions = batch['physics'].to(self.device)
            batch_size = x_start.shape[0]
            
            t = torch.randint(0, self.diffusion.timesteps, (batch_size,), device=self.device).long()
            x_noisy, true_noise = self.diffusion.q_sample(x_start, t)
            
            noise_pred = self.model(x_noisy, t.unsqueeze(-1).float(), physics_conditions)
            
            x_pred = x_noisy - noise_pred
            _, base_loss, physics_loss_raw = self.loss_fn(
                noise_pred, true_noise, x_pred, physics_conditions
            )
            
            # 组合 Loss
            loss = base_loss + (physics_loss_raw * physics_weight_factor * 0.5)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)

# ==================== 3. 全能评估器 (集成物理验证) ====================
class IntegratedEvaluator:
    def __init__(self, csv_path, device='cpu'):
        self.csv_path = csv_path
        self.device = device
        self.feature_names = ['水灰比', '辐照注量', '温度', '类型'] 
    
    def prepare_data(self):
        print("\n1. 准备数据 (含增强)...")
        preprocessor = ConcreteDataPreprocessor(self.csv_path)
        input_data, output_data, physics_data = preprocessor.preprocess()
        
        self.preprocessor = preprocessor
        self.raw_input = input_data
        
        # 数据增强 x5
        aug_inputs, aug_physics = [input_data], [physics_data]
        for _ in range(4):
            aug_inputs.append(input_data + np.random.normal(0, 0.05, input_data.shape))
            aug_physics.append(physics_data + np.random.normal(0, 0.02, physics_data.shape))
            
        self.train_input = np.concatenate(aug_inputs).astype(np.float32)
        self.train_physics = np.concatenate(aug_physics).astype(np.float32)
        self.train_output = np.tile(output_data, (5, 1)).astype(np.float32)
        
        return self.train_input, self.train_physics

    def train(self, epochs=1200):
        print(f"\n2. 训练模型 ({epochs} Epochs)...")
        dataset = ConcreteDataset(self.train_input, self.train_output, self.train_physics)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
        
        self.model = SimpleConditionalDiffusion(
            input_dim=self.train_input.shape[1],
            physics_dim=self.train_physics.shape[1],
            hidden_dim=128
        ).to(self.device)
        
        self.diffusion = DiffusionManager(timesteps=300, device=self.device)
        trainer = ImprovedDiffusionTrainer(self.model, self.diffusion, ImprovedPhysicsAwareLoss(), device=self.device)
        
        trainer.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            trainer.optimizer, max_lr=1e-3, total_steps=epochs*len(dataloader), pct_start=0.3
        )
        
        for epoch in range(epochs):
            loss = trainer.train_epoch(dataloader, epoch, epochs)
            if (epoch+1) % 200 == 0:
                print(f"  Epoch {epoch+1}/{epochs} | Loss: {loss:.4f}")

    def generate(self, num_samples=300):
        print("\n3. 生成样本...")
        generator = ConcreteGenerator(self.model, self.diffusion, self.preprocessor.scalers, self.device)
        
        idx = np.random.choice(len(self.train_physics), num_samples, replace=False)
        cond = torch.tensor(self.train_physics[idx]).to(self.device)
        
        self.synth_data, _ = generator.generate_samples(num_samples, cond)
        self.real_data = self.raw_input[:num_samples] if len(self.raw_input) > num_samples else self.raw_input
        self.synth_data = self.synth_data[:len(self.real_data)]

    def evaluate_comprehensive(self):
        print("\n4. 执行全能评估 (统计 + 物理)...")
        real = self.real_data
        synth = self.synth_data
        
        # --- A. 基础统计指标 ---
        print("-" * 40)
        print("📊 A. 统计一致性 (Statistical Fidelity)")
        print("-" * 40)
        
        wd_scores = []
        for i in range(real.shape[1]):
            wd = wasserstein_distance(real[:, i], synth[:, i])
            wd_scores.append(wd)
        print(f"  平均 Wasserstein 距离: {np.mean(wd_scores):.4f} (越小越好)")
        
        corr_real = np.corrcoef(real.T)
        corr_synth = np.corrcoef(synth.T)
        corr_real = np.nan_to_num(corr_real)
        corr_synth = np.nan_to_num(corr_synth)
        corr_diff = np.mean(np.abs(corr_real - corr_synth))
        print(f"  相关性矩阵平均误差: {corr_diff:.4f} (越小越好)")

        # --- B. 物理深度验证 (新增) ---
        print("\n" + "-" * 40)
        print("🧪 B. 物理约束验证 (Physics Verification)")
        print("-" * 40)
        
        # 1. 硬约束检查
        print("[1] 硬约束边界检查:")
        # 辐照注量(Index 1)非负检查 (阈值 -0.5)
        fluence_violation = np.mean(synth[:, 1] < -0.5)
        print(f"  - 辐照注量非负违反率: {fluence_violation:.2%} (阈值: -0.5)")
        # 水灰比(Index 0)范围检查 (±2.5)
        wcr_violation = np.mean((synth[:, 0] < -2.5) | (synth[:, 0] > 2.5))
        print(f"  - 水灰比范围违反率: {wcr_violation:.2%} (阈值: ±2.5)")
        
        if fluence_violation < 0.05 and wcr_violation < 0.05:
            print("  ✅ 硬约束检查通过!")
        else:
            print("  ⚠️ 警告: 存在物理约束违反，可能需要调整Loss权重。")

        # 2. 相关性方向一致性 (Spearman)
        print("\n[2] 物理趋势方向一致性 (Spearman相关系数):")
        print(f"  {'特征对':<20} | {'真实相关性':<10} | {'生成相关性':<10} | {'一致性'}")
        print("-" * 65)
        
        n_features = real.shape[1]
        consistent_count = 0
        total_significant = 0
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                r_real, _ = spearmanr(real[:, i], real[:, j])
                r_synth, _ = spearmanr(synth[:, i], synth[:, j])
                
                # 只有当真实相关性足够强(>0.1)时，才强制要求方向一致
                is_significant = abs(r_real) > 0.1
                match = (np.sign(r_real) == np.sign(r_synth))
                
                icon = "✅" if (match or not is_significant) else "❌"
                if is_significant:
                    total_significant += 1
                    if match: consistent_count += 1
                
                pair_name = f"{self.feature_names[i]} vs {self.feature_names[j]}"
                print(f"  {pair_name:<20} | {r_real:>10.3f} | {r_synth:>10.3f} | {icon}")
        
        if total_significant > 0:
            print(f"\n  > 显著相关性方向一致率: {consistent_count}/{total_significant} ({consistent_count/total_significant:.1%})")

        # --- C. 可视化 ---
        self._create_visualizations(real, synth, corr_real, corr_synth)

    def _create_visualizations(self, real, synth, corr_real, corr_synth):
        print("\n5. 生成可视化图表...")
        
        # 1. PCA
        pca = PCA(n_components=2)
        real_pca = pca.fit_transform(real)
        synth_pca = pca.transform(synth)
        
        plt.figure(figsize=(8, 6))
        plt.scatter(real_pca[:, 0], real_pca[:, 1], c='blue', alpha=0.5, label='真实', s=30)
        plt.scatter(synth_pca[:, 0], synth_pca[:, 1], c='red', alpha=0.5, label='生成', s=30, marker='x')
        plt.title('PCA 流形分布对比')
        plt.legend()
        plt.savefig('results/figures/final_pca.png')
        
        # 2. 相关性热力图
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.heatmap(corr_real, annot=True, fmt=".2f", cmap='coolwarm', ax=axes[0], vmin=-1, vmax=1)
        axes[0].set_title('真实数据相关性')
        sns.heatmap(corr_synth, annot=True, fmt=".2f", cmap='coolwarm', ax=axes[1], vmin=-1, vmax=1)
        axes[1].set_title('生成数据相关性')
        plt.savefig('results/figures/final_heatmap.png')
        
        # 3. 物理趋势图 (Feature 0 vs Feature 1)
        plt.figure(figsize=(8, 6))
        sns.regplot(x=real[:, 0], y=real[:, 1], scatter_kws={'alpha':0.2}, label='真实趋势', color='blue')
        sns.regplot(x=synth[:, 0], y=synth[:, 1], scatter_kws={'alpha':0.2, 'marker':'x'}, label='生成趋势', color='red')
        plt.xlabel(self.feature_names[0])
        plt.ylabel(self.feature_names[1])
        plt.title(f'物理一致性: {self.feature_names[0]} vs {self.feature_names[1]}')
        plt.legend()
        plt.savefig('results/figures/final_physics_trend.png')
        
        print("✅ 图表已保存至 results/figures/")

if __name__ == "__main__":
    print("="*60)
    print("启动最终全能评估流程 (含物理验证)")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    csv_path = "data/processed/concrete_processed_with_physics_features.csv"
    
    evaluator = IntegratedEvaluator(csv_path, device)
    evaluator.prepare_data()
    evaluator.train(epochs=1200)
    evaluator.generate()
    evaluator.evaluate_comprehensive()
    
    print("\n✅ 所有任务完成！")