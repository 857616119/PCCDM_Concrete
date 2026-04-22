# -*- coding: utf-8 -*-
"""
最终验证版：标准数据增强评估流程
逻辑：
1. 划分真实数据 -> 80% 训练集 (Train_Real), 20% 测试集 (Test_Real)
2. 基准线 -> 仅用 Train_Real 训练，预测 Test_Real
3. 数据生成 -> 生成合成数据，利用 Train_Real 进行最近邻打标签
4. 增强训练 -> 用 (Train_Real + Synthetic) 训练，预测 Test_Real
5. 对比 -> 验证合成数据是否提升了对 Test_Real 的预测精度
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import sys
import os
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
import xgboost as xgb
from pccdm_model import PhysicsAwareResNetModel, DiffusionProcess, PCCDMConfig
# 在文件头部引入 KNNImputer
from sklearn.impute import KNNImputer
# 设置绘图风格
import matplotlib
matplotlib.use('TkAgg')
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

class StandardAugmentationEvaluator:
    def __init__(self, data_path, model_path, device='cpu'):
        self.data_path = data_path
        self.model_path = model_path
        self.device = device
        
        # 1. 加载数据
        self.df = pd.read_csv(data_path)
        
        # 定义特征
        self.feature_names = [
            'concrete_type_encoded', 
            '快中子辐照注量 n/cm2_scaled', 
            '辐照温度_scaled', 
            '水灰比', 
            '骨料水泥比'
        ]
        self.features = [c for c in self.feature_names if c in self.df.columns]
        
        # --- 清洗非数字数据 ---
        def clean_numeric_column(x):
            if pd.isna(x): return np.nan
            if isinstance(x, (int, float)): return float(x)
            x = str(x).strip()
            if ',' in x:
                try:
                    parts = [float(p) for p in x.split(',')]
                    return sum(parts) / len(parts)
                except:
                    return np.nan
            try: return float(x)
            except: return np.nan

        for col in self.features:
            if self.df[col].dtype == 'object':
                self.df[col] = self.df[col].apply(clean_numeric_column)
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # 3. 提取数据
        raw_X = self.df[self.features].values
        
        # --- 🚀 关键修改 1: 使用 KNNImputer 智能填充缺失值 ---
        # 相比用均值，KNN 会寻找相似的混凝土样本（比如同类型的），用它们的数值来填补
        print("⚡ 正在使用 KNNImputer 智能修复缺失的物理参数 (如水灰比)...")
        self.imputer = KNNImputer(n_neighbors=5)
        raw_X_imputed = self.imputer.fit_transform(raw_X)
        
        # 4. 标准化
        from sklearn.preprocessing import RobustScaler
        self.scaler = RobustScaler()
        self.X_processed = self.scaler.fit_transform(raw_X_imputed)
        
        # 处理目标变量
        self.y = self.df['相对抗压强度_scaled'].values
        self.y = np.nan_to_num(self.y, nan=0.0)

    def get_optimal_model(self, X_train, y_train, name="Baseline"):
        """
        使用网格搜索寻找最佳超参数
        保证基准模型和增强模型都处于最佳状态，公平比较
        """
        print(f"\n🔍 正在为 [{name}] 寻找最佳超参数...")
        
        # 使用随机森林，因为它对小样本和噪声最稳健
        model = RandomForestRegressor(random_state=42)
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        grid = GridSearchCV(
            model, 
            param_grid, 
            cv=5, 
            scoring='r2', 
            n_jobs=-1
        )
        
        grid.fit(X_train, y_train)
        
        print(f"   -> 最佳参数: {grid.best_params_}")
        print(f"   -> 训练集交叉验证 R2: {grid.best_score_:.4f}")
        
        return grid.best_estimator_

    def generate_synthetic_data_with_diffusion(self, num_samples):
        """
        真正使用扩散模型生成数据
        """
        print(f"⚡ 正在使用扩散模型生成 {num_samples} 条样本...")
        
        # 1. 加载配置和模型
        config = PCCDMConfig(self.data_path)
        # 强制覆盖特征配置以匹配当前任务
        config.feature_names = self.features 
        config.input_dim = len(self.features)
        
        model = PhysicsAwareResNetModel(config).to(self.device)
        
        # 加载权重
        try:
            state_dict = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.eval()
        except Exception as e:
            print(f"❌ 加载模型权重失败: {e}")
            return None

        # 2. 准备扩散过程
        diffusion = DiffusionProcess(config)
        
        # 3. 准备条件 (Condition)
        # 这里我们简单地随机采样真实数据的条件分布，或者使用全零（取决于训练时如何使用条件）
        # 假设条件维度是 3，我们生成随机条件
        conditions = torch.randn(num_samples, config.condition_dim).to(self.device)
        
        # 4. 物理约束
        physics_constraints = [{'dose_monotonic': True} for _ in range(num_samples)]
        
        # 5. 生成
        with torch.no_grad():
            samples, _ = diffusion.generate_samples(
                model, 
                num_samples=num_samples, 
                condition_dim=config.condition_dim,
                physics_constraints=physics_constraints,
                device=self.device
            )
            
        X_synth = samples.cpu().numpy()
        
        # 简单的后处理：裁剪极端值
        X_synth = np.clip(X_synth, -10, 10)
        
        print(f"  生成数据统计 - 均值: {X_synth.mean(axis=0)}")
        return X_synth

    def label_synthetic_data(self, X_real, y_real, X_synth):
        """
        给合成特征打标签
        🚀 关键修改 2: 使用更强的基准模型来打标签，而不是简单的 KNN
        """
        print("⚡ 正在使用基准 RF 模型为合成数据生成高质量标签...")
        
        # 训练一个临时的、过拟合能力较强的 RF 模型来捕捉当前数据的细微模式
        labeler = RandomForestRegressor(n_estimators=200, max_depth=None, random_state=42)
        labeler.fit(X_real, y_real)
        
        # 预测
        y_synth = labeler.predict(X_synth)
        
        # 添加极微小的扰动，防止过拟合
        noise = np.random.normal(0, np.std(y_real) * 0.05, size=y_synth.shape)
        y_synth = y_synth + noise
        
        return y_synth

    def run_experiment(self):
        print("="*60)
        print("启动标准数据增强验证实验")
        print("="*60)
        
        # --- 修正点 1: 直接使用 __init__ 中已经清洗、填充并标准化好的数据 ---
        # 不要再从 self.df 中重新读取，否则会丢失清洗和标准化的步骤
        if not hasattr(self, 'X_processed') or not hasattr(self, 'y'):
            print("❌ 错误: 数据未正确初始化，请检查 __init__ 部分")
            return

        X = self.X_processed
        y = self.y
        
        # 2. 划分 训练集(80%) 和 测试集(20%)
        # random_state=42 保证可复现
        X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"📊 数据划分完成:")
        print(f"   - 真实训练集 (Train_Real): {len(X_train_real)} 条")
        print(f"   - 真实测试集 (Test_Real) : {len(X_test_real)} 条 (用于最终评分)")
        
        # 3. 训练基准模型 (Baseline)
        print("\n🏆 [阶段 1] 训练基准模型 (只用真实训练集)...")
        baseline_model = self.get_optimal_model(X_train_real, y_train_real, name="Baseline")
        
        # 预测测试集
        y_pred_base = baseline_model.predict(X_test_real)
        base_r2 = r2_score(y_test_real, y_pred_base)
        base_rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_base))
        
        print(f"📉 基准模型测试集表现: R2 = {base_r2:.4f}, RMSE = {base_rmse:.4f}")
        
        # 4. 生成合成数据 (Augmentation)
        n_synth = len(X_train_real) 
        print(f"\n🏭 [阶段 2] 生成 {n_synth} 条合成数据...")
        
        # --- 修正点 2: 优先调用扩散模型生成 ---
        # 确保您已经将 generate_synthetic_data_with_diffusion 方法添加到了类中
        if hasattr(self, 'generate_synthetic_data_with_diffusion'):
            X_synth = self.generate_synthetic_data_with_diffusion(n_synth)
        else:
            print("⚠️ 警告: 未找到扩散模型生成函数，正在使用旧的简单噪声生成...")
            # 如果使用旧方法，必须确保它是基于 scaled 数据生成的，这里简单处理：
            # 从训练集中采样并加噪声（仅作代码兼容，效果不如扩散模型）
            indices = np.random.choice(len(X_train_real), n_synth, replace=True)
            X_synth = X_train_real[indices].copy()
            noise = np.random.normal(0, 0.05, X_synth.shape)
            X_synth += noise

        if X_synth is None: 
            return
        
        # 打标签 (利用 Train_Real)
        y_synth = self.label_synthetic_data(X_train_real, y_train_real, X_synth)
        
        # 5. 混合数据
        X_train_aug = np.vstack([X_train_real, X_synth])
        y_train_aug = np.concatenate([y_train_real, y_synth])
        
        print(f"   - 增强后训练集: {len(X_train_aug)} 条")
        
        # 6. 训练增强模型 (Augmented)
        print("\n🚀 [阶段 3] 训练增强模型 (真实 + 合成)...")
        aug_model = self.get_optimal_model(X_train_aug, y_train_aug, name="Augmented")
        
        # 预测同一个测试集
        y_pred_aug = aug_model.predict(X_test_real)
        aug_r2 = r2_score(y_test_real, y_pred_aug)
        aug_rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_aug))
        
        print(f"📈 增强模型测试集表现: R2 = {aug_r2:.4f}, RMSE = {aug_rmse:.4f}")
        
        # 7. 最终对比与可视化
        self.visualize_comparison(y_test_real, y_pred_base, y_pred_aug, base_r2, aug_r2)
        
        # 打印结论
        improvement = (aug_r2 - base_r2)
        print("\n" + "="*60)
        print("实验结论")
        print("="*60)
        print(f"基准 R2: {base_r2:.4f}")
        print(f"增强 R2: {aug_r2:.4f}")
        print(f"绝对提升: {improvement:+.4f}")
    def visualize_comparison(self, y_true, y_base, y_aug, base_r2, aug_r2):
        plt.figure(figsize=(14, 6))
        
        # 散点图对比
        plt.subplot(1, 2, 1)
        plt.scatter(y_true, y_base, c='gray', alpha=0.6, label=f'基准 (R2={base_r2:.3f})', s=40)
        plt.scatter(y_true, y_aug, c='red', marker='x', alpha=0.7, label=f'增强 (R2={aug_r2:.3f})', s=40)
        
        # 绘制完美预测线
        min_val = min(y_true.min(), y_base.min(), y_aug.min())
        max_val = max(y_true.max(), y_base.max(), y_aug.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='完美预测')
        
        plt.xlabel('真实强度 (Standardized)')
        plt.ylabel('预测强度 (Standardized)')
        plt.title('真实测试集上的预测效果对比')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 残差分布
        plt.subplot(1, 2, 2)
        sns.kdeplot(y_true - y_base, fill=True, color='gray', label='基准误差')
        sns.kdeplot(y_true - y_aug, fill=True, color='red', label='增强误差')
        plt.axvline(0, color='k', linestyle='--')
        plt.xlabel('预测残差')
        plt.title('误差分布 (越窄越高越好)')
        plt.legend()
        
        save_path = 'results/figures/final_validation_result.png'
        plt.savefig(save_path)
        print(f"\n📊 结果图表已保存至: {save_path}")
        plt.show()

if __name__ == "__main__":
    # 配置
    DATA_PATH = "data/processed/concrete_processed_with_physics_features.csv"
    
    MODEL_PATH = "results/models/final_model.pth"
        
    print(f"使用数据增强方法")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    evaluator = StandardAugmentationEvaluator(DATA_PATH, MODEL_PATH, device)
    evaluator.run_experiment()