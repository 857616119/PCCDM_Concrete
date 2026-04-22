# -*- coding: utf-8 -*-
"""
批量稳健性验证脚本 (Robustness Validation Loop)
功能：运行多次实验，验证数据增强效果的统计显著性
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import torch
import sys
import os
import warnings
from tqdm import tqdm

# 引用你刚才成功的脚本中的类
from downstream_prediction_copy import StandardAugmentationEvaluator

# 设置绘图风格
import matplotlib
matplotlib.use('TkAgg')
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

def run_robustness_check(n_experiments=20):
    print("="*60)
    print(f"🚀 启动批量稳健性验证 (共 {n_experiments} 次实验)")
    print("="*60)
    
    # 配置路径
    DATA_PATH = "data/processed/concrete_processed_with_physics_features.csv"
    model_dir = "results/models"
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    if 'final_model.pth' in model_files:
        MODEL_PATH = os.path.join(model_dir, 'final_model.pth')
    else:
        MODEL_PATH = os.path.join(model_dir, model_files[-1])
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 记录结果
    results = {
        'seed': [],
        'base_r2': [],
        'aug_r2': [],
        'diff': [],
        'base_rmse': [],
        'aug_rmse': []
    }
    
    # 循环实验
    for i in tqdm(range(n_experiments), desc="Running Experiments"):
        seed = 42 + i  # 每次变更加随机种子
        
        # 初始化评估器
        evaluator = StandardAugmentationEvaluator(DATA_PATH, MODEL_PATH, device)
        
        # --- 核心逻辑复刻 (为了能传入 seed，我们需要稍微修改调用方式或在这里重写核心流) ---
        # 为了方便，我们直接调用 evaluator 的方法，但手动控制 split 的 seed
        
        # 1. 准备数据
        X = evaluator.df[evaluator.features].values
        y = evaluator.df[evaluator.target].values
        X = np.nan_to_num(X, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)
        
        # 手动切分 (使用当前循环的 seed)
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score, mean_squared_error
        
        X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )
        
        # 2. 基准模型
        baseline_model = evaluator.get_optimal_model(X_train_real, y_train_real, name=f"Base-{seed}")
        y_pred_base = baseline_model.predict(X_test_real)
        base_r2 = r2_score(y_test_real, y_pred_base)
        base_rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_base))
        
        # 3. 生成数据 (1:1)
        n_synth = len(X_train_real)
        # 注意：这里需要在 generator 内部也设置 seed 才能保证生成多样性，但现在的代码结构可能固定了。
        # 不过 train_test_split 的变化本身就足够验证稳健性了。
        X_synth = evaluator.generate_synthetic_features(n_synth)
        if X_synth is None: continue
            
        y_synth = evaluator.label_synthetic_data(X_train_real, y_train_real, X_synth)
        
        # 4. 增强训练
        X_train_aug = np.vstack([X_train_real, X_synth])
        y_train_aug = np.concatenate([y_train_real, y_synth])
        
        aug_model = evaluator.get_optimal_model(X_train_aug, y_train_aug, name=f"Aug-{seed}")
        y_pred_aug = aug_model.predict(X_test_real)
        aug_r2 = r2_score(y_test_real, y_pred_aug)
        aug_rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_aug))
        
        # 记录
        results['seed'].append(seed)
        results['base_r2'].append(base_r2)
        results['aug_r2'].append(aug_r2)
        results['diff'].append(aug_r2 - base_r2)
        results['base_rmse'].append(base_rmse)
        results['aug_rmse'].append(aug_rmse)
        
        # 打印单次简报
        print(f"Seed {seed}: Base={base_r2:.4f} -> Aug={aug_r2:.4f} (Diff={aug_r2-base_r2:+.4f})")

    # 统计分析
    df_res = pd.DataFrame(results)
    mean_diff = df_res['diff'].mean()
    positive_ratio = (df_res['diff'] > 0).mean()
    
    # T检验
    t_stat, p_val = stats.ttest_rel(df_res['base_r2'], df_res['aug_r2'], alternative='less')
    # 注意：scipy的单尾检验假设是 popmean < 0 (即base < aug)，所以 p < 0.05 意味着显著提升
    
    print("\n" + "="*60)
    print("📊 批量实验统计报告")
    print("="*60)
    print(f"平均提升 (Delta R2): {mean_diff:+.4f}")
    print(f"正向提升比例: {positive_ratio:.1%} ({sum(df_res['diff']>0)}/{n_experiments})")
    print(f"P-value (单尾T检验): {p_val:.4f}")
    
    if p_val < 0.05:
        print("✅ 结果具有统计显著性 (Significantly Positive)！")
    else:
        print("⚠️ 结果在统计上不显著，可能是方差太大。")
        
    # 绘图
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_res[['base_r2', 'aug_r2']], palette=['gray', 'red'])
    sns.swarmplot(data=df_res[['base_r2', 'aug_r2']], color='black', alpha=0.5)
    
    # 连线显示配对变化
    for i in range(len(df_res)):
        plt.plot([0, 1], [df_res['base_r2'][i], df_res['aug_r2'][i]], 'k-', alpha=0.2)
        
    plt.title(f'基准 vs 增强 (N={n_experiments})\n平均提升: {mean_diff:+.4f} (P={p_val:.3f})')
    plt.ylabel('Test R2 Score')
    plt.savefig('results/figures/robustness_check_boxplot.png')
    plt.show()
    
    # 保存详细数据
    df_res.to_csv('results/evaluation/robustness_results.csv', index=False)
    print("详细结果已保存至 results/evaluation/robustness_results.csv")

if __name__ == "__main__":
    # 建议跑 10-20 次
    run_robustness_check(n_experiments=30)