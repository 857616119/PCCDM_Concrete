#!/usr/bin/env python3
'''
主执行流程：物理约束扩散模型用于混凝土辐照数据增强
'''

import os
import sys
import argparse
import yaml
import json
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_config(config_path='config.yaml'):
    '''加载配置文件'''
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 转换为对象以便点访问
    class ConfigObject:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                if isinstance(value, dict):
                    setattr(self, key, ConfigObject(value))
                else:
                    setattr(self, key, value)
    
    return ConfigObject(config)

def setup_environment(config):
    '''设置运行环境'''
    print("="*80)
    print("物理约束扩散模型 - 混凝土辐照数据增强系统")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # 设置随机种子
    if hasattr(config, 'random_seed'):
        np.random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.random_seed)
    
    # 创建目录结构
    directories = [
        'data/raw',
        'data/processed', 
        'data/synthetic',
        'results/models',
        'results/figures',
        'results/evaluation',
        'results/training_history',
        'results/cross_validation',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"创建目录: {directory}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"PyTorch版本: {torch.__version__}")
    
    return device

def run_data_preprocessing(config):
    '''运行数据预处理'''
    print("\n" + "="*80)
    print("阶段1: 数据预处理")
    print("="*80)
    
    try:
        from data_preprocessing import ConcreteDataPreprocessor
        
        preprocessor = ConcreteDataPreprocessor(config)
        
        # 加载数据
        df = preprocessor.load_data(config.data_path)
        
        # 处理缺失值
        df = preprocessor.handle_missing_values(df)
        
        # 编码分类变量
        df = preprocessor.encode_categorical(df)
        
        # 特征标准化
        feature_cols = getattr(config, 'feature_columns', 
                             ['neutron_fluence', 'temperature', 'age_days'])
        df, _ = preprocessor.scale_features(df, feature_cols)
        
        # 目标变量标准化
        target_col = getattr(config, 'target_column', 'compressive_strength')
        df, _ = preprocessor.scale_target(df, target_col)
        
        # 划分数据
        test_size = getattr(config, 'test_size', 0.2)
        train_df, test_df = preprocessor.split_data(df, test_size, config.random_seed)
        
        # 保存处理后的数据
        processed_path = 'data/processed/concrete_processed.csv'
        metadata = preprocessor.save_processed_data(df, processed_path)
        
        print(f"\n数据预处理完成!")
        print(f"训练集: {train_df.shape}")
        print(f"测试集: {test_df.shape}")
        
        return df, train_df, test_df, metadata
        
    except Exception as e:
        print(f"数据预处理错误: {e}")
        raise

def run_physics_feature_engineering(df, config):
    '''运行物理特征工程'''
    print("\n" + "="*80)
    print("阶段2: 物理特征工程")
    print("="*80)
    
    try:
        from physics_feature_engineering import PhysicsFeatureEngineer
        
        engineer = PhysicsFeatureEngineer()
        
        # 创建物理引导特征
        df_enhanced, physics_features = engineer.create_irradiation_damage_features(df)
        
        # 创建微观结构特征
        micro_features = engineer.create_microstructure_features(df)
        
        # 合并所有特征
        df_final = pd.concat([df_enhanced, micro_features], axis=1)
        
        # 计算物理约束
        target_col = getattr(config, 'target_column', 'compressive_strength')
        constraints = engineer.calculate_physics_constraints(df_final, target_col)
        
        # 验证特征有效性
        validation = engineer.validate_physics_features(
            df_final, 
            list(physics_features) + list(micro_features.columns),
            target_col
        )
        
        print(f"\n物理特征工程完成!")
        print(f"原始特征数: {len(df.columns)}")
        print(f"增强后特征数: {len(df_final.columns)}")
        print(f"物理约束数: {len(constraints)}")
        
        # 保存增强数据
        enhanced_path = 'data/processed/concrete_enhanced.csv'
        df_final.to_csv(enhanced_path, index=False)
        
        # 保存约束
        constraints_path = 'data/processed/physics_constraints.json'
        with open(constraints_path, 'w') as f:
            json.dump(constraints, f, indent=2)
        
        return df_final, physics_features, micro_features.columns.tolist(), constraints
        
    except Exception as e:
        print(f"物理特征工程错误: {e}")
        raise

def prepare_diffusion_data(df_final, config):
    '''准备扩散模型数据'''
    print("\n" + "="*80)
    print("阶段3: 准备扩散模型数据")
    print("="*80)
    
    try:
        from data_preprocessing import ConcreteDataPreprocessor
        
        preprocessor = ConcreteDataPreprocessor(config)
        
        # 选择扩散模型使用的特征
        diffusion_features = getattr(config, 'diffusion_features', [
            'concrete_type_encoded',
            'neutron_fluence_scaled',
            'temperature_scaled', 
            'age_days_scaled',
            'compressive_strength_scaled'
        ])
        
        # 过滤存在的特征
        available_features = [f for f in diffusion_features if f in df_final.columns]
        X_diffusion = df_final[available_features].values
        
        print(f"扩散模型数据形状: {X_diffusion.shape}")
        print(f"使用特征: {available_features}")
        
        # 准备条件数据（可选）
        condition_features = getattr(config, 'condition_features', [])
        if condition_features:
            available_conditions = [f for f in condition_features if f in df_final.columns]
            conditions = df_final[available_conditions].values
            print(f"条件数据形状: {conditions.shape}")
        else:
            conditions = None
        
        return X_diffusion, conditions, available_features
        
    except Exception as e:
        print(f"准备扩散模型数据错误: {e}")
        raise

def run_diffusion_model_training(X_diffusion, conditions, physics_constraints, 
                                feature_names, config, device):
    '''运行扩散模型训练'''
    print("\n" + "="*80)
    print("阶段4: 物理约束扩散模型训练")
    print("="*80)
    
    try:
        from pccdm_model import (
            PhysicsAwareConditionalDiffusionModel, 
            DiffusionProcess, 
            PCCDMConfig
        )
        
        from training_pipeline import (
            PCCDMTrainer, 
            prepare_dataloaders,
            IrradiationDataset
        )
        
        # 创建模型配置
        pccdm_config = PCCDMConfig()
        pccdm_config.input_dim = X_diffusion.shape[1]
        pccdm_config.condition_dim = conditions.shape[1] if conditions is not None else 3
        pccdm_config.feature_names = feature_names
        
        # 更新配置
        if hasattr(config, 'diffusion'):
            for key, value in config.diffusion.items():
                setattr(pccdm_config, key, value)
        
        # 创建模型
        model = PhysicsAwareConditionalDiffusionModel(pccdm_config).to(device)
        diffusion = DiffusionProcess(pccdm_config)
        
        # 准备数据加载器
        # 使用目标变量作为训练数据（扩散模型学习数据分布）
        targets = X_diffusion[:, -1].reshape(-1, 1)  # 最后一列是目标
        
        # 创建物理约束列表（每个样本一个）
        if physics_constraints and isinstance(physics_constraints, dict):
            # 如果是单个字典，为每个样本复制
            n_samples = len(X_diffusion)
            physics_constraints_list = [physics_constraints] * n_samples
        else:
            physics_constraints_list = None
        
        train_loader, val_loader = prepare_dataloaders(
            X_diffusion, targets, conditions, physics_constraints_list, pccdm_config
        )
        
        # 创建训练器
        trainer = PCCDMTrainer(model, diffusion, pccdm_config, device)
        
        # 开始训练
        epochs = getattr(config, 'training_epochs', 500)
        history = trainer.train(train_loader, val_loader, epochs)
        
        print(f"\n扩散模型训练完成!")
        print(f"最终训练损失: {history['train_loss'][-1]:.6f}")
        print(f"最终验证损失: {history['val_loss'][-1]:.6f}")
        
        return model, diffusion, history
        
    except Exception as e:
        print(f"扩散模型训练错误: {e}")
        raise

def generate_synthetic_data(model, diffusion, config, device, 
                           num_samples=None, conditions=None):
    '''生成合成数据'''
    print("\n" + "="*80)
    print("阶段5: 生成合成数据")
    print("="*80)
    
    try:
        if num_samples is None:
            num_samples = getattr(config, 'synthetic_samples', 1000)
        
        # 准备物理约束
        physics_constraints = {
            'dose_monotonic': True,
            'temperature_correlation': -0.5,
            'expected_trends': {
                'fluence_strength': 'decreasing',
                'temperature_strength': 'negative'
            }
        }
        
        # 生成条件（如果需要）
        if conditions is None:
            condition_dim = model.config.condition_dim
            conditions = torch.randn(num_samples, condition_dim, device=device)
        
        # 生成样本
        print(f"生成 {num_samples} 个合成样本...")
        
        shape = (num_samples, model.config.input_dim)
        
        with torch.no_grad():
            samples = diffusion.p_sample_loop(
                model, shape, conditions, physics_constraints, device, progress=True
            )
        
        # 转换为numpy
        synthetic_data = samples.cpu().numpy()
        
        print(f"合成数据形状: {synthetic_data.shape}")
        print(f"数据范围: [{synthetic_data.min():.3f}, {synthetic_data.max():.3f}]")
        
        # 保存合成数据
        synthetic_path = 'data/synthetic/concrete_synthetic.npy'
        np.save(synthetic_path, synthetic_data)
        
        # 保存为CSV
        feature_names = model.config.feature_names
        synthetic_df = pd.DataFrame(synthetic_data, columns=feature_names)
        synthetic_csv_path = 'data/synthetic/concrete_synthetic.csv'
        synthetic_df.to_csv(synthetic_csv_path, index=False)
        
        print(f"合成数据保存至:")
        print(f"  {synthetic_path}")
        print(f"  {synthetic_csv_path}")
        
        return synthetic_data, synthetic_df
        
    except Exception as e:
        print(f"生成合成数据错误: {e}")
        raise

def run_comprehensive_evaluation(original_data, synthetic_data, feature_names, 
                                model=None, config=None):
    """运行综合评估 - 修复函数名冲突"""
    print("\n" + "="*80)
    print("阶段6: 综合评估")
    print("="*80)
    
    try:
        from evaluation_framework import run_comprehensive_evaluation as run_eval
        
        # 准备背景数据（用于SHAP分析）
        background_data = original_data[:100] if len(original_data) > 100 else original_data  # 使用部分原始数据
        
        # 运行评估
        evaluation_results = run_eval(
            original_data, synthetic_data, feature_names, model, background_data
        )
        
        print(f"\n综合评估完成!")
        
        return evaluation_results
        
    except Exception as e:
        print(f"综合评估错误: {e}")
        raise

def run_downstream_analysis(original_data, synthetic_data, feature_names, config):
    """运行下游分析"""
    print("\n" + "="*80)
    print("阶段7: 下游GBDT预测分析")
    print("="*80)
    
    try:
        from downstream_prediction_last import run_downstream_analysis as run_downstream
        
        # 运行下游分析
        downstream_results = run_downstream(
            original_data, synthetic_data, feature_names, 
            target_idx=-1,  # 假设最后一列是目标
            test_size=getattr(config, 'test_size', 0.2)
        )
        
        print(f"\n下游分析完成!")
        
        return downstream_results
        
    except Exception as e:
        print(f"下游分析错误: {e}")
        raise

def generate_final_report(config, results_dict, execution_time):
    '''生成最终报告'''
    print("\n" + "="*80)
    print("阶段8: 生成最终报告")
    print("="*80)
    
    try:
        report = {
            'project': '物理约束扩散模型用于混凝土辐照数据增强',
            'execution_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'execution_time_seconds': execution_time,
            'config': {
                k: str(v) for k, v in config.__dict__.items() 
                if not k.startswith('_')
            },
            'results_summary': {}
        }
        
        # 提取关键结果
        if 'evaluation' in results_dict:
            eval_results = results_dict['evaluation']
            if 'data_quality' in eval_results:
                report['results_summary']['data_quality'] = {
                    'mmd_score': eval_results['data_quality'].get('mmd_score', 'N/A'),
                    'average_kld': eval_results['data_quality'].get('average_kld', 'N/A'),
                    'correlation_similarity': eval_results['data_quality'].get('correlation_similarity', 'N/A')
                }
        
        if 'downstream' in results_dict:
            downstream_results = results_dict['downstream']
            if 'benchmark_results' in downstream_results:
                # 找到最佳模型
                benchmark = downstream_results['benchmark_results']
                if benchmark:
                    best_model = max(benchmark.keys(), 
                                   key=lambda x: benchmark[x]['test_metrics']['test_r2'])
                    report['results_summary']['best_model'] = {
                        'name': best_model,
                        'test_r2': benchmark[best_model]['test_metrics']['test_r2'],
                        'test_rmse': benchmark[best_model]['test_metrics']['test_rmse']
                    }
        
        # 保存报告
        report_path = 'results/final_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 生成Markdown报告
        markdown_report = generate_markdown_report(report, results_dict)
        md_path = 'results/final_report.md'
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(markdown_report)
        
        print(f"最终报告保存至:")
        print(f"  {report_path}")
        print(f"  {md_path}")
        
        # 打印摘要
        print("\n" + "="*80)
        print("项目执行摘要")
        print("="*80)
        print(f"执行时间: {execution_time:.1f} 秒")
        
        if 'data_quality' in report['results_summary']:
            dq = report['results_summary']['data_quality']
            print(f"数据质量:")
            print(f"  MMD分数: {dq.get('mmd_score', 'N/A'):.6f}")
            print(f"  平均KL散度: {dq.get('average_kld', 'N/A'):.6f}")
            print(f"  相关性保留度: {dq.get('correlation_similarity', 'N/A'):.4f}")
        
        if 'best_model' in report['results_summary']:
            bm = report['results_summary']['best_model']
            print(f"最佳模型: {bm['name']}")
            print(f"  测试R²: {bm['test_r2']:.4f}")
            print(f"  测试RMSE: {bm['test_rmse']:.4f}")
        
        print("\n所有结果文件:")
        for root, dirs, files in os.walk('results'):
            for file in files:
                if file.endswith(('.png', '.json', '.csv', '.pkl')):
                    filepath = os.path.join(root, file)
                    print(f"  {filepath}")
        
        return report
        
    except Exception as e:
        print(f"生成报告错误: {e}")
        raise

def generate_markdown_report(report, results_dict):
    '''生成Markdown格式报告'''
    md = f"""
# 物理约束扩散模型用于混凝土辐照数据增强 - 实验报告

## 1. 项目概述

- **项目名称**: {report['project']}
- **执行时间**: {report['execution_date']}
- **总执行时间**: {report['execution_time_seconds']:.1f} 秒

## 2. 配置参数

```yaml
{json.dumps(report['config'], indent=2)}
"""