import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import os
import json
from datetime import datetime
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.preprocessing import StandardScaler

class IrradiationDataset(Dataset):
    '''混凝土辐照数据集'''
    
    def __init__(self, features, targets, conditions=None, physics_constraints=None):
        '''
        Args:
            features: 特征数据 [n_samples, n_features]
            targets: 目标数据 [n_samples]
            conditions: 条件信息 [n_samples, n_conditions]
            physics_constraints: 物理约束数据 [n_samples, n_constraints]
        '''
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        
        if conditions is not None:
            self.conditions = torch.FloatTensor(conditions)
        else:
            # 默认条件 - 创建一些有意义的默认值
            self.conditions = torch.randn(self.features.size(0), 3)  # 3个条件维度
        
        # 处理物理约束
        if physics_constraints is not None:
            if isinstance(physics_constraints[0], dict):
                # 如果是字典列表，转换为张量
                self.physics_constraints = self._dict_constraints_to_tensor(physics_constraints)
            else:
                # 如果已经是张量，直接使用
                self.physics_constraints = torch.FloatTensor(physics_constraints)
        else:
            # 默认物理约束
            self.physics_constraints = torch.ones(self.features.size(0), 2)  # 2个约束维度
        
    def _dict_constraints_to_tensor(self, constraints_dicts):
        '''将字典约束转换为张量'''
        constraint_tensors = []
        for constraint in constraints_dicts:
            tensor = torch.tensor([
                float(constraint.get('dose_monotonic', 1.0)),
                float(constraint.get('temperature_correlation', -0.5))
            ])
            constraint_tensors.append(tensor)
        return torch.stack(constraint_tensors)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        item = {
            'features': self.features[idx],
            'targets': self.targets[idx].unsqueeze(-1),
            'conditions': self.conditions[idx],
            'physics_constraints': self.physics_constraints[idx]
        }
        
        return item

class PCCDMTrainer:
    '''物理约束扩散模型训练器'''
    
    def __init__(self, model, diffusion, config, device='cuda'):
        self.model = model
        self.diffusion = diffusion
        self.config = config
        self.device = device
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=20
        )
        
        # 早停机制
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta
        )
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'physics_loss': [],
            'noise_loss': [],
            'learning_rates': []
        }
        
    def train_epoch(self, train_loader):
        '''训练一个epoch'''
        self.model.train()
        total_loss = 0
        total_physics_loss = 0
        total_noise_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc='训练')
        for batch in pbar:
            # 移动到设备
            features = batch['features'].to(self.device)
            conditions = batch['conditions'].to(self.device)
            physics_constraints = batch.get('physics_constraints', None)
            
            # 如果没有物理约束，创建默认值
            if physics_constraints is None:
                physics_constraints = torch.ones(features.size(0), 2).to(self.device)
            else:
                physics_constraints = physics_constraints.to(self.device)
            
            # 随机时间步
            batch_size = features.shape[0]
            t = torch.randint(0, self.diffusion.timesteps, (batch_size,), 
                            device=self.device).long()
            
            # 前向传播
            total_loss_batch, noise_loss_batch, physics_loss_batch = \
                self.diffusion.p_losses(
                    self.model, features, t, conditions, physics_constraints
                )
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            # 记录损失
            total_loss += total_loss_batch.item()
            total_physics_loss += physics_loss_batch.item()
            total_noise_loss += noise_loss_batch.item()
            num_batches += 1
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{total_loss_batch.item():.4f}',
                'physics': f'{physics_loss_batch.item():.4f}'
            })
        
        return {
            'total_loss': total_loss / num_batches,
            'physics_loss': total_physics_loss / num_batches,
            'noise_loss': total_noise_loss / num_batches
        }
    
    def validate(self, val_loader):
        '''验证'''
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(self.device)
                conditions = batch['conditions'].to(self.device)
                physics_constraints = batch.get('physics_constraints', None)
                
                if physics_constraints is None:
                    physics_constraints = torch.ones(features.size(0), 2).to(self.device)
                else:
                    physics_constraints = physics_constraints.to(self.device)
                
                batch_size = features.shape[0]
                t = torch.randint(0, self.diffusion.timesteps, (batch_size,), 
                                device=self.device).long()
                
                loss, _, _ = self.diffusion.p_losses(
                    self.model, features, t, conditions, physics_constraints
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, train_loader, val_loader, epochs):
        '''完整训练循环'''
        print("="*60)
        print("开始训练物理约束扩散模型")
        print(f"设备: {self.device}")
        print(f"训练样本: {len(train_loader.dataset)}")
        print(f"验证样本: {len(val_loader.dataset)}")
        print(f"训练轮数: {epochs}")
        print("="*60)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 40)
            
            # 训练
            train_metrics = self.train_epoch(train_loader)
            
            # 验证
            val_loss = self.validate(val_loader)
            
            # 记录历史
            self.history['train_loss'].append(train_metrics['total_loss'])
            self.history['val_loss'].append(val_loss)
            self.history['physics_loss'].append(train_metrics['physics_loss'])
            self.history['noise_loss'].append(train_metrics['noise_loss'])
            self.history['learning_rates'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # 早停检查
            self.early_stopping(val_loss)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(f'best_model_epoch_{epoch+1}.pth')
                print(f"保存最佳模型，验证损失: {val_loss:.6f}")
            
            # 定期保存检查点
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
            
            # 打印进度
            print(f"训练损失: {train_metrics['total_loss']:.6f}")
            print(f"物理损失: {train_metrics['physics_loss']:.6f}")
            print(f"噪声损失: {train_metrics['noise_loss']:.6f}")
            print(f"验证损失: {val_loss:.6f}")
            print(f"学习率: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 早停
            if self.early_stopping.early_stop:
                print(f"早停触发于 epoch {epoch+1}")
                break
        
        # 保存最终模型
        self.save_checkpoint('final_model.pth')
        
        # 保存训练历史
        self.save_training_history()
        
        return self.history
    
    def save_checkpoint(self, filename):
        '''保存模型检查点'''
        checkpoint_dir = 'results/models/checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        
        checkpoint = {
            'epoch': len(self.history['train_loss']),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'config': self.config.__dict__,
            'best_val_loss': min(self.history['val_loss']) if self.history['val_loss'] else float('inf')
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"检查点保存至: {checkpoint_path}")
    
    def save_training_history(self):
        '''保存训练历史'''
        history_dir = 'results/training_history'
        os.makedirs(history_dir, exist_ok=True)
        
        # 保存为JSON
        history_file = os.path.join(history_dir, f'training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        
        # 转换张量为列表
        serializable_history = {}
        for key, value in self.history.items():
            if isinstance(value, list):
                serializable_history[key] = [
                    float(v.item()) if torch.is_tensor(v) else float(v) 
                    for v in value
                ]
            else:
                serializable_history[key] = value
        
        with open(history_file, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        print(f"训练历史保存至: {history_file}")

class EarlyStopping:
    '''早停机制'''
    
    def __init__(self, patience=30, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class KFoldCrossValidator:
    '''K折交叉验证'''
    
    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
    def cross_validate(self, features, targets, train_func):
        '''
        执行K折交叉验证
        Args:
            features: 特征数据
            targets: 目标数据
            train_func: 训练函数，接受(train_idx, val_idx)并返回模型和指标
        '''
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(self.kfold.split(features)):
            print(f"\n{'='*60}")
            print(f"第 {fold+1}/{self.n_splits} 折交叉验证")
            print(f"训练集大小: {len(train_idx)}, 验证集大小: {len(val_idx)}")
            print('='*60)
            
            # 训练模型
            model, metrics = train_func(train_idx, val_idx)
            
            # 保存结果
            fold_results.append({
                'fold': fold + 1,
                'train_indices': train_idx.tolist(),
                'val_indices': val_idx.tolist(),
                'metrics': metrics,
                'model_path': f'results/models/fold_{fold+1}_model.pth'
            })
            
            # 保存模型
            torch.save(model.state_dict(), f'results/models/fold_{fold+1}_model.pth')
        
        # 计算平均指标
        avg_metrics = self.compute_average_metrics(fold_results)
        
        # 保存交叉验证结果
        self.save_cv_results(fold_results, avg_metrics)
        
        return fold_results, avg_metrics
    
    def compute_average_metrics(self, fold_results):
        '''计算平均指标'''
        all_metrics = {}
        
        for result in fold_results:
            for metric_name, metric_value in result['metrics'].items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(metric_value)
        
        avg_metrics = {
            metric_name: {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values
            }
            for metric_name, values in all_metrics.items()
        }
        
        return avg_metrics
    
    def save_cv_results(self, fold_results, avg_metrics):
        '''保存交叉验证结果'''
        cv_dir = 'results/cross_validation'
        os.makedirs(cv_dir, exist_ok=True)
        
        results_file = os.path.join(cv_dir, f'cv_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        
        results = {
            'n_splits': self.n_splits,
            'random_state': self.random_state,
            'fold_results': fold_results,
            'average_metrics': avg_metrics
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: float(x) if torch.is_tensor(x) else x)
        
        print(f"交叉验证结果保存至: {results_file}")
        
        # 打印摘要
        print("\n交叉验证结果摘要:")
        print("-" * 60)
        for metric_name, stats in avg_metrics.items():
            print(f"{metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                  f"(范围: {stats['min']:.4f} - {stats['max']:.4f})")

def prepare_dataloaders(features, targets, conditions=None, 
                       physics_constraints=None, config=None):
    '''准备数据加载器'''
    if config is None:
        config = type('Config', (), {
            'batch_size': 32,
            'val_split': 0.2,
            'random_state': 42
        })()
    
    # 划分训练验证集
    n_samples = len(features)
    indices = np.random.RandomState(config.random_state).permutation(n_samples)
    split_idx = int(n_samples * (1 - config.val_split))
    
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    
    # 创建数据集
    train_dataset = IrradiationDataset(
        features=features[train_idx],
        targets=targets[train_idx],
        conditions=conditions[train_idx] if conditions is not None else None,
        physics_constraints=[physics_constraints[i] for i in train_idx] 
        if physics_constraints else None
    )
    
    val_dataset = IrradiationDataset(
        features=features[val_idx],
        targets=targets[val_idx],
        conditions=conditions[val_idx] if conditions is not None else None,
        physics_constraints=[physics_constraints[i] for i in val_idx] 
        if physics_constraints else None
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"数据加载器准备完成:")
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(val_dataset)} 样本")
    print(f"  批次大小: {config.batch_size}")
    
    return train_loader, val_loader

# 训练配置
class TrainingConfig:
    def __init__(self, csv_path="data/processed/concrete_processed_with_physics_features.csv"):
        # 从CSV数据中获取实际的特征维度
        df = pd.read_csv(csv_path)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if '相对抗压强度' in numeric_cols:
            numeric_cols.remove('相对抗压强度')
        self.input_dim = len(numeric_cols)  # 根据实际数据调整
        self.condition_dim = 3  # 条件维度
        self.hidden_dim = 128
        self.num_layers = 4
        self.physics_weight = 0.1  # 物理损失权重

        # 数据参数
        self.batch_size = 32
        self.val_split = 0.2
        self.random_state = 42
        
        # 训练参数
        self.epochs = 500
        self.learning_rate = 1e-3
        self.weight_decay = 1e-4
        
        # 早停参数
        self.patience = 30
        self.min_delta = 1e-4
        
        # 保存参数
        self.save_interval = 50
        
        # 扩散模型参数
        self.timesteps = 1000
        self.beta_start = 0.0001
        self.beta_end = 0.02
        self.schedule = 'cosine'

def create_physics_constraints_from_data(df):
    '''从数据中创建物理约束张量'''
    constraints = []
    
    # 使用数据中的物理特征作为约束
    # 选择几个相关的物理特征列
    physics_feature_cols = [
        'physics_constraint_1',
        'physics_constraint_2', 
        'physics_constraint_3',
        'defect_density',
        'amorphous_fraction',
        'crack_density'
    ]
    
    # 只选择实际存在的列
    available_cols = [col for col in physics_feature_cols if col in df.columns]
    
    if not available_cols:
        # 如果没有找到物理特征列，创建默认约束
        print("警告：未找到物理特征列，使用默认约束")
        for _ in range(len(df)):
            constraints.append({
                'dose_monotonic': 1.0,
                'temperature_correlation': -0.5
            })
    else:
        # 使用现有的物理特征
        print(f"使用物理特征列: {available_cols}")
        for _, row in df.iterrows():
            constraint_dict = {}
            # 前两个约束使用特定的键名
            if len(available_cols) >= 1:
                constraint_dict['dose_monotonic'] = float(row[available_cols[0]])
            if len(available_cols) >= 2:
                constraint_dict['temperature_correlation'] = float(row[available_cols[1]])
            
            # 添加其他约束作为额外信息
            for i, col in enumerate(available_cols[2:], start=3):
                constraint_dict[f'constraint_{i}'] = float(row[col])
            
            constraints.append(constraint_dict)
    
    return constraints

if __name__ == "__main__":
    # 从实际数据中加载
    print("从实际数据加载开始训练管道")
    print("=" * 60)
    
    # 读取CSV数据
    csv_path = "data/processed/concrete_processed_with_physics_features.csv"
    df = pd.read_csv(csv_path)
    print(f"加载数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    # 获取数值列作为特征
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target_col = '相对抗压强度'
    
    if target_col in numeric_cols:
        feature_cols = [col for col in numeric_cols if col != target_col]
    else:
        # 如果没有找到目标列，选择除了最后一个特征外的所有数值列
        feature_cols = numeric_cols[:-1]
        target_col = numeric_cols[-1]  # 假设最后一列是目标列
    
    print(f"特征列: {feature_cols}")
    print(f"目标列: {target_col}")
    print(f"特征数量: {len(feature_cols)}")
    
    # 准备特征和目标
    features = df[feature_cols].values.astype(np.float32)
    targets = df[target_col].values.astype(np.float32)
    
    # 数据标准化 - 修复数据中可能存在的问题
    print("标准化数据...")
    # 检查是否有无穷大或NaN值
    features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
    targets = np.nan_to_num(targets, nan=0.0, posinf=1e6, neginf=-1e6)
    
    feature_scaler = StandardScaler()
    features_scaled = feature_scaler.fit_transform(features)
    
    target_scaler = StandardScaler()
    targets_scaled = target_scaler.fit_transform(targets.reshape(-1, 1)).flatten()
    
    print(f"特征形状: {features_scaled.shape}")
    print(f"目标形状: {targets_scaled.shape}")
    print(f"特征范围: [{features_scaled.min():.2f}, {features_scaled.max():.2f}]")
    print(f"目标范围: [{targets_scaled.min():.2f}, {targets_scaled.max():.2f}]")
    
    # 创建物理约束（从数据中提取）
    physics_constraints = create_physics_constraints_from_data(df)
    print(f"创建了 {len(physics_constraints)} 个物理约束")
    
    # 创建配置
    config = TrainingConfig(csv_path)
    print(f"模型输入维度: {config.input_dim}")
    print(f"条件维度: {config.condition_dim}")
    
    # 准备数据加载器
    train_loader, val_loader = prepare_dataloaders(
        features_scaled, targets_scaled, physics_constraints=physics_constraints, config=config
    )
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    from pccdm_model import PhysicsAwareConditionalDiffusionModel, DiffusionProcess, PCCDMConfig as ModelPCCDMConfig
    
    # 使用实际的模型配置
    model_config = ModelPCCDMConfig(csv_path)
    model = PhysicsAwareConditionalDiffusionModel(model_config).to(device)
    diffusion = DiffusionProcess(model_config)
    
    # 创建训练器
    trainer = PCCDMTrainer(model, diffusion, config, device)
    
    # 开始训练
    print("\n开始训练...")
    history = trainer.train(train_loader, val_loader, epochs=500)  # 先训练100轮查看效果
    
    print("\n训练完成！")
    print(f"最终训练损失: {history['train_loss'][-1]:.6f}")
    print(f"最终验证损失: {history['val_loss'][-1]:.6f}")
    if history['val_loss']:
        print(f"最佳验证损失: {min(history['val_loss']):.6f}")