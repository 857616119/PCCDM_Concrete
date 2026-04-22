import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

# 读取数据
df = pd.read_csv('data/processed/concrete_processed.csv')

# 1. 清洗 '骨料水泥比'，将其转换为数值
def clean_aggregate_ratio(x):
    if isinstance(x, str):
        if ',' in x:
            vals = [float(v) for v in x.split(',')]
            return sum(vals) / len(vals) # 取范围的平均值
        return float(x)
    return float(x)

df['骨料水泥比_numeric'] = df['骨料水泥比'].apply(clean_aggregate_ratio)

# 2. 填补 '水灰比' 缺失值
# 使用 KNN 填补，利用其他数值特征辅助推断
numeric_cols_impute = ['水灰比', '骨料水泥比_numeric', '快中子辐照注量 n/cm2', '辐照温度', '相对抗压强度']
imputer = KNNImputer(n_neighbors=5)
df_imputed_array = imputer.fit_transform(df[numeric_cols_impute])
df_imputed = pd.DataFrame(df_imputed_array, columns=numeric_cols_impute)
df['水灰比'] = df_imputed['水灰比'] # 更新填补后的值

# 3. 数据增强 (Mixup 策略)
def mixup_data(df, target_count=500):
    current_count = len(df)
    needed = target_count - current_count
    new_rows = []
    
    # 区分连续数值列和离散/类别列
    float_cols = df.select_dtypes(include=['float']).columns.tolist()
    other_cols = [c for c in df.columns if c not in float_cols and c != '数据ID']
    
    for _ in range(needed):
        # 随机选择两个样本
        idx1, idx2 = np.random.choice(df.index, 2, replace=False)
        row1 = df.loc[idx1]
        row2 = df.loc[idx2]
        
        # 生成混合比例 lambda
        lam = np.random.beta(0.2, 0.2)
        
        new_row = {}
        # 数值型特征：线性混合
        for col in float_cols:
            new_row[col] = lam * row1[col] + (1 - lam) * row2[col]
            
        # 类别型特征：按权重选择其中一个父样本的值
        parent_row = row1 if lam > 0.5 else row2
        for col in other_cols:
            new_row[col] = parent_row[col]
            
        new_rows.append(new_row)
        
    return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

# 执行增强
df_augmented = mixup_data(df, target_count=800)

# 重置ID
df_augmented['数据ID'] = range(1, len(df_augmented) + 1)

# 保存
df_augmented.to_csv('data/processed/concrete_augmented.csv', index=False)
print("数据增强完成，当前数据量:", len(df_augmented))