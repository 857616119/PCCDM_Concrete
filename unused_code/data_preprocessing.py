import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
import os
import re
warnings.filterwarnings('ignore')

class ConcreteDataPreprocessor:
    '''
    混凝土辐照数据预处理模块
    处理国家科学数据中心公开数据集
    '''
    
    def __init__(self, config):
        self.config = config
        self.scalers = {}
        
    def load_data(self, data_path):
        '''
        加载并验证数据集
        Args:
            data_path: 数据集Excel文件路径
        Returns:
            DataFrame: 原始数据
        '''
        print("="*50)
        print("步骤1: 数据加载与验证")
        print("="*50)
        
        # 检查文件扩展名以决定使用哪种读取方法
        if data_path.endswith('.xlsx') or data_path.endswith('.xls'):
            # 读取Excel文件的所有工作表
            all_sheets = pd.read_excel(data_path, sheet_name=None)
            print(f"Excel文件包含 {len(all_sheets)} 个工作表: {list(all_sheets.keys())}")
            
            # 打印每个工作表的形状和列名
            for sheet_name, sheet_df in all_sheets.items():
                print(f"工作表 '{sheet_name}' 形状: {sheet_df.shape}")
                print(f"工作表 '{sheet_name}' 列名: {list(sheet_df.columns)}")
            
            # 重点关注"国内外公开的混凝土中子辐照试验数据"和"构筑物材料基本信息"工作表
            if '国内外公开的混凝土中子辐照试验数据' in all_sheets and '构筑物材料基本信息' in all_sheets:
                # 读取两个关键工作表
                irradiation_data = all_sheets['国内外公开的混凝土中子辐照试验数据'].copy()
                material_data = all_sheets['构筑物材料基本信息'].copy()
                
                print(f"\n'国内外公开的混凝土中子辐照试验数据' 工作表形状: {irradiation_data.shape}")
                print(f"'构筑物材料基本信息' 工作表形状: {material_data.shape}")
                
                # 检查是否有共同的合并键（如数据ID）
                common_cols = set(irradiation_data.columns) & set(material_data.columns)
                print(f"共同列: {common_cols}")
                
                # 使用数据ID进行合并
                if '数据ID' in irradiation_data.columns and '数据ID' in material_data.columns:
                    # 使用数据ID作为键进行合并
                    df = pd.merge(material_data, irradiation_data, on='数据ID', how='inner')
                    print(f"合并后数据形状: {df.shape}")
                    print(f"合并后列名: {list(df.columns)}")
                else:
                    print("警告: 未找到'数据ID'列，将使用索引进行合并")
                    # 如果没有共同的ID列，使用索引进行合并
                    irradiation_data_reset = irradiation_data.reset_index(drop=True)
                    material_data_reset = material_data.reset_index(drop=True)
                    
                    # 确保长度相同
                    min_len = min(len(irradiation_data_reset), len(material_data_reset))
                    irradiation_data_reset = irradiation_data_reset.iloc[:min_len]
                    material_data_reset = material_data_reset.iloc[:min_len]
                    
                    # 合并两个数据框，避免重复列
                    df = pd.concat([
                        material_data_reset.reset_index(drop=True), 
                        irradiation_data_reset.drop(columns=['序号'], errors='ignore').reset_index(drop=True)
                    ], axis=1)
                    
                    # 删除重复的列（保留一份）
                    df = df.loc[:, ~df.columns.duplicated()]
                    
                    print(f"合并后数据形状: {df.shape}")
                    print(f"合并后列名: {list(df.columns)}")
            else:
                print("错误: 未找到关键工作表，使用所有工作表合并")
                # 合并所有工作表
                df_list = []
                for sheet_name, sheet_df in all_sheets.items():
                    sheet_df['source_sheet'] = sheet_name
                    df_list.append(sheet_df)
                
                # 合并所有工作表
                df = pd.concat(df_list, ignore_index=True)
        else:
            # 加载CSV数据
            df = pd.read_csv(data_path)
        
        print(f"最终数据集形状: {df.shape}")
        print(f"数据列名: {list(df.columns)}")
        print(f"数据类型:\n{df.dtypes}")
        print(f"\n数据概览:")
        print(df.head())
        
        # 保存列信息供后续使用
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        return df

    def clean_numeric_columns(self, df):
        '''清理数值列中的非数值字符'''
        print("\n清理数值列中的非数值字符...")
        
        # 清理'快中子辐照注量 n/cm2'列
        if '快中子辐照注量 n/cm2' in df.columns:
            def clean_fluence(val):
                if pd.isna(val) or val == '':
                    return np.nan
                if isinstance(val, str):
                    # 提取数字部分
                    match = re.search(r'[\d.]+(?:e[+-]?\d+)?', val)
                    if match:
                        return float(match.group())
                    else:
                        return np.nan
                return float(val)
            df['快中子辐照注量 n/cm2'] = df['快中子辐照注量 n/cm2'].apply(clean_fluence)
            print(f"'快中子辐照注量 n/cm2'列已清理，非数值项转为NaN")
        
        # 清理'辐照温度'列
        if '辐照温度' in df.columns:
            def clean_temperature(val):
                if pd.isna(val) or val == '':
                    return np.nan
                if isinstance(val, str):
                    # 如果是范围如'20~100'，取平均值
                    if '~' in val:
                        try:
                            parts = val.split('~')
                            if len(parts) == 2:
                                low = float(re.search(r'[\d.]+', parts[0]).group()) if re.search(r'[\d.]+', parts[0]) else np.nan
                                high = float(re.search(r'[\d.]+', parts[1]).group()) if re.search(r'[\d.]+', parts[1]) else np.nan
                                return (low + high) / 2 if not (pd.isna(low) or pd.isna(high)) else np.nan
                        except:
                            return np.nan
                    # 如果是单个数字
                    match = re.search(r'[\d.]+', val)
                    if match:
                        return float(match.group())
                    else:
                        return np.nan
                return float(val)
            df['辐照温度'] = df['辐照温度'].apply(clean_temperature)
            print(f"'辐照温度'列已清理，非数值项转为NaN")
        
        # 更新数值列和分类列列表
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        return df
    
    def handle_missing_values(self, df):
        '''处理缺失值 - 采用更合理的策略'''
        print("\n处理缺失值...")
        initial_shape = df.shape
        
        print(f"原始数据形状: {initial_shape}")
        
        # 1. 首先删除缺失值超过50%的列
        missing_threshold = 0.5  # 缺失值超过50%的列被删除
        total_rows = len(df)
        
        columns_to_drop = []
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_ratio = missing_count / total_rows
            if missing_ratio > missing_threshold:
                columns_to_drop.append(col)
                print(f"  列 '{col}': {missing_count}/{total_rows} ({missing_ratio:.2%}) 缺失 -> 删除")
        
        df = df.drop(columns=columns_to_drop)
        print(f"删除 {len(columns_to_drop)} 个高缺失率列")
        
        # 更新数值列和分类列列表
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # 2. 对于剩余列，采用不同的策略
        print("处理剩余列的缺失值...")
        
        # 数值列：如果缺失率较低，用中位数填充；否则保留NaN并标记
        for col in self.numeric_cols:
            missing_count = df[col].isnull().sum()
            if missing_count == 0:
                continue
                
            missing_ratio = missing_count / len(df)
            
            if missing_ratio <= 0.1:  # 缺失率≤10%，用中位数填充
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"  数值列 '{col}': {missing_count} 个缺失值 -> 用中位数 {median_val:.2f} 填充")
            elif missing_ratio <= 0.3:  # 缺失率≤30%，用-999标记（保留缺失信息）
                df[col] = df[col].fillna(-999)
                print(f"  数值列 '{col}': {missing_count} 个缺失值 -> 用 -999 标记")
            else:  # 缺失率>30%，保留NaN并创建缺失标记列
                df[f"{col}_missing"] = df[col].isnull().astype(int)
                print(f"  数值列 '{col}': {missing_count} 个缺失值 -> 保留NaN，创建缺失标记列 '{col}_missing'")
        
        # 分类列：如果缺失率较低，用众数填充；否则保留NaN并标记
        for col in self.categorical_cols:
            missing_count = df[col].isnull().sum()
            if missing_count == 0:
                continue
                
            missing_ratio = missing_count / len(df)
            
            if missing_ratio <= 0.1:  # 缺失率≤10%，用众数填充
                mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col] = df[col].fillna(mode_val)
                print(f"  分类列 '{col}': {missing_count} 个缺失值 -> 用众数 '{mode_val}' 填充")
            elif missing_ratio <= 0.3:  # 缺失率≤30%，用'Unknown'标记
                df[col] = df[col].fillna('Unknown')
                print(f"  分类列 '{col}': {missing_count} 个缺失值 -> 用 'Unknown' 标记")
            else:  # 缺失率>30%，保留NaN并创建缺失标记列
                df[f"{col}_missing"] = df[col].isnull().astype(int)
                print(f"  分类列 '{col}': {missing_count} 个缺失值 -> 保留NaN，创建缺失标记列 '{col}_missing'")
        
        # 更新数值列和分类列列表（可能新增了标记列）
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        final_shape = df.shape
        print(f"最终数据形状: {initial_shape} -> {final_shape}")
        
        return df
    
    def identify_key_columns(self, df):
        '''识别关键列并映射到预设列名'''
        print("\n识别关键列...")
        
        # 根据实际数据定义可能的列名映射
        column_mappings = {
            'concrete_type': ['水泥类型', 'concrete_type', 'concrete', 'type', '混凝土类型', '类型'],
            'neutron_fluence': ['快中子辐照注量 n/cm2', 'neutron_fluence', 'fluence', 'neutron', '中子注量', '注量', '辐射量'],
            'temperature': ['辐照温度℃', '辐照温度', 'temperature', 'temp', '温度', '温'],
            'age_days': ['age_days', 'age', 'days', '龄期', '天数', '时间'],
            'compressive_strength': ['相对抗压强度', '抗压强度Mpa', 'compressive_strength', 'strength', 'compressive', '抗压强度', '强度'],
            'elastic_modulus': ['elastic_modulus', 'modulus', '弹性模量', '模量']
        }
        
        # 创建列映射
        self.column_mapping = {}
        for standard_col, possible_names in column_mappings.items():
            for name in possible_names:
                if name in df.columns:
                    self.column_mapping[standard_col] = name
                    print(f"  '{standard_col}' -> '{name}'")
                    break
            else:
                print(f"  未找到 '{standard_col}' 对应的列")
        
        print(f"列映射: {self.column_mapping}")
        return self.column_mapping
    
    def encode_categorical(self, df):
        '''编码分类变量'''
        print("\n编码分类变量...")
        
        if 'concrete_type' in self.column_mapping:
            actual_col = self.column_mapping['concrete_type']
            if actual_col in df.columns:
                # 混凝土类型编码 - 根据实际数据中的唯一值进行编码
                unique_values = df[actual_col].unique()
                type_mapping = {val: idx for idx, val in enumerate(unique_values)}
                df['concrete_type_encoded'] = df[actual_col].map(type_mapping)
                print(f"混凝土类型编码: {type_mapping}")
                print(f"编码列 '{actual_col}' -> 'concrete_type_encoded'")
                
                # 保存映射关系
                self.type_mapping = type_mapping
        
        return df
    
    def scale_features(self, df, feature_cols):
        '''特征标准化'''
        print(f"\n标准化特征: {feature_cols}")
        
        # 过滤实际存在的列
        available_features = [col for col in feature_cols if col in df.columns]
        if not available_features:
            print(f"警告: 没有找到特征列 {feature_cols}")
            return df, None
        
        # 确保所有特征列都是数值类型
        for col in available_features:
            if not pd.api.types.is_numeric_dtype(df[col]):
                print(f"警告: 列 '{col}' 不是数值类型，跳过标准化")
                continue
        
        # 只保留数值列
        numeric_features = [col for col in available_features if pd.api.types.is_numeric_dtype(df[col])]
        if not numeric_features:
            print("警告: 没有找到数值类型的特征列")
            return df, None
        
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[numeric_features])
        
        # 创建新的DataFrame
        scaled_df = pd.DataFrame(scaled_features, 
                                columns=[f"{col}_scaled" for col in numeric_features],
                                index=df.index)
        
        # 保存scaler
        self.scalers['feature'] = scaler
        
        return pd.concat([df, scaled_df], axis=1), scaler
    
    def scale_target(self, df, target_col):
        '''目标变量标准化'''
        if target_col not in df.columns:
            print(f"警告: 目标列 '{target_col}' 不存在，跳过标准化")
            return df, None
            
        print(f"\n标准化目标变量: {target_col}")
        
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_target = scaler.fit_transform(df[[target_col]])
        
        df[f"{target_col}_scaled"] = scaled_target
        self.scalers['target'] = scaler
        
        return df, scaler
    
    def prepare_diffusion_data(self, df):
        '''准备扩散模型训练数据'''
        print("\n准备扩散模型训练数据...")
        
        # 基于实际映射构建扩散模型列
        diffusion_cols = []
        for standard_col in ['concrete_type_encoded', 'neutron_fluence', 'temperature', 'compressive_strength']:
            if standard_col == 'concrete_type_encoded':
                # 特殊处理编码列
                if 'concrete_type_encoded' in df.columns:
                    diffusion_cols.append('concrete_type_encoded')
            else:
                # 映射到实际列名
                if standard_col in self.column_mapping:
                    actual_col = self.column_mapping[standard_col]
                    if actual_col + '_scaled' in df.columns:
                        diffusion_cols.append(actual_col + '_scaled')
                    elif actual_col in df.columns and pd.api.types.is_numeric_dtype(df[actual_col]):
                        # 如果没有缩放列，使用原始列（如果是数值类型）
                        diffusion_cols.append(actual_col)
        
        print(f"扩散模型列: {diffusion_cols}")
        
        if not diffusion_cols:
            print("警告: 没有找到适合扩散模型的列，将使用所有数值列")
            # 使用所有数值列
            diffusion_cols = [col for col in self.numeric_cols if col != 'source_sheet']
            if not diffusion_cols:
                # 如果没有数值列，使用所有可用列
                diffusion_cols = [col for col in df.columns if col not in ['source_sheet', '数据ID', '标题', 'DOI', '数据摘要', '关键词', '来源', '引用']]
        
        # 确保列存在
        available_cols = [col for col in diffusion_cols if col in df.columns]
        if not available_cols:
            raise ValueError(f"没有找到可用的列用于扩散模型: {diffusion_cols}")
        
        X_diffusion = df[available_cols].values
        print(f"扩散模型输入形状: {X_diffusion.shape}")
        print(f"使用的列: {available_cols}")
        
        return X_diffusion
    
    def split_data(self, df, test_size=0.2, random_state=42):
        '''划分训练测试集'''
        print(f"\n划分数据集 (测试集比例: {test_size})...")
        
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state
        )
        
        print(f"训练集: {train_df.shape}")
        print(f"测试集: {test_df.shape}")
        
        return train_df, test_df
    
    def save_processed_data(self, df, output_path):
        '''保存处理后的数据'''
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"创建输出目录: {output_dir}")
        
        df.to_csv(output_path, index=False)
        print(f"\n处理后的数据已保存至: {output_path}")
        
        # 保存元数据
        metadata = {
            'numeric_cols': self.numeric_cols,
            'categorical_cols': self.categorical_cols,
            'column_mapping': getattr(self, 'column_mapping', {}),
            'scalers': self.scalers
        }
        
        import pickle
        with open(output_path.replace('.csv', '_metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        return metadata

# 配置类
class Config:
    def __init__(self):
        self.data_path = "data/2019YFB1900903-004-001-中子辐照和高温环境下混凝土材料性能数据.xlsx"  # 修改为Excel文件
        self.output_path = "data/processed/concrete_processed.csv"
        self.test_size = 0.2
        self.random_state = 42

if __name__ == "__main__":
    # 示例用法
    config = Config()
    preprocessor = ConcreteDataPreprocessor(config)
    
    # 完整流程
    df = preprocessor.load_data(config.data_path)
    
    # 清理数值列中的非数值字符
    df = preprocessor.clean_numeric_columns(df)
    
    # 识别关键列
    column_mapping = preprocessor.identify_key_columns(df)
    
    df = preprocessor.handle_missing_values(df)
    df = preprocessor.encode_categorical(df)
    
    # 确定用于标准化的特征列
    feature_cols = []
    for col in ['neutron_fluence', 'temperature', 'compressive_strength']:
        if col in preprocessor.column_mapping:
            actual_col = preprocessor.column_mapping[col]
            if actual_col in df.columns:
                feature_cols.append(actual_col)
    
    if feature_cols:
        df, _ = preprocessor.scale_features(df, feature_cols)
    
    # 标准化目标变量 - 使用相对抗压强度作为目标
    target_col = ''
    if 'compressive_strength' in preprocessor.column_mapping:
        target_col = preprocessor.column_mapping['compressive_strength']
    elif '相对抗压强度' in df.columns:
        target_col = '相对抗压强度'
    
    if target_col and target_col in df.columns:
        df, _ = preprocessor.scale_target(df, target_col)
    
    # 划分数据
    train_df, test_df = preprocessor.split_data(df, config.test_size, config.random_state)
    
    # 保存数据
    metadata = preprocessor.save_processed_data(df, config.output_path)
    
    print("\n数据预处理完成!")