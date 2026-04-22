import numpy as np
import pandas as pd
from scipy import special

class PhysicsFeatureEngineer:
    '''
    基于辐照损伤物理机理的特征工程
    为混凝土辐照数据构建物理引导特征
    '''
    
    def __init__(self):
        # 混凝土材料基础物理参数（示例值，需根据实际调整）
        self.material_properties = {
            'ordinary': {
                'activation_energy': 40.0,  # kJ/mol, 典型值
                'thermal_diffusivity': 1.0e-6,  # m²/s
                'density': 2400,  # kg/m³
                'specific_heat': 900,  # J/(kg·K)
            },
            'heavy': {
                'activation_energy': 45.0,
                'thermal_diffusivity': 0.8e-6,
                'density': 3500,
                'specific_heat': 850,
            }
        }
        
    def create_irradiation_damage_features(self, df):
        '''
        创建辐照损伤相关物理特征
        基于中子辐照损伤理论
        '''
        print("="*50)
        print("创建辐照损伤物理特征")
        print("="*50)
        
        features = pd.DataFrame(index=df.index)
        
        # 识别实际列名
        concrete_type_col = None
        neutron_fluence_col = None
        temperature_col = None
        age_col = None
        strength_col = None
        elastic_modulus_col = None

        # 查找混凝土类型列
        possible_concrete_cols = ['水泥类型', 'concrete_type', 'concrete', 'type', '混凝土类型', '类型']
        for col in possible_concrete_cols:
            if col in df.columns:
                concrete_type_col = col
                break

        # 查找中子注量列
        possible_fluence_cols = ['快中子辐照注量 n/cm2', 'neutron_fluence', 'fluence', 'neutron', '中子注量', '注量', '辐射量']
        for col in possible_fluence_cols:
            if col in df.columns:
                neutron_fluence_col = col
                break

        # 查找温度列
        possible_temp_cols = ['辐照温度℃', '辐照温度', 'temperature', 'temp', '温度', '温']
        for col in possible_temp_cols:
            if col in df.columns:
                temperature_col = col
                break

        # 查找年龄列（如果存在）
        possible_age_cols = ['age_days', 'age', 'days', '龄期', '天数', '时间']
        for col in possible_age_cols:
            if col in df.columns:
                age_col = col
                break

        # 查找强度列
        possible_strength_cols = ['相对抗压强度', '抗压强度Mpa', 'compressive_strength', 'strength', 'compressive', '抗压强度', '强度']
        for col in possible_strength_cols:
            if col in df.columns:
                strength_col = col
                break

        # 查找弹性模量列
        possible_elastic_cols = ['elastic_modulus', 'modulus', '弹性模量', '模量']
        for col in possible_elastic_cols:
            if col in df.columns:
                elastic_modulus_col = col
                break

        print(f"识别的列名映射:")
        print(f"  混凝土类型: {concrete_type_col}")
        print(f"  中子注量: {neutron_fluence_col}")
        print(f"  温度: {temperature_col}")
        print(f"  年龄: {age_col}")
        print(f"  强度: {strength_col}")
        print(f"  弹性模量: {elastic_modulus_col}")

        # 1. 剂量率相关特征
        if neutron_fluence_col and age_col:
            # 假设年龄代表辐照时间（简化）
            df[age_col] = pd.to_numeric(df[age_col], errors='coerce')
            df[neutron_fluence_col] = pd.to_numeric(df[neutron_fluence_col], errors='coerce')
            
            # 对数剂量（辐照剂量通常跨度大）
            features['log_fluence'] = np.log10(df[neutron_fluence_col] + 1)
            
            # DPA等效（位移损伤）的简化估计
            dpa_conversion = 1e-19  # 简化转换因子
            features['dpa_equivalent'] = df[neutron_fluence_col] * dpa_conversion
            features['log_dpa'] = np.log10(features['dpa_equivalent'] + 1e-10)

        # 2. 温度-剂量协同效应特征
        if temperature_col and neutron_fluence_col:
            T = df[temperature_col] + 273.15  # 转换为开尔文
            
            # Arrhenius类型交互项：反映热激活过程
            features['arrhenius_term'] = np.exp(-self.material_properties['ordinary']['activation_energy'] * 1000 / (8.314 * T))
            
            # 温度-剂量乘积项（协同效应）
            features['temp_fluence_product'] = df[temperature_col] * df[neutron_fluence_col]
            
            # 温度归一化剂量
            features['fluence_temp_normalized'] = df[neutron_fluence_col] / (T + 1e-10)

        # 3. 时间相关特征（老化效应）
        if age_col:
            # 对数时间（老化过程通常非线性）
            features['log_age'] = np.log10(df[age_col] + 1)

        # 4. 综合损伤参数（基于经验公式）
        if neutron_fluence_col and temperature_col and age_col:
            # 综合损伤指数（简化）
            if 'log_dpa' in features.columns and 'arrhenius_term' in features.columns and 'log_age' in features.columns:
                features['damage_index'] = (
                    features['log_dpa'] * 
                    features['arrhenius_term'] * 
                    features['log_age']
                )
            elif 'log_dpa' in features.columns and 'arrhenius_term' in features.columns:
                features['damage_index'] = features['log_dpa'] * features['arrhenius_term']
            elif 'log_dpa' in features.columns:
                features['damage_index'] = features['log_dpa']
            else:
                # 如果没有log_dpa列，使用对数剂量
                features['damage_index'] = np.log10(df[neutron_fluence_col] + 1) * features['arrhenius_term'] if 'arrhenius_term' in features.columns else np.log10(df[neutron_fluence_col] + 1)

        # 5. 优化的物理特征 - 基于损伤理论的增强特征
        if neutron_fluence_col and temperature_col:
            # 基于辐照损伤动力学的特征
            fluence = df[neutron_fluence_col].values
            temp = df[temperature_col].values + 273.15  # 转换为开尔文
            
            # 热激活损伤模型：考虑温度对损伤恢复的影响
            features['thermal_damage_factor'] = np.exp(-5000/temp) * fluence  # 简化模型
            
            # 温度补偿剂量（阿伦尼乌斯型）
            features['temperature_compensated_dose'] = fluence * np.exp(0.1 * temp/300)  # 简化模型
            
            # 损伤饱和模型（考虑损伤饱和效应）
            features['damage_saturation'] = 1 - np.exp(-fluence * 1e-18)  # 假设饱和剂量为1e18 n/cm²

        # 6. 物理约束特征 - 用于指导扩散模型
        if neutron_fluence_col and temperature_col:
            # 基于物理的约束特征
            if 'log_fluence' in features.columns and 'arrhenius_term' in features.columns:
                features['physics_constraint_1'] = features['log_fluence'] * features['arrhenius_term']
            elif 'arrhenius_term' in features.columns:
                features['physics_constraint_1'] = np.log10(df[neutron_fluence_col] + 1) * features['arrhenius_term']
            else:
                features['physics_constraint_1'] = np.log10(df[neutron_fluence_col] + 1) * np.exp(-self.material_properties['ordinary']['activation_energy'] * 1000 / (8.314 * (df[temperature_col] + 273.15)))
            
            if 'log_age' in features.columns:
                features['physics_constraint_2'] = features['temp_fluence_product'] / (features['log_age'] + 1) if 'temp_fluence_product' in features.columns else df[temperature_col] * df[neutron_fluence_col] / (features['log_age'] + 1)
            else:
                features['physics_constraint_2'] = features['temp_fluence_product'] if 'temp_fluence_product' in features.columns else df[temperature_col] * df[neutron_fluence_col]
            
            if 'log_dpa' in features.columns:
                features['physics_constraint_3'] = features['log_dpa'] * df[temperature_col]
            else:
                features['physics_constraint_3'] = np.log10(df[neutron_fluence_col] * 1e-19 + 1e-10) * df[temperature_col]

        print(f"创建了 {len(features.columns)} 个物理引导特征:")
        for i, col in enumerate(features.columns, 1):
            print(f"  {i:2d}. {col}")
        
        # 合并到原始数据
        df_enhanced = pd.concat([df, features], axis=1)
        
        return df_enhanced, features.columns.tolist()
    
    def create_microstructure_features(self, df):
        '''
        创建基于微观结构演化的特征
        模拟混凝土辐照损伤的微观机理
        '''
        print("\n创建微观结构演化特征...")
        
        micro_features = pd.DataFrame(index=df.index)

        # 查找相关列
        neutron_fluence_col = None
        temperature_col = None
        age_col = None

        # 查找中子注量列
        possible_fluence_cols = ['快中子辐照注量 n/cm2', 'neutron_fluence', 'fluence', 'neutron', '中子注量', '注量', '辐射量']
        for col in possible_fluence_cols:
            if col in df.columns:
                neutron_fluence_col = col
                break

        # 查找温度列
        possible_temp_cols = ['辐照温度℃', '辐照温度', 'temperature', 'temp', '温度', '温']
        for col in possible_temp_cols:
            if col in df.columns:
                temperature_col = col
                break

        # 查找年龄列（如果存在）
        possible_age_cols = ['age_days', 'age', 'days', '龄期', '天数', '时间']
        for col in possible_age_cols:
            if col in df.columns:
                age_col = col
                break

        # 简化微观结构参数（基于分子动力学模拟的启发）
        if neutron_fluence_col:
            # 缺陷密度估计（随剂量增加而饱和）
            fluence = df[neutron_fluence_col].values
            defect_saturation = 1e18  # 饱和剂量（n/cm²）
            micro_features['defect_density'] = 1 - np.exp(-fluence / defect_saturation)
            
            # 非晶化程度估计
            amorphization_factor = 1e-17  # 非晶化系数
            micro_features['amorphous_fraction'] = 1 - np.exp(-fluence * amorphization_factor)
            
            # 微裂纹密度（与温度、剂量相关）
            if temperature_col:
                T = df[temperature_col].values
                # 温度促进微裂纹扩展
                crack_growth_factor = np.exp(-10000 / (T + 273.15))
                micro_features['crack_density'] = micro_features['defect_density'] * crack_growth_factor

        print(f"创建了 {len(micro_features.columns)} 个微观结构特征")
        print(f"微观结构特征: {list(micro_features.columns)}")
        
        return micro_features

    def filter_features_by_quality(self, df, features, target_col, correlation_threshold=0.05, 
                                  physical_intuition_required=True):
        '''
        根据质量和物理直觉筛选特征
        '''
        print(f"\n筛选高质量物理特征 (相关性阈值: {correlation_threshold})")
        
        if target_col not in df.columns:
            print(f"警告: 目标列 '{target_col}' 不存在，跳过特征筛选")
            return features

        selected_features = []
        feature_quality_scores = {}

        for feature in features:
            if feature not in df.columns:
                continue
                
            # 计算与目标变量的相关性
            correlation = df[feature].corr(df[target_col])
            abs_correlation = abs(correlation)
            
            # 检查是否符合物理直觉
            physical_intuition = True
            if physical_intuition_required:
                # 检查特征名称是否暗示与损伤相关的物理直觉（应与强度负相关）
                if any(indicator in feature for indicator in ['fluence', 'dpa', 'damage', 'swelling', 'defect', 'crack', 'damage_index', 'thermal_damage_factor', 'damage_saturation']):
                    # 这些特征通常应该与强度负相关（损伤越大，强度越小）
                    physical_intuition = correlation < 0
                elif any(indicator in feature for indicator in ['strength', 'modulus']):
                    # 强度或模量相关特征通常应该与目标强度正相关
                    physical_intuition = correlation > 0

            # 特征质量评分
            quality_score = abs_correlation
            if physical_intuition:
                quality_score *= 1.2  # 物理直觉符合的特征给予额外权重

            feature_quality_scores[feature] = {
                'correlation': correlation,
                'abs_correlation': abs_correlation,
                'physical_intuition': physical_intuition,
                'quality_score': quality_score
            }

            # 筛选条件
            if abs_correlation >= correlation_threshold and physical_intuition:
                selected_features.append(feature)

        print(f"原始特征数: {len(features)}")
        print(f"筛选后特征数: {len(selected_features)}")
        print(f"筛选出的特征: {selected_features}")

        # 打印详细的质量评分
        print(f"\n特征质量评分 (按质量评分排序):")
        print("-" * 100)
        print(f"{'特征':<20} {'相关性':<10} {'绝对值':<10} {'物理直觉':<12} {'质量评分':<10}")
        print("-" * 100)
        
        sorted_features = sorted(feature_quality_scores.items(), 
                                key=lambda x: x[1]['quality_score'], reverse=True)
        
        for feature, scores in sorted_features:
            corr = scores['correlation']
            abs_corr = scores['abs_correlation']
            phys_int = "符合" if scores['physical_intuition'] else "不符合"
            q_score = scores['quality_score']
            included = "✓" if feature in selected_features else "✗"
            print(f"{feature:<20} {corr:<10.3f} {abs_corr:<10.3f} {phys_int:<12} {q_score:<10.3f} {included}")

        return selected_features, feature_quality_scores
    
    def calculate_physics_constraints(self, df, target_col=None):
        '''
        计算物理约束条件，用于引导扩散模型
        '''
        print("\n计算物理约束条件...")
        
        constraints = {}
        
        # 自动检测目标列
        if target_col is None:
            possible_target_cols = ['相对抗压强度', '抗压强度Mpa', 'compressive_strength', 'strength', 'compressive', '抗压强度', '强度']
            for col in possible_target_cols:
                if col in df.columns:
                    target_col = col
                    break

        # 检查是否找到目标列
        if target_col is None:
            print("警告: 未找到目标列，跳过物理约束计算")
            return constraints

        print(f"使用目标列: {target_col}")

        # 查找相关列
        neutron_fluence_col = None
        temperature_col = None

        # 查找中子注量列
        possible_fluence_cols = ['快中子辐照注量 n/cm2', 'neutron_fluence', 'fluence', 'neutron', '中子注量', '注量', '辐射量']
        for col in possible_fluence_cols:
            if col in df.columns:
                neutron_fluence_col = col
                break

        # 查找温度列
        possible_temp_cols = ['辐照温度℃', '辐照温度', 'temperature', 'temp', '温度', '温']
        for col in possible_temp_cols:
            if col in df.columns:
                temperature_col = col
                break

        # 1. 单调性约束
        # 在相同条件下，强度应随剂量增加而降低（一般规律）
        if neutron_fluence_col and target_col:
            # 按剂量分组计算平均强度变化趋势
            dose_bins = pd.cut(df[neutron_fluence_col], bins=10)
            mean_strength_by_dose = df.groupby(dose_bins)[target_col].mean()
            
            # 检查单调性
            if mean_strength_by_dose.is_monotonic_decreasing:
                constraints['dose_monotonic'] = True
                constraints['dose_trend'] = 'decreasing'
            else:
                constraints['dose_monotonic'] = False
                constraints['dose_trend'] = 'non-monotonic'
        
        # 2. 温度约束
        # 在相同剂量下，高温强度通常较低
        if temperature_col and target_col and neutron_fluence_col:
            # 选择相近剂量范围
            dose_median = df[neutron_fluence_col].median()
            dose_range = df[neutron_fluence_col].quantile(0.75) - df[neutron_fluence_col].quantile(0.25)
            
            mask = abs(df[neutron_fluence_col] - dose_median) < dose_range
            if mask.sum() > 10:  # 有足够样本
                temp_strength_corr = df.loc[mask, [temperature_col, target_col]].corr().iloc[0, 1]
                constraints['temperature_correlation'] = temp_strength_corr

        # 3. 材料类型约束
        possible_concrete_cols = ['水泥类型', 'concrete_type', 'concrete', 'type', '混凝土类型', '类型']
        concrete_type_col = None
        for col in possible_concrete_cols:
            if col in df.columns:
                concrete_type_col = col
                break

        if concrete_type_col:
            type_strength = df.groupby(concrete_type_col)[target_col].mean()
            constraints['type_strength_ratio'] = type_strength.to_dict()
        
        print("物理约束条件:")
        for key, value in constraints.items():
            print(f"  {key}: {value}")
        
        return constraints

    def validate_physics_features(self, df, features, target_col=None):
        '''
        验证物理特征的有效性
        '''
        print("\n验证物理特征有效性...")
        print(f"物理引导特征列表: {list(features)}")
        
        validation_results = {}

        # 自动检测目标列
        if target_col is None:
            possible_target_cols = ['相对抗压强度', '抗压强度Mpa', 'compressive_strength', 'strength', 'compressive', '抗压强度', '强度']
            for col in possible_target_cols:
                if col in df.columns:
                    target_col = col
                    break

        if target_col is None:
            print("警告: 未找到目标列，跳过特征验证")
            return validation_results

        print(f"使用目标列: {target_col}")

        # 1. 相关性分析
        corr_threshold = 0.3  # 相关性阈值
        
        for feature in features:
            if feature in df.columns:
                correlation = df[[feature, target_col]].corr().iloc[0, 1]
                validation_results[feature] = {
                    'correlation': correlation,
                    'abs_correlation': abs(correlation),
                    'significant': abs(correlation) > corr_threshold,
                    'correlation_strength': self._get_correlation_strength(abs(correlation))
                }
            else:
                print(f"警告: 特征 '{feature}' 不存在于数据中")
                validation_results[feature] = {
                    'correlation': np.nan,
                    'abs_correlation': np.nan,
                    'significant': False,
                    'correlation_strength': 'N/A'
                }
        
        # 2. 物理一致性检查
        physically_consistent = []
        for feature in features:
            if feature in validation_results and feature in df.columns:
                # 检查是否符合物理直觉
                correlation = validation_results[feature]['correlation']
                # 检查特征名称是否暗示与损伤相关的物理直觉（应与强度负相关）
                if any(indicator in feature for indicator in ['fluence', 'dpa', 'damage', 'swelling', 'defect', 'crack', 'damage_index', 'thermal_damage_factor', 'damage_saturation']):
                    # 这些特征通常应该与强度负相关（损伤越大，强度越小）
                    if correlation < 0:
                        physically_consistent.append(feature)
                else:
                    # 其他特征只要不违反明显物理直觉即可
                    physically_consistent.append(feature)
        
        validation_results['physically_consistent_features'] = physically_consistent

        # 打印详细结果
        print(f"\n物理特征与目标变量 '{target_col}' 的相关性分析:")
        print("-" * 80)
        print(f"{'特征':<25} {'相关性':<10} {'绝对值':<10} {'显著性':<10} {'相关强度':<15}")
        print("-" * 80)
        for feature, results in validation_results.items():
            if feature != 'physically_consistent_features':  # 跳过汇总信息
                if feature in df.columns:
                    corr = results['correlation']
                    abs_corr = results['abs_correlation']
                    sig = "是" if results['significant'] else "否"
                    strength = results['correlation_strength']
                    print(f"{feature:<25} {corr:<10.3f} {abs_corr:<10.3f} {sig:<10} {strength:<15}")
                else:
                    print(f"{feature:<25} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<15}")
        
        print(f"\n物理一致性特征 ({len(physically_consistent)} 个):")
        for feat in physically_consistent:
            print(f"  - {feat}")

        return validation_results

    def _get_correlation_strength(self, abs_corr):
        '''
        根据绝对相关性值判断相关性强度
        '''
        if abs_corr >= 0.7:
            return "强相关"
        elif abs_corr >= 0.5:
            return "中等相关"
        elif abs_corr >= 0.3:
            return "弱相关"
        else:
            return "微弱相关"

    def analyze_feature_importance(self, df, features, target_col=None, method='correlation'):
        '''
        分析物理特征的重要性
        '''
        print(f"\n分析物理特征重要性 ({method})...")
        print(f"物理引导特征列表: {list(features)}")
        
        importance_results = {}

        # 自动检测目标列
        if target_col is None:
            possible_target_cols = ['相对抗压强度', '抗压强度Mpa', 'compressive_strength', 'strength', 'compressive', '抗压强度', '强度']
            for col in possible_target_cols:
                if col in df.columns:
                    target_col = col
                    break

        if target_col is None:
            print("警告: 未找到目标列，跳过特征重要性分析")
            return importance_results

        print(f"使用目标列: {target_col}")

        # 计算特征与目标的相关性
        for feature in features:
            if feature in df.columns:
                # 计算皮尔逊相关系数
                correlation = df[feature].corr(df[target_col])
                importance_results[feature] = {
                    'importance': abs(correlation),  # 使用绝对值作为重要性指标
                    'correlation': correlation,
                    'method': 'correlation'
                }
            else:
                print(f"警告: 特征 '{feature}' 不存在于数据中")
                importance_results[feature] = {
                    'importance': 0,
                    'correlation': 0,
                    'method': 'correlation'
                }

        # 按重要性排序
        sorted_features = sorted(importance_results.items(), key=lambda x: x[1]['importance'], reverse=True)

        print(f"\n物理特征重要性排序 ({method}):")
        print("-" * 80)
        print(f"{'特征':<25} {'重要性':<10} {'相关性':<10} {'方法':<10}")
        print("-" * 80)
        for feature, results in sorted_features:
            imp = results['importance']
            corr = results['correlation']
            method_name = results['method']
            print(f"{feature:<25} {imp:<10.3f} {corr:<10.3f} {method_name:<10}")

        return importance_results

    def evaluate_feature_quality(self, df, features, target_col=None):
        '''
        评估物理特征质量，包括相关性、显著性和物理一致性
        '''
        print("\n" + "="*60)
        print("评估物理特征质量（相关性、显著性、物理一致性）")
        print("="*60)

        quality_results = {}

        # 自动检测目标列
        if target_col is None:
            possible_target_cols = ['相对抗压强度', '抗压强度Mpa', 'compressive_strength', 'strength', 'compressive', '抗压强度', '强度']
            for col in possible_target_cols:
                if col in df.columns:
                    target_col = col
                    break

        if target_col is None:
            print("警告: 未找到目标列，跳过特征质量评估")
            return quality_results

        print(f"使用目标列: {target_col}")

        # 1. 计算相关性和显著性
        for feature in features:
            if feature in df.columns:
                correlation = df[feature].corr(df[target_col])
                abs_corr = abs(correlation)
                # 显著性判断（基于相关性绝对值阈值）
                significant = abs_corr > 0.1  # 相关性绝对值大于0.1认为显著
                
                quality_results[feature] = {
                    'correlation': correlation,
                    'abs_correlation': abs_corr,
                    'significant': significant,
                    'quality_score': abs_corr,  # 简单的质量评分
                    'correlation_strength': self._get_correlation_strength(abs_corr)
                }
            else:
                print(f"警告: 特征 '{feature}' 不存在于数据中")
                quality_results[feature] = {
                    'correlation': np.nan,
                    'abs_correlation': np.nan,
                    'significant': False,
                    'quality_score': 0,
                    'correlation_strength': 'N/A'
                }

        # 2. 评估物理一致性
        for feature in features:
            if feature in quality_results and feature in df.columns:
                # 检查是否符合物理直觉
                correlation = quality_results[feature]['correlation']
                # 检查特征名称是否暗示与损伤相关的物理直觉（应与强度负相关）
                if any(indicator in feature for indicator in ['fluence', 'dpa', 'damage', 'swelling', 'defect', 'crack', 'damage_index', 'thermal_damage_factor', 'damage_saturation']):
                    # 这些特征通常应该与强度负相关（损伤越大，强度越小）
                    quality_results[feature]['physical_intuition'] = correlation < 0
                elif any(indicator in feature for indicator in ['strength', 'modulus']):
                    # 强度或模量相关特征通常应该与目标强度正相关
                    quality_results[feature]['physical_intuition'] = correlation > 0
                else:
                    # 其他特征只要不违反明显物理直觉即可
                    quality_results[feature]['physical_intuition'] = True

        # 3. 按质量评分排序
        sorted_features = sorted(quality_results.items(), key=lambda x: x[1]['quality_score'], reverse=True)

        print(f"\n物理特征质量评估结果:")
        print("-" * 110)
        print(f"{'特征':<20} {'相关性':<10} {'绝对值':<10} {'显著性':<10} {'相关强度':<12} {'物理直觉':<12} {'质量评分':<10}")
        print("-" * 110)
        for feature, results in sorted_features:
            corr = results['correlation']
            abs_corr = results['abs_correlation']
            sig = "是" if results['significant'] else "否"
            strength = results['correlation_strength']
            phys_int = "符合" if results.get('physical_intuition', True) else "不符合"
            score = results['quality_score']
            print(f"{feature:<20} {corr:<10.3f} {abs_corr:<10.3f} {sig:<10} {strength:<12} {phys_int:<12} {score:<10.3f}")

        # 4. 统计信息
        significant_features = [f for f, r in quality_results.items() if r['significant']]
        physically_consistent_features = [f for f, r in quality_results.items() if r.get('physical_intuition', True)]
        highly_correlated_features = [f for f, r in quality_results.items() if r['abs_correlation'] > 0.3]
        medium_correlated_features = [f for f, r in quality_results.items() if 0.1 < r['abs_correlation'] <= 0.3]

        print(f"\n统计信息:")
        print(f"  总物理特征数: {len(features)}")
        print(f"  显著特征数 (>0.1): {len(significant_features)} ({len(significant_features)/len(features)*100:.1f}%)")
        print(f"  中等以上相关特征数 (>0.3): {len(highly_correlated_features)} ({len(highly_correlated_features)/len(features)*100:.1f}%)")
        print(f"  低相关特征数 (0.1-0.3): {len(medium_correlated_features)} ({len(medium_correlated_features)/len(features)*100:.1f}%)")
        print(f"  符合物理直觉特征数: {len(physically_consistent_features)} ({len(physically_consistent_features)/len(features)*100:.1f}%)")

        print(f"\n显著特征列表: {significant_features}")
        print(f"中等以上相关特征列表: {highly_correlated_features}")
        print(f"物理一致性特征列表: {physically_consistent_features}")

        return quality_results

# 使用实际数据进行处理
if __name__ == "__main__":
    # 读取实际数据
    try:
        df = pd.read_csv("data/processed/concrete_processed.csv")
        print(f"成功加载数据，形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
    except FileNotFoundError:
        print("错误: 找不到处理后的数据文件 'data/processed/concrete_processed.csv'")
        print("请先运行数据预处理脚本生成处理后的数据文件")
        exit(1)
    
    engineer = PhysicsFeatureEngineer()
    
    # 创建物理特征 - 这会返回增强后的数据和物理特征列表
    enhanced_data, physics_features = engineer.create_irradiation_damage_features(df)
    
    # 创建微观特征 - 也需要将它们添加到增强数据中
    micro_features_df = engineer.create_microstructure_features(enhanced_data)
    enhanced_data = pd.concat([enhanced_data, micro_features_df], axis=1)
    all_micro_features = micro_features_df.columns.tolist()
    
    # 合并所有物理特征
    all_features = all_micro_features + physics_features
    
    # 筛选高质量物理特征
    target_col = '相对抗压强度'  # 假设这是目标列
    if target_col not in enhanced_data.columns:
        # 如果相对抗压强度不存在，尝试其他可能的目标列名
        possible_targets = ['抗压强度Mpa', 'compressive_strength', 'strength', 'compressive', '抗压强度', '强度']
        for col in possible_targets:
            if col in enhanced_data.columns:
                target_col = col
                break
    
    if target_col in enhanced_data.columns:
        selected_features, quality_scores = engineer.filter_features_by_quality(
            enhanced_data, all_features, target_col, 
            correlation_threshold=0.05,  # 相关性阈值
            physical_intuition_required=True  # 要求符合物理直觉
        )
        
        # 只保留高质量特征
        print(f"\n保留高质量特征: {selected_features}")
        filtered_features_df = enhanced_data[selected_features]
        
        # 计算物理约束
        constraints = engineer.calculate_physics_constraints(enhanced_data)
        
        # 验证物理特征（使用筛选后的特征）
        validation = engineer.validate_physics_features(enhanced_data, selected_features)
        
        # 分析特征重要性
        importance = engineer.analyze_feature_importance(enhanced_data, selected_features)
        
        # 评估特征质量
        quality = engineer.evaluate_feature_quality(enhanced_data, selected_features)
        
        # 更新增强数据，只保留高质量特征
        enhanced_data = pd.concat([df, filtered_features_df], axis=1)
        all_features = selected_features
    else:
        print(f"警告: 未找到目标列 '{target_col}'，跳过特征筛选")
        constraints = engineer.calculate_physics_constraints(enhanced_data)
        validation = engineer.validate_physics_features(enhanced_data, all_features)
        importance = engineer.analyze_feature_importance(enhanced_data, all_features)
        quality = engineer.evaluate_feature_quality(enhanced_data, all_features)
    
    print(f"\n原始数据形状: {df.shape}")
    print(f"增强后数据形状: {enhanced_data.shape}")
    print(f"总物理特征数: {len(all_features)}")
    print(f"所有物理特征: {all_features}")
    
    # 保存增强后的数据
    enhanced_data.to_csv("data/processed/concrete_processed_with_physics_features.csv", index=False)
    print(f"\n增强后的数据已保存至: data/processed/concrete_processed_with_physics_features.csv")