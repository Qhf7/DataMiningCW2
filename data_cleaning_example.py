#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
情绪数据集清洗示例
本脚本演示了使用Pandas进行数据清洗的完整流程
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# -----------------------------------------------------------------------------
# 第1步：数据导入和初步检查
# -----------------------------------------------------------------------------
def load_and_explore_data(file_path):
    """
    加载数据并进行初步探索
    """
    print(f"\n{'='*20} 加载并探索数据 {'='*20}")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在！")
        return None
    
    # 尝试加载不同格式的文件
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)
    else:
        print(f"不支持的文件格式: {file_path}")
        return None
    
    # 打印基本信息
    print(f"\n数据集大小: {df.shape[0]} 行 x {df.shape[1]} 列")
    print("\n数据类型和非空值统计:")
    print(df.info())
    
    # 查看前几行数据
    print("\n数据预览 (前5行):")
    print(df.head())
    
    # 数值列统计摘要
    print("\n数值列统计摘要:")
    print(df.describe())
    
    # 检查缺失值
    missing_values = df.isnull().sum()
    print("\n每列缺失值数量:")
    print(missing_values)
    print(f"\n总缺失值数量: {missing_values.sum()}")
    
    # 检查重复行
    duplicates = df.duplicated().sum()
    print(f"\n重复行数量: {duplicates}")
    
    return df

# -----------------------------------------------------------------------------
# 第2步：数据清洗 - 处理缺失值
# -----------------------------------------------------------------------------
def handle_missing_values(df):
    """
    处理数据集中的缺失值
    """
    print(f"\n{'='*20} 处理缺失值 {'='*20}")
    
    if df is None:
        return None
    
    # 原始数据集的缺失值统计
    print(f"\n清洗前的形状: {df.shape}")
    print(f"清洗前的缺失值总数: {df.isnull().sum().sum()}")
    
    # 复制数据集，避免修改原始数据
    df_cleaned = df.copy()
    
    # 针对不同列采用不同的缺失值处理策略
    
    # 1. 对于标识列，如ID列，有缺失通常意味着数据不完整，可能需要删除
    id_cols = [col for col in df_cleaned.columns if 'id' in col.lower()]
    for col in id_cols:
        if col in df_cleaned.columns and df_cleaned[col].isnull().any():
            print(f"列 {col} 有缺失值，将删除这些行")
            df_cleaned = df_cleaned.dropna(subset=[col])
    
    # 2. 对于文本列，用适当的标记填充缺失值
    text_cols = ['title', 'post', 'text', 'feelings']
    for col in text_cols:
        if col in df_cleaned.columns and df_cleaned[col].isnull().any():
            missing_count = df_cleaned[col].isnull().sum()
            print(f"列 {col} 有 {missing_count} 个缺失值，用'未知'填充")
            df_cleaned[col] = df_cleaned[col].fillna('未知')
    
    # 3. 对于数值列，用中位数或均值填充
    numeric_cols = df_cleaned.select_dtypes(include=['int64', 'float64']).columns.tolist()
    for col in numeric_cols:
        if df_cleaned[col].isnull().any():
            missing_count = df_cleaned[col].isnull().sum()
            # 使用中位数填充（对异常值不敏感）
            median_value = df_cleaned[col].median()
            print(f"列 {col} 有 {missing_count} 个缺失值，用中位数 {median_value} 填充")
            df_cleaned[col] = df_cleaned[col].fillna(median_value)
    
    # 4. 对于日期列，用前一个值或后一个值填充
    date_cols = [col for col in df_cleaned.columns if 'date' in col.lower() or 'time' in col.lower()]
    for col in date_cols:
        if col in df_cleaned.columns and df_cleaned[col].isnull().any():
            missing_count = df_cleaned[col].isnull().sum()
            print(f"列 {col} 有 {missing_count} 个缺失值，用前向填充方法处理")
            df_cleaned[col] = df_cleaned[col].fillna(method='ffill')
            # 如果第一行就是缺失值，用后向填充
            df_cleaned[col] = df_cleaned[col].fillna(method='bfill')
    
    # 清洗后的缺失值统计
    print(f"\n清洗后的形状: {df_cleaned.shape}")
    print(f"清洗后的缺失值总数: {df_cleaned.isnull().sum().sum()}")
    
    return df_cleaned

# -----------------------------------------------------------------------------
# 第3步：数据清洗 - 处理数据类型
# -----------------------------------------------------------------------------
def convert_data_types(df):
    """
    转换和修正数据类型
    """
    print(f"\n{'='*20} 转换数据类型 {'='*20}")
    
    if df is None:
        return None
    
    # 复制数据集
    df_typed = df.copy()
    
    # 1. 处理日期列
    date_cols = [col for col in df_typed.columns if 'date' in col.lower() or 'time' in col.lower()]
    for col in date_cols:
        if col in df_typed.columns:
            try:
                # 尝试转换为日期时间类型
                df_typed[col] = pd.to_datetime(df_typed[col])
                print(f"已将 {col} 转换为日期时间类型")
                
                # 提取日期特征
                df_typed[f'{col}_year'] = df_typed[col].dt.year
                df_typed[f'{col}_month'] = df_typed[col].dt.month
                df_typed[f'{col}_day'] = df_typed[col].dt.day
                df_typed[f'{col}_dayofweek'] = df_typed[col].dt.dayofweek
                print(f"已从 {col} 提取年、月、日和星期几特征")
                
            except Exception as e:
                print(f"无法将 {col} 转换为日期时间类型: {e}")
    
    # 2. 处理数值列
    # 例如，upvotes列应为整数
    if 'upvotes' in df_typed.columns:
        try:
            df_typed['upvotes'] = df_typed['upvotes'].astype(int)
            print(f"已将 upvotes 转换为整数类型")
        except Exception as e:
            print(f"无法将 upvotes 转换为整数: {e}")
    
    # 3. 处理文本列 - 确保所有文本列都是字符串类型
    text_cols = ['title', 'post', 'text', 'feelings']
    for col in text_cols:
        if col in df_typed.columns:
            df_typed[col] = df_typed[col].astype(str)
            print(f"已将 {col} 转换为字符串类型")
    
    # 4. 将类别型数据转换为category类型可以节省内存
    categorical_cols = ['id']  # 假设ID是分类变量
    for col in categorical_cols:
        if col in df_typed.columns:
            df_typed[col] = df_typed[col].astype('category')
            print(f"已将 {col} 转换为分类类型")
    
    return df_typed

# -----------------------------------------------------------------------------
# 第4步：数据清洗 - 处理异常值
# -----------------------------------------------------------------------------
def handle_outliers(df):
    """
    检测并处理异常值
    """
    print(f"\n{'='*20} 处理异常值 {'='*20}")
    
    if df is None:
        return None
    
    # 复制数据集
    df_no_outliers = df.copy()
    
    # 仅处理数值列的异常值
    numeric_cols = df_no_outliers.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    for col in numeric_cols:
        # 跳过日期派生的数值列
        if col.endswith(('_year', '_month', '_day', '_dayofweek')):
            continue
            
        # 计算IQR (四分位距)
        Q1 = df_no_outliers[col].quantile(0.25)
        Q3 = df_no_outliers[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # 定义异常值边界
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # 识别异常值
        outliers = df_no_outliers[(df_no_outliers[col] < lower_bound) | (df_no_outliers[col] > upper_bound)]
        
        if not outliers.empty:
            print(f"\n列 {col} 中发现 {len(outliers)} 个异常值")
            print(f"正常范围: {lower_bound} 到 {upper_bound}")
            
            # 展示一些异常值示例
            if len(outliers) > 0:
                print("异常值示例:")
                print(outliers[col].head())
            
            # 处理异常值 - 这里使用截断法（限制在边界内）
            df_no_outliers[col] = df_no_outliers[col].clip(lower=lower_bound, upper=upper_bound)
            print(f"已将 {col} 中的异常值截断至正常范围内")
            
            # 可视化异常值处理前后的分布
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            sns.boxplot(x=df[col])
            plt.title(f"{col} - 处理前的箱线图")
            
            plt.subplot(1, 2, 2)
            sns.boxplot(x=df_no_outliers[col])
            plt.title(f"{col} - 处理后的箱线图")
            
            plt.tight_layout()
            plt.savefig(f"outliers_{col}.png")
            plt.close()
            print(f"已保存异常值处理前后的箱线图至 outliers_{col}.png")
    
    return df_no_outliers

# -----------------------------------------------------------------------------
# 第5步：数据清洗 - 处理重复行
# -----------------------------------------------------------------------------
def remove_duplicates(df):
    """
    检测并删除重复行
    """
    print(f"\n{'='*20} 处理重复行 {'='*20}")
    
    if df is None:
        return None
    
    # 检查重复行
    duplicates = df.duplicated()
    duplicate_count = duplicates.sum()
    
    print(f"发现 {duplicate_count} 行完全重复的数据")
    
    if duplicate_count > 0:
        # 展示一些重复行示例
        print("\n重复行示例:")
        print(df[duplicates].head())
        
        # 删除完全重复的行
        df_unique = df.drop_duplicates()
        print(f"已删除 {duplicate_count} 行完全重复的数据")
        
        # 检查基于特定列的重复
        # 例如，我们可能认为具有相同text但ID不同的行可能是重复内容
        if 'text' in df.columns:
            text_duplicates = df.duplicated(subset=['text'], keep='first')
            text_duplicate_count = text_duplicates.sum()
            print(f"基于'text'列发现 {text_duplicate_count} 行可能重复的内容")
            
            if text_duplicate_count > 0 and text_duplicate_count < 10:
                print("\n基于'text'列的重复内容示例:")
                print(df[text_duplicates][['id', 'text']].head())
        
        return df_unique
    else:
        return df

# -----------------------------------------------------------------------------
# 第6步：数据清洗 - 文本数据处理
# -----------------------------------------------------------------------------
def clean_text_data(df):
    """
    清洗文本数据列
    """
    print(f"\n{'='*20} 清洗文本数据 {'='*20}")
    
    if df is None:
        return None
    
    # 复制数据集
    df_text_cleaned = df.copy()
    
    # 定义要处理的文本列
    text_cols = ['title', 'post', 'text']
    
    for col in text_cols:
        if col in df_text_cleaned.columns:
            # 记录清洗前的统计信息
            before_length = df_text_cleaned[col].str.len().mean()
            print(f"\n清洗前 {col} 列的平均长度: {before_length:.2f} 字符")
            
            # 去除前后空白字符
            df_text_cleaned[col] = df_text_cleaned[col].str.strip()
            
            # 替换多个空格为单个空格
            df_text_cleaned[col] = df_text_cleaned[col].str.replace(r'\s+', ' ', regex=True)
            
            # 替换常见的HTML实体
            df_text_cleaned[col] = df_text_cleaned[col].str.replace('&amp;', '&')
            df_text_cleaned[col] = df_text_cleaned[col].str.replace('&lt;', '<')
            df_text_cleaned[col] = df_text_cleaned[col].str.replace('&gt;', '>')
            df_text_cleaned[col] = df_text_cleaned[col].str.replace('&quot;', '"')
            
            # 可选：转换为小写
            # df_text_cleaned[col] = df_text_cleaned[col].str.lower()
            
            # 记录清洗后的统计信息
            after_length = df_text_cleaned[col].str.len().mean()
            print(f"清洗后 {col} 列的平均长度: {after_length:.2f} 字符")
            print(f"减少了 {before_length - after_length:.2f} 字符 ({(before_length - after_length) / before_length * 100:.2f}%)")
    
    # 处理情绪标签列
    if 'feelings' in df_text_cleaned.columns:
        print("\n处理情绪标签列...")
        
        # 规范化情绪标签，去除多余空格并转为小写
        df_text_cleaned['feelings'] = df_text_cleaned['feelings'].str.lower().str.strip()
        
        # 提取所有出现的情绪标签
        all_emotions = []
        for feelings in df_text_cleaned['feelings']:
            # 分割情绪标签（假设用逗号分隔）
            if isinstance(feelings, str):
                emotions = [emotion.strip() for emotion in feelings.split(',')]
                all_emotions.extend(emotions)
        
        # 统计情绪标签出现次数
        emotion_counts = pd.Series(all_emotions).value_counts()
        print("\n情绪标签出现次数:")
        print(emotion_counts.head(10))  # 显示前10个最常见的情绪
        
        # 创建情绪标签的独热编码
        print("\n创建情绪标签的独热编码...")
        
        # 获取前8种最常见的情绪
        top_emotions = emotion_counts.index[:8].tolist()
        
        # 为每种情绪创建一个新列
        for emotion in top_emotions:
            df_text_cleaned[f'emotion_{emotion}'] = df_text_cleaned['feelings'].str.contains(emotion, case=False, na=False).astype(int)
        
        print(f"已为以下情绪创建独热编码: {', '.join(top_emotions)}")
    
    return df_text_cleaned

# -----------------------------------------------------------------------------
# 第7步：数据标准化和归一化
# -----------------------------------------------------------------------------
def normalize_data(df):
    """
    标准化或归一化数值数据
    """
    print(f"\n{'='*20} 数据标准化和归一化 {'='*20}")
    
    if df is None:
        return None
    
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    
    # 复制数据集
    df_normalized = df.copy()
    
    # 选择要标准化的数值列
    numeric_cols = df_normalized.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # 排除已标准化的列、ID列和日期衍生列
    cols_to_normalize = [col for col in numeric_cols 
                        if not col.startswith(('id', 'emotion_')) 
                        and not col.endswith(('_year', '_month', '_day', '_dayofweek'))]
    
    if cols_to_normalize:
        print(f"\n将标准化以下数值列: {cols_to_normalize}")
        
        # 创建MinMaxScaler实例（归一化到0-1范围）
        min_max_scaler = MinMaxScaler()
        
        # 应用归一化
        df_temp = pd.DataFrame(min_max_scaler.fit_transform(df_normalized[cols_to_normalize]), 
                              columns=cols_to_normalize,
                              index=df_normalized.index)
        
        # 将归一化结果添加到原始数据框
        for col in cols_to_normalize:
            df_normalized[f'{col}_normalized'] = df_temp[col]
            print(f"已添加 {col}_normalized 列")
        
        # 创建StandardScaler实例（标准化到均值0、标准差1）
        std_scaler = StandardScaler()
        
        # 应用标准化
        df_temp = pd.DataFrame(std_scaler.fit_transform(df_normalized[cols_to_normalize]), 
                              columns=cols_to_normalize,
                              index=df_normalized.index)
        
        # 将标准化结果添加到原始数据框
        for col in cols_to_normalize:
            df_normalized[f'{col}_standardized'] = df_temp[col]
            print(f"已添加 {col}_standardized 列")
    else:
        print("没有适合标准化的数值列")
    
    return df_normalized

# -----------------------------------------------------------------------------
# 第8步：创建特征工程
# -----------------------------------------------------------------------------
def engineer_features(df):
    """
    创建新特征
    """
    print(f"\n{'='*20} 特征工程 {'='*20}")
    
    if df is None:
        return None
    
    # 复制数据集
    df_featured = df.copy()
    
    # 1. 文本长度特征
    text_cols = ['title', 'post', 'text']
    for col in text_cols:
        if col in df_featured.columns:
            df_featured[f'{col}_length'] = df_featured[col].str.len()
            print(f"已添加 {col}_length 特征，表示 {col} 列的文本长度")
    
    # 2. 情绪复杂度特征 - 一个帖子中提到的情绪种类数量
    if 'feelings' in df_featured.columns:
        df_featured['emotion_count'] = df_featured['feelings'].str.split(',').str.len()
        print("已添加 emotion_count 特征，表示每个帖子中情绪的数量")
    
    # 3. 时间特征（如果有日期列）
    date_cols = [col for col in df_featured.columns if pd.api.types.is_datetime64_any_dtype(df_featured[col])]
    for col in date_cols:
        # 是否为周末
        df_featured[f'{col}_is_weekend'] = df_featured[col].dt.dayofweek >= 5
        print(f"已添加 {col}_is_weekend 特征，表示 {col} 是否为周末")
        
        # 一天中的小时
        if df_featured[col].dt.hour.nunique() > 1:  # 确保有小时信息
            df_featured[f'{col}_hour'] = df_featured[col].dt.hour
            print(f"已添加 {col}_hour 特征，表示 {col} 的小时信息")
    
    # 4. Upvotes相关特征（如果有upvotes列）
    if 'upvotes' in df_featured.columns:
        # 创建upvotes分类特征
        upvotes_median = df_featured['upvotes'].median()
        df_featured['high_upvotes'] = (df_featured['upvotes'] > upvotes_median).astype(int)
        print(f"已添加 high_upvotes 特征，表示upvotes是否高于中位数 {upvotes_median}")
    
    return df_featured

# -----------------------------------------------------------------------------
# 第9步：保存清洗后的数据
# -----------------------------------------------------------------------------
def save_cleaned_data(df, output_file):
    """
    保存清洗后的数据集
    """
    print(f"\n{'='*20} 保存清洗后的数据 {'='*20}")
    
    if df is None:
        print("没有数据可保存")
        return
    
    # 确定输出格式
    if output_file.endswith('.csv'):
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
    elif output_file.endswith('.xlsx'):
        df.to_excel(output_file, index=False)
    elif output_file.endswith('.json'):
        df.to_json(output_file, orient='records', force_ascii=False)
    else:
        # 默认为CSV
        output_file = output_file + '.csv'
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"已将清洗后的数据保存至 {output_file}")
    print(f"数据集大小: {df.shape[0]} 行 x {df.shape[1]} 列")

# -----------------------------------------------------------------------------
# 主函数
# -----------------------------------------------------------------------------
def main():
    # 设置输入和输出文件路径
    input_file = "emotions_dataset.csv"  # 替换为您的输入文件
    output_file = "emotions_dataset_cleaned.csv"  # 清洗后的输出文件
    
    print(f"\n{'='*20} 开始数据清洗流程 {'='*20}")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    
    # 步骤1: 加载和探索数据
    df = load_and_explore_data(input_file)
    
    # 步骤2: 处理缺失值
    df = handle_missing_values(df)
    
    # 步骤3: 转换数据类型
    df = convert_data_types(df)
    
    # 步骤4: 处理异常值
    df = handle_outliers(df)
    
    # 步骤5: 删除重复行
    df = remove_duplicates(df)
    
    # 步骤6: 清洗文本数据
    df = clean_text_data(df)
    
    # 步骤7: 标准化和归一化数据
    df = normalize_data(df)
    
    # 步骤8: 特征工程
    df = engineer_features(df)
    
    # 步骤9: 保存清洗后的数据
    save_cleaned_data(df, output_file)
    
    print(f"\n{'='*20} 数据清洗流程完成 {'='*20}")

if __name__ == "__main__":
    main() 