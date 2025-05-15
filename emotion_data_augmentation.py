import pandas as pd
import numpy as np
import nltk
from nltk.corpus import wordnet
import random
import re
import jieba
from transformers import BertTokenizer, BertModel
import torch
from sklearn.utils import resample
from collections import Counter

# 下载必要的NLTK资源
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('punkt')

class EmotionDataAugmentation:
    """情绪数据增强类，用于多标签分类任务"""
    
    def __init__(self, random_state=42):
        """
        初始化数据增强器
        
        参数:
            random_state: 随机种子，确保结果可重现
        """
        self.random_state = random_state
        random.seed(random_state)
        np.random.seed(random_state)
        
        # 加载中文BERT模型用于同义词替换和句子嵌入
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.bert_model = BertModel.from_pretrained('bert-base-chinese')
        
        # 情绪类别
        self.emotion_categories = [
            "悲伤", "绝望", "无价值感", "孤独", "自我厌恶", 
            "空虚", "缺乏能量", "愤怒", "自杀意图", "认知功能障碍"
        ]
        
    def load_data(self, file_path):
        """
        加载情绪数据集
        
        参数:
            file_path: CSV文件路径，包含文本和情绪标签
            
        返回:
            加载的数据框
        """
        df = pd.read_csv(file_path)
        print(f"原始数据集大小: {len(df)} 条记录")
        return df
    
    def analyze_label_distribution(self, df, label_column='feelings'):
        """
        分析标签分布情况
        
        参数:
            df: 数据框
            label_column: 标签列名
            
        返回:
            标签分布统计
        """
        # 计算每个情绪类别的出现次数
        emotion_counts = {}
        
        for emotions_str in df[label_column]:
            emotions = str(emotions_str).split(',')
            for emotion in emotions:
                emotion = emotion.strip()
                if emotion:
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # 转换为DataFrame便于查看
        distribution_df = pd.DataFrame({
            '情绪类别': list(emotion_counts.keys()),
            '出现次数': list(emotion_counts.values())
        }).sort_values('出现次数', ascending=False)
        
        print("标签分布情况:")
        print(distribution_df)
        
        return distribution_df
    
    def _synonym_replacement(self, text, n=2):
        """
        同义词替换数据增强
        
        参数:
            text: 输入文本
            n: 要替换的词数量
            
        返回:
            增强后的文本
        """
        words = list(jieba.cut(text))
        
        # 如果文本太短，减少替换数量
        n = min(n, max(1, len(words) // 3))
        
        # 随机选择n个词来替换
        replace_indices = random.sample(range(len(words)), n)
        
        for i in replace_indices:
            word = words[i]
            # 使用BERT模型获取相似词
            if len(word) > 1:  # 只替换长度大于1的词
                try:
                    # 使用BERT获取上下文相关的同义词
                    inputs = self.tokenizer(word, return_tensors="pt")
                    with torch.no_grad():
                        outputs = self.bert_model(**inputs)
                    
                    # 简单的相似词替换策略 - 实际项目中可以使用更复杂的同义词查找方法
                    synonyms = ["类似的词1", "类似的词2", "相似表达"]  # 此处应使用真实同义词库
                    if synonyms:
                        words[i] = random.choice(synonyms)
                except:
                    continue
        
        return ''.join(words)
    
    def _random_insertion(self, text, n=2):
        """
        随机插入情绪相关词语
        
        参数:
            text: 输入文本
            n: 要插入的词数量
            
        返回:
            增强后的文本
        """
        words = list(jieba.cut(text))
        
        # 情绪相关词库 - 在实际应用中应扩充此词库
        emotion_words = {
            "悲伤": ["伤心", "难过", "哀伤", "忧伤", "悲痛", "心碎"],
            "绝望": ["绝望", "无望", "无助", "走投无路", "失望透顶"],
            "无价值感": ["没用", "无价值", "一无是处", "无足轻重", "毫无意义"],
            "孤独": ["孤独", "寂寞", "孤单", "形单影只", "独自一人"],
            "自我厌恶": ["恨自己", "讨厌自己", "自责", "自我嫌弃", "厌恶自己"],
            "空虚": ["空虚", "空洞", "虚无", "没有意义", "乏味"],
            "缺乏能量": ["疲惫", "无力", "累", "精疲力竭", "提不起劲"],
            "愤怒": ["生气", "愤怒", "恼火", "暴怒", "气愤"],
            "自杀意图": ["想死", "结束生命", "自杀", "了结自己", "解脱"],
            "认知功能障碍": ["记不住", "思维混乱", "注意力不集中", "迷糊", "健忘"]
        }
        
        # 随机选择一个情绪类别和对应的词
        emotion = random.choice(list(emotion_words.keys()))
        emotional_words_to_insert = random.sample(emotion_words[emotion], min(n, len(emotion_words[emotion])))
        
        for _ in range(n):
            if len(words) > 0:
                insert_position = random.randint(0, len(words))
                word_to_insert = random.choice(emotional_words_to_insert)
                words.insert(insert_position, word_to_insert)
        
        return ''.join(words)
    
    def _random_swap(self, text, n=2):
        """
        随机交换词语位置
        
        参数:
            text: 输入文本
            n: 要交换的次数
            
        返回:
            增强后的文本
        """
        words = list(jieba.cut(text))
        
        if len(words) <= 1:
            return text
            
        n = min(n, len(words) // 2)  # 确保不会交换太多
        
        for _ in range(n):
            i, j = random.sample(range(len(words)), 2)
            words[i], words[j] = words[j], words[i]
            
        return ''.join(words)
    
    def _back_translation(self, text):
        """
        回译增强
        注意: 此函数需要外部翻译API，这里只是示例框架
        
        参数:
            text: 输入文本
            
        返回:
            回译后的文本
        """
        # 这里应该调用翻译API进行中英互译
        # 为简化演示，这里直接返回原文
        try:
            # 中文 -> 英文 -> 中文的回译过程
            # translated_to_english = translate_api(text, source='zh', target='en')
            # back_to_chinese = translate_api(translated_to_english, source='en', target='zh')
            # return back_to_chinese
            return text + "(回译版本)"  # 示例
        except:
            return text
    
    def augment_minority_class(self, df, label_column='feelings', text_column='text', 
                              min_samples=30, methods=None):
        """
        对少数类进行数据增强
        
        参数:
            df: 数据框
            label_column: 标签列名
            text_column: 文本列名
            min_samples: 每个类别的最小样本数
            methods: 使用的增强方法列表
            
        返回:
            增强后的数据框
        """
        if methods is None:
            methods = ['synonym', 'insertion', 'swap', 'backtrans']
        
        # 统计每个情绪标签的出现次数
        emotion_samples = {emotion: [] for emotion in self.emotion_categories}
        
        # 遍历数据，按情绪分组
        for idx, row in df.iterrows():
            emotions = str(row[label_column]).split(',')
            emotions = [e.strip() for e in emotions if e.strip()]
            
            for emotion in emotions:
                if emotion in self.emotion_categories:
                    emotion_samples[emotion].append(idx)
        
        # 存储新增的样本
        augmented_samples = []
        
        # 对每个少数类情绪进行增强
        for emotion, indices in emotion_samples.items():
            current_count = len(indices)
            print(f"情绪 '{emotion}' 原始样本数: {current_count}")
            
            if current_count < min_samples:
                # 需要增强的样本数
                samples_to_add = min_samples - current_count
                
                # 从现有样本中随机选择用于增强
                selected_indices = random.choices(indices, k=samples_to_add)
                
                for idx in selected_indices:
                    original_text = df.loc[idx, text_column]
                    original_emotions = df.loc[idx, label_column]
                    
                    # 随机选择一种增强方法
                    method = random.choice(methods)
                    
                    if method == 'synonym':
                        augmented_text = self._synonym_replacement(original_text)
                    elif method == 'insertion':
                        augmented_text = self._random_insertion(original_text)
                    elif method == 'swap':
                        augmented_text = self._random_swap(original_text)
                    elif method == 'backtrans':
                        augmented_text = self._back_translation(original_text)
                    else:
                        augmented_text = original_text
                    
                    augmented_samples.append({
                        text_column: augmented_text,
                        label_column: original_emotions,
                        'augmentation_method': method,
                        'original_index': idx
                    })
        
        # 创建增强样本的数据框
        if augmented_samples:
            augmented_df = pd.DataFrame(augmented_samples)
            result_df = pd.concat([df, augmented_df], ignore_index=True)
            print(f"增强后的数据集大小: {len(result_df)} 条记录")
            return result_df
        else:
            print("没有进行数据增强")
            return df
    
    def generate_balanced_samples(self, df, label_column='feelings', text_column='text'):
        """
        生成平衡的训练集，对过采样和数据增强的组合
        
        参数:
            df: 数据框
            label_column: 标签列名
            text_column: 文本列名
            
        返回:
            平衡后的数据框
        """
        # 对数据集进行数据增强
        augmented_df = self.augment_minority_class(df, label_column, text_column)
        
        # 统计增强后各情绪的样本数
        emotion_counts = {}
        for idx, row in augmented_df.iterrows():
            emotions = str(row[label_column]).split(',')
            emotions = [e.strip() for e in emotions if e.strip()]
            
            for emotion in emotions:
                if emotion in self.emotion_categories:
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # 找出最多的类别的样本数
        max_samples = max(emotion_counts.values()) if emotion_counts else 0
        
        # 对每个情绪类别进行过采样或欠采样以平衡样本
        balanced_samples = []
        
        for emotion in self.emotion_categories:
            # 获取包含此情绪的样本索引
            emotion_indices = []
            for idx, row in augmented_df.iterrows():
                emotions = str(row[label_column]).split(',')
                if emotion in [e.strip() for e in emotions]:
                    emotion_indices.append(idx)
            
            # 如果没有此情绪的样本，跳过
            if not emotion_indices:
                continue
                
            # 使用过采样方法平衡样本
            if len(emotion_indices) < max_samples:
                # 过采样少数类
                sampled_indices = resample(
                    emotion_indices,
                    replace=True,
                    n_samples=max_samples,
                    random_state=self.random_state
                )
            else:
                # 对多数类进行欠采样
                sampled_indices = resample(
                    emotion_indices,
                    replace=False,
                    n_samples=max_samples,
                    random_state=self.random_state
                )
            
            for idx in sampled_indices:
                sample = augmented_df.loc[idx].copy()
                balanced_samples.append(sample)
        
        # 创建平衡后的数据框
        balanced_df = pd.DataFrame(balanced_samples)
        print(f"平衡后的数据集大小: {len(balanced_df)} 条记录")
        
        return balanced_df
    
    def save_augmented_data(self, df, output_file):
        """
        保存增强后的数据
        
        参数:
            df: 数据框
            output_file: 输出文件路径
        """
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"增强后的数据已保存至: {output_file}")


# 使用示例
if __name__ == "__main__":
    # 创建数据增强器
    augmenter = EmotionDataAugmentation()
    
    # 加载原始数据
    data = augmenter.load_data("emotions_dataset.csv")
    
    # 分析标签分布
    augmenter.analyze_label_distribution(data)
    
    # 对少数类进行数据增强
    augmented_data = augmenter.augment_minority_class(
        data, 
        label_column='feelings',
        text_column='text',
        min_samples=30
    )
    
    # 生成平衡的数据集
    balanced_data = augmenter.generate_balanced_samples(
        augmented_data, 
        label_column='feelings',
        text_column='text'
    )
    
    # 保存增强后的数据
    augmenter.save_augmented_data(balanced_data, "augmented_emotions_dataset.csv") 