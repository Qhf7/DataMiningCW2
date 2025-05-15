import os
import numpy as np
import pandas as pd
import re
import jieba
import torch
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from transformers import BertTokenizer, BertModel
import logging
from collections import Counter


class TextPreprocessor:
    """文本预处理器：负责文本清洗、分词和特征提取"""
    
    def __init__(self, 
                 stop_words_path=None, 
                 bert_model_name='bert-base-chinese',
                 use_cuda=True):
        """
        初始化文本预处理器
        
        Args:
            stop_words_path: 停用词文件路径，如果为None则不使用停用词
            bert_model_name: BERT模型名称
            use_cuda: 是否使用GPU
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
        
        # 加载停用词
        self.stop_words = set()
        if stop_words_path and os.path.exists(stop_words_path):
            with open(stop_words_path, 'r', encoding='utf-8') as f:
                self.stop_words = set([line.strip() for line in f])
        
        # 初始化BERT分词器
        try:
            self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
            self.bert_model = BertModel.from_pretrained(bert_model_name).to(self.device)
            self.bert_model.eval()  # 设置为评估模式
            self.bert_available = True
        except Exception as e:
            logging.warning(f"BERT模型加载失败: {str(e)}")
            self.bert_available = False
        
        # 初始化TF-IDF向量化器
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000)
        
        # 初始化LDA主题模型
        self.lda_model = LatentDirichletAllocation(n_components=10, random_state=42)
        
        # 初始化词袋模型
        self.bow_vectorizer = CountVectorizer(max_features=1000)
        
        # 记录是否已经拟合
        self.is_fitted = False
    
    def clean_text(self, text):
        """
        文本清洗
        
        Args:
            text: 输入文本
            
        Returns:
            清洗后的文本
        """
        if not isinstance(text, str):
            return ""
            
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text)
        
        # 移除特殊字符
        text = re.sub(r'[^\w\s\u4e00-\u9fff。，！？、；：""''（）【】《》]', '', text)
        
        # 移除数字（可选）
        # text = re.sub(r'\d+', '', text)
        
        return text.strip()
    
    def segment(self, text, remove_stop_words=True):
        """
        分词
        
        Args:
            text: 输入文本
            remove_stop_words: 是否移除停用词
            
        Returns:
            分词后的列表
        """
        text = self.clean_text(text)
        
        # 使用jieba分词
        words = list(jieba.cut(text))
        
        # 移除停用词
        if remove_stop_words and self.stop_words:
            words = [word for word in words if word not in self.stop_words and len(word.strip()) > 0]
            
        return words
    
    def extract_statistical_features(self, text):
        """
        提取统计特征
        
        Args:
            text: 输入文本
            
        Returns:
            特征字典
        """
        features = {}
        
        # 清洗文本
        cleaned_text = self.clean_text(text)
        
        # 分词
        words = self.segment(cleaned_text)
        
        # 文本长度特征
        features['text_length'] = len(cleaned_text)
        features['word_count'] = len(words)
        features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
        
        # 词频特征
        word_counts = Counter(words)
        features['unique_word_count'] = len(word_counts)
        features['lexical_diversity'] = len(word_counts) / len(words) if words else 0
        
        # 标点符号特征
        features['comma_count'] = cleaned_text.count('，')
        features['period_count'] = cleaned_text.count('。')
        features['question_count'] = cleaned_text.count('？')
        features['exclamation_count'] = cleaned_text.count('！')
        
        # 情感词汇计数
        depression_words = ["难过", "无助", "悲伤", "痛苦", "孤独", "绝望", "失落",
                            "疲惫", "自责", "愧疚", "压力", "焦虑", "恐惧", "失眠"]
        positive_words = ["快乐", "开心", "不错", "好", "满足", "愉快", "高兴",
                          "热爱", "享受", "期待", "满意", "幸福", "成功", "希望"]
        
        features['depression_word_count'] = sum(1 for word in depression_words if word in cleaned_text)
        features['positive_word_count'] = sum(1 for word in positive_words if word in cleaned_text)
        
        # 情感比率
        total_sentiment_words = features['depression_word_count'] + features['positive_word_count']
        features['depression_ratio'] = features['depression_word_count'] / total_sentiment_words if total_sentiment_words > 0 else 0
        features['positive_ratio'] = features['positive_word_count'] / total_sentiment_words if total_sentiment_words > 0 else 0
        
        # 简单情感分数
        features['sentiment_score'] = features['positive_word_count'] - features['depression_word_count']
        
        return features
    
    def fit(self, texts):
        """
        拟合预处理模型
        
        Args:
            texts: 文本列表
        """
        # 清洗并分词
        cleaned_texts = [self.clean_text(text) for text in texts]
        segmented_texts = [' '.join(self.segment(text)) for text in cleaned_texts]
        
        # 拟合TF-IDF向量化器
        self.tfidf_vectorizer.fit(segmented_texts)
        
        # 拟合词袋模型
        self.bow_vectorizer.fit(segmented_texts)
        
        # 拟合LDA主题模型
        bow_matrix = self.bow_vectorizer.transform(segmented_texts)
        self.lda_model.fit(bow_matrix)
        
        # 训练Word2Vec模型（可选）
        if len(texts) > 10:  # 确保有足够的数据
            segmented_docs = [self.segment(text) for text in cleaned_texts]
            self.word2vec_model = Word2Vec(sentences=segmented_docs, vector_size=100, window=5, min_count=1, workers=4)
        
        self.is_fitted = True
        
    def transform(self, text, output_format='bert'):
        """
        将文本转换为特征向量
        
        Args:
            text: 输入文本
            output_format: 输出格式，可选'statistical', 'tfidf', 'lda', 'word2vec', 'bert', 'combined'
            
        Returns:
            特征向量
        """
        if not self.is_fitted and output_format not in ['statistical', 'bert']:
            raise ValueError("模型尚未拟合，请先调用fit方法")
        
        # 清洗文本
        cleaned_text = self.clean_text(text)
        
        if output_format == 'statistical':
            # 提取统计特征
            features_dict = self.extract_statistical_features(cleaned_text)
            return np.array(list(features_dict.values()))
            
        elif output_format == 'tfidf':
            # 提取TF-IDF特征
            segmented_text = ' '.join(self.segment(cleaned_text))
            return self.tfidf_vectorizer.transform([segmented_text]).toarray().flatten()
            
        elif output_format == 'lda':
            # 提取主题特征
            segmented_text = ' '.join(self.segment(cleaned_text))
            bow_vector = self.bow_vectorizer.transform([segmented_text])
            return self.lda_model.transform(bow_vector).flatten()
            
        elif output_format == 'word2vec':
            # 提取Word2Vec特征
            if not hasattr(self, 'word2vec_model'):
                raise ValueError("Word2Vec模型尚未训练")
                
            words = self.segment(cleaned_text)
            valid_words = [word for word in words if word in self.word2vec_model.wv]
            
            if not valid_words:
                return np.zeros(self.word2vec_model.vector_size)
                
            word_vectors = [self.word2vec_model.wv[word] for word in valid_words]
            return np.mean(word_vectors, axis=0)
            
        elif output_format == 'bert':
            # 提取BERT特征
            if not self.bert_available:
                raise ValueError("BERT模型不可用")
                
            # 对文本进行分词和编码
            inputs = self.bert_tokenizer(cleaned_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 获取BERT输出
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                
            # 使用[CLS]标记的输出作为文本表示
            cls_output = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
            return cls_output
            
        elif output_format == 'combined':
            # 组合多种特征
            statistical_features = self.extract_statistical_features(cleaned_text)
            statistical_vector = np.array(list(statistical_features.values()))
            
            # TF-IDF特征
            segmented_text = ' '.join(self.segment(cleaned_text))
            tfidf_vector = self.tfidf_vectorizer.transform([segmented_text]).toarray().flatten()
            
            # 主题特征
            bow_vector = self.bow_vectorizer.transform([segmented_text])
            lda_vector = self.lda_model.transform(bow_vector).flatten()
            
            # 结合所有特征
            combined_vector = np.concatenate([
                statistical_vector,
                tfidf_vector[:100],  # 限制TF-IDF特征数量
                lda_vector
            ])
            
            return combined_vector
        
        else:
            raise ValueError(f"不支持的输出格式: {output_format}")
            
    def get_bert_tokenizer_outputs(self, text, max_length=512):
        """
        获取BERT分词器的输出，用于模型输入
        
        Args:
            text: 输入文本
            max_length: 最大文本长度
            
        Returns:
            input_ids, attention_mask
        """
        if not self.bert_available:
            raise ValueError("BERT模型不可用")
            
        # 对文本进行分词和编码
        cleaned_text = self.clean_text(text)
        outputs = self.bert_tokenizer(cleaned_text, padding='max_length', truncation=True, 
                                      max_length=max_length, return_tensors="pt")
        
        return outputs.input_ids, outputs.attention_mask


class TextAugmenter:
    """文本增强器：提供多种文本增强方法"""
    
    def __init__(self, preprocessor=None):
        """
        初始化文本增强器
        
        Args:
            preprocessor: TextPreprocessor实例，用于文本预处理
        """
        self.preprocessor = preprocessor or TextPreprocessor()
        
        # 同义词字典
        self.synonyms = {
            "难过": ["悲伤", "忧伤", "伤心", "悲痛", "哀伤", "悲恸"],
            "无助": ["无力", "绝望", "无望", "束手无策", "无能为力", "无依无靠"],
            "快乐": ["高兴", "开心", "愉悦", "喜悦", "欢喜", "欣喜"],
            "感觉": ["觉得", "认为", "体会到", "感受到", "体验到"],
            "疲惫": ["疲劳", "困乏", "精疲力竭", "筋疲力尽", "疲乏"],
            "焦虑": ["紧张", "不安", "担忧", "忧虑", "烦躁", "惶恐"],
            "开心": ["高兴", "欢乐", "愉快", "欢喜", "快活", "欢快"],
            "痛苦": ["痛楚", "苦痛", "折磨", "煎熬", "苦楚"],
            "自责": ["自咎", "内疚", "自责自疚", "责备自己", "愧疚"],
            "失眠": ["睡不着", "无法入眠", "难以入睡", "彻夜难眠", "辗转反侧"]
        }
        
    def synonym_replacement(self, text, n=2):
        """
        同义词替换
        
        Args:
            text: 输入文本
            n: 替换词数量
            
        Returns:
            增强后的文本
        """
        if not isinstance(text, str) or not text.strip():
            return text
            
        words = self.preprocessor.segment(text, remove_stop_words=False)
        new_words = words.copy()
        
        # 记录可替换的词位置
        replaced_positions = []
        
        # 尝试替换n个词
        replacement_count = 0
        for i, word in enumerate(words):
            if word in self.synonyms and i not in replaced_positions and replacement_count < n:
                synonyms = self.synonyms[word]
                if synonyms:
                    # 随机选择一个同义词进行替换
                    replacement = np.random.choice(synonyms)
                    new_words[i] = replacement
                    replaced_positions.append(i)
                    replacement_count += 1
        
        # 如果没有足够的同义词替换，可以再尝试其他方法
        
        return ''.join(new_words)
    
    def random_insertion(self, text, n=1):
        """
        随机插入
        
        Args:
            text: 输入文本
            n: 插入次数
            
        Returns:
            增强后的文本
        """
        if not isinstance(text, str) or not text.strip():
            return text
            
        words = self.preprocessor.segment(text, remove_stop_words=False)
        
        # 情感词汇
        depression_words = ["孤独", "疲惫", "失落", "压抑", "痛苦", "悲伤", "绝望"]
        positive_words = ["轻松", "舒适", "满足", "温暖", "安心", "开心", "喜悦"]
        
        # 判断文本情感倾向
        if any(word in text for word in ["难过", "无助", "痛苦", "焦虑", "绝望"]):
            word_list = depression_words
        elif any(word in text for word in ["不错", "开心", "快乐", "满足", "喜悦"]):
            word_list = positive_words
        else:
            # 如果无法判断情感，随机选择
            word_list = depression_words if np.random.random() < 0.5 else positive_words
            
        # 执行插入
        for _ in range(n):
            word_to_insert = np.random.choice(word_list)
            insert_pos = np.random.randint(0, len(words) + 1)
            words.insert(insert_pos, word_to_insert)
            
        return ''.join(words)
    
    def random_swap(self, text, n=1):
        """
        随机交换
        
        Args:
            text: 输入文本
            n: 交换次数
            
        Returns:
            增强后的文本
        """
        if not isinstance(text, str) or not text.strip():
            return text
            
        words = self.preprocessor.segment(text, remove_stop_words=False)
        
        if len(words) < 2:
            return text
            
        # 执行交换
        for _ in range(n):
            idx1, idx2 = np.random.choice(len(words), 2, replace=False)
            words[idx1], words[idx2] = words[idx2], words[idx1]
            
        return ''.join(words)
    
    def back_translation(self, text):
        """
        回译（模拟）
        
        实际应用中可接入翻译API实现中英互译
        
        Args:
            text: 输入文本
            
        Returns:
            模拟回译后的文本
        """
        # 这里只是模拟回译效果，实际应用中应该调用翻译API
        depression_translations = {
            "我感到非常难过和无助": "我感到极度悲伤和绝望。似乎没有任何事物能够让我感到一丝希望。",
            "我最近总是感到疲惫": "近期我一直感到精疲力竭，仿佛被抽干了所有精力。",
            "我经常无法入睡": "我总是辗转反侧，难以进入梦乡，即使睡着也会频繁醒来。",
            "我对以前喜欢的事情失去了兴趣": "那些曾经让我满怀热情的活动，现在看起来毫无吸引力，只感到空虚。",
        }
        
        normal_translations = {
            "今天是个不错的日子": "今天天气宜人，我的心情非常愉快，感到十分开心。",
            "我最近工作很顺利": "近期我的工作进展非常顺利，一切都按计划进行，我感到很满足。",
            "我和朋友们相处得很愉快": "我与朋友们之间的互动充满欢乐，我们共度了许多美好时光。",
            "我的睡眠质量很好": "我每晚都能够安稳入睡，醒来时感到精力充沛，为新的一天做好准备。",
        }
        
        # 查找匹配的文本进行替换
        for source, target in {**depression_translations, **normal_translations}.items():
            if source in text:
                return text.replace(source, target)
                
        return text
    
    def eda_augment(self, text, alpha=0.3):
        """
        整合多种增强方法
        
        Args:
            text: 输入文本
            alpha: 应用每种增强方法的概率
            
        Returns:
            增强后的文本
        """
        if not isinstance(text, str) or not text.strip():
            return text
            
        augmented_text = text
        
        # 以一定概率应用每种增强方法
        if np.random.random() < alpha:
            augmented_text = self.synonym_replacement(augmented_text)
            
        if np.random.random() < alpha:
            augmented_text = self.random_insertion(augmented_text)
            
        if np.random.random() < alpha:
            augmented_text = self.random_swap(augmented_text)
            
        if np.random.random() < alpha:
            augmented_text = self.back_translation(augmented_text)
            
        return augmented_text
    
    def generate_augmented_samples(self, text, n_samples=4):
        """
        生成多个增强样本
        
        Args:
            text: 输入文本
            n_samples: 生成样本数量
            
        Returns:
            增强样本列表
        """
        if not isinstance(text, str) or not text.strip():
            return [text] * n_samples
            
        augmented_samples = []
        methods = [
            self.synonym_replacement,
            self.random_insertion,
            self.random_swap,
            self.back_translation,
            self.eda_augment
        ]
        
        # 生成增强样本
        for _ in range(n_samples):
            # 随机选择一种增强方法
            method = np.random.choice(methods)
            augmented_text = method(text)
            augmented_samples.append(augmented_text)
            
        return augmented_samples 