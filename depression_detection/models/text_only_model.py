import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertModel, RobertaModel, XLNetModel, ElectraModel, BertTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation


class TextFeatureExtractor(nn.Module):
    """文本特征提取器"""
    
    def __init__(self, model_type='bert', freeze_bert_layers=6):
        """
        初始化文本特征提取器
        
        Args:
            model_type: 预训练模型类型，可选 'bert', 'roberta', 'xlnet', 'electra'
            freeze_bert_layers: 冻结BERT底部N层，如果为0则不冻结
        """
        super(TextFeatureExtractor, self).__init__()
        
        # 选择预训练模型
        if model_type == 'bert':
            self.bert = BertModel.from_pretrained('bert-base-chinese')
            self.hidden_size = self.bert.config.hidden_size
        elif model_type == 'roberta':
            self.bert = RobertaModel.from_pretrained('hfl/chinese-roberta-wwm-ext')
            self.hidden_size = self.bert.config.hidden_size
        elif model_type == 'xlnet':
            self.bert = XLNetModel.from_pretrained('hfl/chinese-xlnet-base')
            self.hidden_size = self.bert.config.d_model
        elif model_type == 'electra':
            self.bert = ElectraModel.from_pretrained('hfl/chinese-electra-base-discriminator')
            self.hidden_size = self.bert.config.hidden_size
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
            
        self.model_type = model_type
        
        # 冻结底部N层
        if freeze_bert_layers > 0:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
                
            for i in range(freeze_bert_layers):
                if model_type in ['bert', 'roberta', 'electra']:
                    for param in self.bert.encoder.layer[i].parameters():
                        param.requires_grad = False
                elif model_type == 'xlnet':
                    for param in self.bert.layer[i].parameters():
                        param.requires_grad = False
                        
        # 特征提取层
        self.feature_projection = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.LayerNorm(512),
            nn.Dropout(0.2),
            nn.ReLU()
        )
        
    def forward(self, input_ids, attention_mask):
        """
        前向传播
        
        Args:
            input_ids: 输入ID [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            
        Returns:
            text_features: 文本特征 [batch_size, hidden_size]
        """
        # 获取BERT输出
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # 使用[CLS]标记的隐藏状态作为文本表示
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # 特征投影
        text_features = self.feature_projection(cls_output)
        
        return text_features


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力机制"""
    
    def __init__(self, hidden_dim, num_heads):
        """
        初始化多头自注意力
        
        Args:
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
        """
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        # 确保hidden_dim可以被num_heads整除
        assert hidden_dim % num_heads == 0
        
        self.depth = hidden_dim // num_heads
        
        self.wq = nn.Linear(hidden_dim, hidden_dim)
        self.wk = nn.Linear(hidden_dim, hidden_dim)
        self.wv = nn.Linear(hidden_dim, hidden_dim)
        
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        
    def split_heads(self, x, batch_size):
        """
        分割头
        
        Args:
            x: 输入张量
            batch_size: 批量大小
            
        Returns:
            分割后的张量
        """
        # 将隐藏层维度分割成num_heads * depth
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        # 变换维度为[batch_size, num_heads, seq_len, depth]
        return x.permute(0, 2, 1, 3)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, seq_len, hidden_dim]
            
        Returns:
            注意力输出
        """
        batch_size = x.size(0)
        
        # 线性变换
        q = self.wq(x)  # [batch_size, seq_len, hidden_dim]
        k = self.wk(x)  # [batch_size, seq_len, hidden_dim]
        v = self.wv(x)  # [batch_size, seq_len, hidden_dim]
        
        # 分割多头
        q = self.split_heads(q, batch_size)  # [batch_size, num_heads, seq_len, depth]
        k = self.split_heads(k, batch_size)  # [batch_size, num_heads, seq_len, depth]
        v = self.split_heads(v, batch_size)  # [batch_size, num_heads, seq_len, depth]
        
        # 计算注意力权重
        matmul_qk = torch.matmul(q, k.transpose(-1, -2))  # [batch_size, num_heads, seq_len, seq_len]
        
        # 缩放
        dk = torch.tensor(self.depth, dtype=torch.float32)
        scaled_attention_logits = matmul_qk / torch.sqrt(dk)
        
        # Softmax
        attention_weights = F.softmax(scaled_attention_logits, dim=-1)  # [batch_size, num_heads, seq_len, seq_len]
        
        # 应用注意力权重
        output = torch.matmul(attention_weights, v)  # [batch_size, num_heads, seq_len, depth]
        
        # 重塑为原始维度
        output = output.permute(0, 2, 1, 3).contiguous()  # [batch_size, seq_len, num_heads, depth]
        output = output.view(batch_size, -1, self.hidden_dim)  # [batch_size, seq_len, hidden_dim]
        
        # 最终线性层
        output = self.fc(output)  # [batch_size, seq_len, hidden_dim]
        
        return output


class TextFeatureFusion(nn.Module):
    """文本特征融合层"""
    
    def __init__(self, hidden_dim, extra_feature_dim, dropout=0.3):
        """
        初始化文本特征融合层
        
        Args:
            hidden_dim: 隐藏层维度
            extra_feature_dim: 额外特征维度
            dropout: Dropout率
        """
        super(TextFeatureFusion, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.extra_feature_dim = extra_feature_dim
        
        # 多头自注意力层
        self.self_attention = MultiHeadSelfAttention(hidden_dim, num_heads=8)
        
        # 额外特征投影层
        self.extra_projection = nn.Sequential(
            nn.Linear(extra_feature_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU()
        )
        
    def forward(self, text_features, extra_features):
        """
        前向传播
        
        Args:
            text_features: 文本特征 [batch_size, hidden_dim]
            extra_features: 额外特征 [batch_size, extra_feature_dim]
            
        Returns:
            融合特征
        """
        # 自注意力处理
        text_features = text_features.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        attended_features = self.self_attention(text_features).squeeze(1)  # [batch_size, hidden_dim]
        
        # 投影额外特征
        projected_extra = self.extra_projection(extra_features)  # [batch_size, hidden_dim // 2]
        
        # 特征融合
        combined_features = torch.cat([attended_features, projected_extra], dim=1)  # [batch_size, hidden_dim + hidden_dim // 2]
        fused_features = self.fusion_layer(combined_features)  # [batch_size, hidden_dim]
        
        return fused_features


class TextOnlyDepressionModel(nn.Module):
    """文本模态抑郁症检测模型"""
    
    def __init__(self, hidden_dim=512, dropout=0.3, num_classes=2, model_type='bert'):
        """
        初始化文本模态抑郁症检测模型
        
        Args:
            hidden_dim: 隐藏层维度
            dropout: Dropout率
            num_classes: 类别数
            model_type: 预训练模型类型
        """
        super(TextOnlyDepressionModel, self).__init__()
        
        # 文本特征提取器
        self.text_feature_extractor = TextFeatureExtractor(model_type=model_type)
        
        # 特征融合层
        self.feature_fusion = TextFeatureFusion(
            hidden_dim=hidden_dim,
            extra_feature_dim=20,  # 根据实际的额外特征维度调整
            dropout=dropout
        )
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # 情感分析任务（辅助任务）
        self.sentiment_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()  # 输出范围为[-1, 1]，表示情感极性
        )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        for module in [self.classifier, self.sentiment_predictor]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, input_ids, attention_mask, extra_features):
        """
        前向传播
        
        Args:
            input_ids: 输入ID [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            extra_features: 额外特征 [batch_size, extra_feature_dim]
            
        Returns:
            logits: 分类logits
            sentiment_score: 情感分数
        """
        # 提取文本特征
        text_features = self.text_feature_extractor(input_ids, attention_mask)
        
        # 特征融合
        fused_features = self.feature_fusion(text_features, extra_features)
        
        # 分类
        logits = self.classifier(fused_features)
        
        # 情感预测
        sentiment_score = self.sentiment_predictor(fused_features)
        
        return logits, sentiment_score


class TextAugmentation:
    """文本增强技术：整合多种先进的文本增强方法"""
    
    def __init__(self, tokenizer=None):
        """
        初始化文本增强器
        
        Args:
            tokenizer: 分词器，默认使用BERT分词器
        """
        self.tokenizer = tokenizer if tokenizer else BertTokenizer.from_pretrained('bert-base-uncased')
        
    def synonym_replacement(self, text, n=1):
        """
        同义词替换增强
        
        Args:
            text: 输入文本
            n: 替换词数量
            
        Returns:
            增强后的文本
        """
        # 简单示例，实际应用中可使用WordNet或词嵌入模型查找同义词
        depression_synonyms = {
            "难过": ["悲伤", "忧伤", "伤心", "悲痛"],
            "无助": ["无力", "绝望", "无望", "束手无策"],
            "快乐": ["高兴", "开心", "愉悦", "喜悦"],
            "感觉": ["觉得", "认为", "体会到", "感受到"]
        }
        
        for word, replacements in depression_synonyms.items():
            if word in text and np.random.random() < 0.5:  # 50%概率替换
                replacement = np.random.choice(replacements)
                text = text.replace(word, replacement, 1)
                n -= 1
                if n <= 0:
                    break
                    
        return text
    
    def back_translation(self, text):
        """
        回译增强（模拟）
        
        实际应用中可接入翻译API实现中英互译
        
        Returns:
            模拟回译后的文本
        """
        if "我感到非常难过" in text:
            return "我感到极其悲伤和绝望。似乎没有什么能够让我感到一丝快乐。"
        elif "今天是个不错的日子" in text:
            return "今天天气很好，我的心情非常愉快，感到十分开心。"
        return text
    
    def random_insertion(self, text):
        """随机插入情感词汇"""
        depression_words = ["孤独", "疲惫", "失落", "压抑", "痛苦"]
        positive_words = ["轻松", "舒适", "满足", "温暖", "安心"]
        
        if "难过" in text or "无助" in text:
            word = np.random.choice(depression_words)
            return text + f" 我感到{word}。"
        elif "不错" in text or "开心" in text:
            word = np.random.choice(positive_words)
            return text + f" 我感到{word}。"
        return text
    
    def eda_augment(self, text):
        """整合多种增强方法"""
        if np.random.random() < 0.3:  # 30%概率使用同义词替换
            text = self.synonym_replacement(text)
        elif np.random.random() < 0.3:  # 30%概率使用回译
            text = self.back_translation(text)
        elif np.random.random() < 0.3:  # 30%概率使用随机插入
            text = self.random_insertion(text)
        return text


class TextFeatureMining:
    """文本特征挖掘：使用传统和先进的文本挖掘技术提取特征"""
    
    def __init__(self, n_topics=10):
        """
        初始化文本特征挖掘器
        
        Args:
            n_topics: 主题数量
        """
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000)
        self.lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        self.n_topics = n_topics
        self.fitted = False
        
    def fit(self, texts):
        """
        拟合文本特征挖掘模型
        
        Args:
            texts: 文本列表
        """
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        self.lda_model.fit(self.tfidf_matrix)
        self.fitted = True
        
    def extract_features(self, text):
        """
        提取文本特征
        
        Args:
            text: 输入文本
            
        Returns:
            特征向量
        """
        if not self.fitted:
            raise ValueError("模型尚未拟合，请先调用fit方法")
            
        tfidf_vector = self.tfidf_vectorizer.transform([text])
        
        # TF-IDF特征
        tfidf_features = tfidf_vector.toarray().flatten()
        
        # 主题特征
        topic_features = self.lda_model.transform(tfidf_vector).flatten()
        
        # 情感特征（简化版）
        sentiment_score = 1.0 if "开心" in text or "不错" in text else -1.0 if "难过" in text or "无助" in text else 0.0
        
        # 整合所有特征
        all_features = np.concatenate([
            tfidf_features[:20],  # 取前20个TF-IDF特征
            topic_features,
            np.array([sentiment_score])
        ])
        
        return all_features 