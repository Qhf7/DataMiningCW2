# 抑郁情绪多标签分类系统

基于深度学习和自然语言处理技术的抑郁情绪多标签分类系统，能够从文本中识别8种不同的抑郁情绪表现。

## 情绪类别

系统可以识别以下8种抑郁相关情绪：

1. 愤怒 (anger)
2. 认知功能障碍/遗忘 (brain dysfunction/forget)
3. 空虚感 (emptiness)
4. 绝望感 (hopelessness)
5. 孤独感 (loneliness)
6. 悲伤 (sadness)
7. 自杀意念 (suicide intent)
8. 无价值感 (worthlessness)

## 系统架构

系统主要由以下几部分组成：

1. **增强型BERT模型**：基于预训练的BERT模型，添加了注意力机制和情感特征融合层
2. **文本增强模块**：对训练数据进行自动增强，提高模型对少见情绪类别的识别能力
3. **评估与可视化工具**：分析模型性能并生成可视化结果

## 核心模型架构

### BERT 模型详解

我们的系统以BERT（Bidirectional Encoder Representations from Transformers）为基础，这是一种预训练的双向Transformer编码器架构。我们主要采用以下BERT变体：

- **bert-base-chinese**: 针对中文文本优化的BERT模型
- **bert-base-uncased**: 用于英文文本处理的基础BERT模型

BERT的优势在于：

- **双向上下文理解**：同时考虑左右上下文信息，理解语义更准确
- **预训练-微调范式**：充分利用大规模无标注数据的预训练表示
- **深层语义编码**：Transformer的多层架构能够捕获不同粒度的语言信息

我们在BERT基础上做了如下改进：

1. **选择性层冻结**：冻结底部6层参数，仅微调上层，减轻过拟合
2. **特征投影层**：添加线性投影层，将BERT输出从768维减少到512维
3. **多头自注意力层**：增强对不同情感表达的关注
4. **情感特征融合**：融合BERT表示与传统情感分析特征

### 文本特征提取模块

TextFeatureExtractor类负责从文本中提取深层语义特征：

```python
class TextFeatureExtractor(nn.Module):
    def __init__(self, model_type='bert', freeze_bert_layers=6):
        super(TextFeatureExtractor, self).__init__()
        # 选择预训练模型
        if model_type == 'bert':
            self.bert = BertModel.from_pretrained('bert-base-chinese')
            self.hidden_size = self.bert.config.hidden_size
        # ...其他模型类型
        
        # 选择性冻结底层
        if freeze_bert_layers > 0:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
            # ...冻结指定层数
            
        # 特征投影层
        self.feature_projection = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.LayerNorm(512),
            nn.Dropout(0.2),
            nn.ReLU()
        )
```

### 特征融合与分类层

我们采用多特征融合策略，结合深度特征和传统文本挖掘特征：

1. **自注意力机制**：捕获文本内部的长距离依赖关系
2. **情感词汇特征**：整合TF-IDF和主题模型提取的特征
3. **多任务学习**：主任务（情绪分类）与辅助任务（情感极性预测）

## 数据清洗与规范化

文本处理流水线包含以下步骤：

### 文本清洗

```python
def clean_text(self, text):
    """文本清洗"""
    if not isinstance(text, str):
        return ""
        
    # 移除多余空格
    text = re.sub(r'\s+', ' ', text)
    
    # 移除特殊字符，仅保留中文、英文、数字和基本标点
    text = re.sub(r'[^\w\s\u4e00-\u9fff。，！？、；：""''（）【】《》]', '', text)
    
    return text.strip()
```

### 文本正则化

1. **字符级标准化**：统一全角/半角符号、简繁体转换
2. **分词与停用词过滤**：使用NLTK分词，移除停用词
3. **标点符号标准化**：统一标点符号格式
4. **特征抽取**：提取统计特征，包括文本长度、词频、标点符号计数等

## 文本增强技术

为解决类别不平衡问题和提高模型泛化能力，我们使用多种文本增强策略：

### 1. 同义词替换增强

```python
def synonym_replacement(self, text, n=2):
    """同义词替换增强"""
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
    
    return ''.join(new_words)
```

### 2. 回译技术（以中文为例）

通过将原始文本翻译成中间语言（如英语）再翻译回原语言，产生语义相似但表达不同的文本变体：

```python
def back_translation(self, text):
    """回译（模拟）"""
    # 抑郁相关文本的回译变体
    depression_translations = {
        "我感到非常难过和无助": "我感到极度悲伤和绝望。似乎没有任何事物能够让我感到一丝希望。",
        "我最近总是感到疲惫": "近期我一直感到精疲力竭，仿佛被抽干了所有精力。",
        # ...更多模板
    }
    
    # 查找匹配的文本进行替换
    for source, target in depression_translations.items():
        if source in text:
            return text.replace(source, target)
            
    return text
```

### 3. 随机插入情感词汇（以中文为例）

```python
def random_insertion(self, text, n=1):
    """随机插入情感词汇"""
    words = self.preprocessor.segment(text, remove_stop_words=False)
    
    # 情感词汇库
    depression_words = ["孤独", "疲惫", "失落", "压抑", "痛苦", "悲伤", "绝望"]
    positive_words = ["轻松", "舒适", "满足", "温暖", "安心", "开心", "喜悦"]
    
    # 根据文本情感倾向选择词库
    word_list = depression_words if any(word in text for word in ["难过", "无助", "痛苦"]) else positive_words
        
    # 执行插入
    for _ in range(n):
        word_to_insert = np.random.choice(word_list)
        insert_pos = np.random.randint(0, len(words) + 1)
        words.insert(insert_pos, word_to_insert)
        
    return ''.join(words)
```

### 4. 随机词语交换

```python
def random_swap(self, text, n=1):
    """随机交换词序"""
    words = self.preprocessor.segment(text, remove_stop_words=False)
    
    if len(words) < 2:
        return text
        
    # 执行交换
    for _ in range(n):
        idx1, idx2 = np.random.choice(len(words), 2, replace=False)
        words[idx1], words[idx2] = words[idx2], words[idx1]
        
    return ''.join(words)
```

### 5. EDA集成增强

我们采用Easy Data Augmentation (EDA)策略，综合多种增强方法：

```python
def eda_augment(self, text, alpha=0.3):
    """整合多种增强方法"""
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
```

## 文本挖掘与特征工程

除深度学习特征外，我们还整合了传统文本挖掘技术：

### 1. TF-IDF特征提取

使用TF-IDF (Term Frequency-Inverse Document Frequency)算法提取关键词特征，捕获文档特异性术语。

### 2. 主题模型 (LDA)

使用Latent Dirichlet Allocation模型提取潜在主题特征：

```python
def extract_features(self, text):
    """提取文本特征"""
    tfidf_vector = self.tfidf_vectorizer.transform([text])
    
    # TF-IDF特征
    tfidf_features = tfidf_vector.toarray().flatten()
    
    # 主题特征
    topic_features = self.lda_model.transform(tfidf_vector).flatten()
    
    # 情感特征（简化版）
    sentiment_score = self.calculate_sentiment_score(text)
    
    # 整合所有特征
    all_features = np.concatenate([
        tfidf_features[:20],  # 取前20个TF-IDF特征
        topic_features,
        np.array([sentiment_score])
    ])
    
    return all_features
```

### 3. 统计特征分析

提取多种统计特征，捕捉文本风格和情感特征：

- 文本长度特征（字符数、词数、平均词长）
- 词频特征（唯一词数、词汇多样性）
- 标点符号特征（逗号、句号、问号、感叹号计数）
- 情感词汇计数和情感比率

## 安装与环境配置

### 环境要求

- Python 3.7+
- PyTorch 1.9+
- Transformers 4.5+
- scikit-learn
- matplotlib
- seaborn
- pandas
- numpy

### 安装依赖

```bash
pip install torch torchvision transformers sklearn matplotlib seaborn pandas numpy jieba
```

## 使用方法

### 数据准备

系统使用JSON格式的数据，每行一个样本，格式如下：

```json
{"id": "sample_id", "title": "标题", "post": "内容", "text": "标题 ### 内容", "upvotes": 102, "date": "2022-12-19 19:50:52", "emotions": ["emptiness", "hopelessness"], "label_id": 110000}
```

数据集需要分为三个文件：
- `train.json`: 训练集
- `val.json`: 验证集
- `test.json`: 测试集

### 模型训练

使用我们提供的脚本运行训练：

```bash
# 使模型训练脚本可执行
chmod +x run_depression_emotion_classifier.sh

# 运行标准训练
./run_depression_emotion_classifier.sh

# 使用Focal Loss进行训练
./run_depression_emotion_classifier.sh focal
```

或者直接使用Python命令：

```bash
# 训练模式
python depression_emotion_classifier.py --mode train --model_name bert-base-cased --train_path Dataset/train.json --val_path Dataset/val.json --epochs 4 --batch_size 8 --max_length 256

# 测试模式
python depression_emotion_classifier.py --mode test --model_name bert-base-cased --model_path best_depression_emotion_model.bin --test_path Dataset/test.json
```

### 结果分析与可视化

训练完成后，可以使用我们的可视化工具分析模型性能：

```bash
python visualize_results.py --results_file outputs/test_results.json --test_path Dataset/test.json --output_path outputs/analysis
```

这将生成以下分析结果：
- 模型整体性能概览
- 各情绪类别的F1分数
- 情绪共病模式分析
- 综合性能报告

## 高级配置

### 模型参数调整

可以通过修改以下参数来调整模型性能：

- `--learning_rate`: 学习率 (默认: 2e-5)
- `--batch_size`: 批次大小 (默认: 8)
- `--max_length`: 最大序列长度 (默认: 256)
- `--epochs`: 训练轮数 (默认: 4)
- `--use_focal_loss`: 使用Focal Loss而非标准BCE损失函数

### 自定义预训练模型

可以使用不同的预训练模型作为基础：

```bash
python depression_emotion_classifier.py --mode train --model_name bert-large-cased --train_path Dataset/train.json --val_path Dataset/val.json
```

支持的模型包括：
- bert-base-cased
- bert-large-cased
- roberta-base
- xlm-roberta-base
- distilbert-base-cased
- albert-base-v2

## 性能示例

在测试集上的性能：

| 指标 | 分数 |
|------|------|
| F1 (micro) | ~0.78 |
| F1 (macro) | ~0.69 |
| 精确度 | ~0.82 |
| 召回率 | ~0.75 |

不同情绪的F1分数:
- 悲伤 (sadness): ~0.87
- 孤独感 (loneliness): ~0.82 
- 绝望感 (hopelessness): ~0.79
- 愤怒 (anger): ~0.75
- 无价值感 (worthlessness): ~0.68
- 自杀意念 (suicide intent): ~0.60
- 空虚感 (emptiness): ~0.56 
- 认知功能障碍 (brain dysfunction): ~0.43
