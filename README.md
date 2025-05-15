---

# Multi-Label Depression Emotion Classification System

A multi-label depression emotion classification system based on deep learning and natural language processing technologies, capable of identifying 8 different depression-related emotions from text.

## Emotion Categories

The system can identify the following 8 depression-related emotions:

1. Anger
2. Brain dysfunction/forgetfulness
3. Emptiness
4. Hopelessness
5. Loneliness
6. Sadness
7. Suicide intent
8. Worthlessness

## System Architecture

The system mainly consists of the following parts:

1. **Enhanced BERT Model**: Based on a pre-trained BERT model, with added attention mechanism and emotion feature fusion layer
2. **Text Augmentation Module**: Automatically augments training data to improve the recognition ability for rare emotion categories
3. **Evaluation and Visualization Tools**: Analyze model performance and generate visualization results

## Core Model Architecture

### BERT Model Details

Our system is based on BERT (Bidirectional Encoder Representations from Transformers), a pre-trained bidirectional Transformer encoder architecture. We mainly use the following BERT variants:

- **bert-base-chinese**: BERT model optimized for Chinese text
- **bert-base-uncased**: Base BERT model for English text processing

The advantages of BERT include:

- **Bidirectional Context Understanding**: Considers both left and right context for more accurate semantic understanding
- **Pretrain-Finetune Paradigm**: Makes full use of pre-trained representations from large-scale unlabeled data
- **Deep Semantic Encoding**: The multi-layer Transformer architecture can capture language information at different granularities

Our improvements on top of BERT include:

1. **Selective Layer Freezing**: Freeze the bottom 6 layers' parameters and only finetune the upper layers to reduce overfitting
2. **Feature Projection Layer**: Add a linear projection layer to reduce BERT's 768-dim output to 512-dim
3. **Multi-head Self-Attention Layer**: Enhance attention to different emotional expressions
4. **Emotion Feature Fusion**: Fuse BERT representations with traditional sentiment analysis features

### Text Feature Extraction Module

The TextFeatureExtractor class is responsible for extracting deep semantic features from text:

```python
class TextFeatureExtractor(nn.Module):
    def __init__(self, model_type='bert', freeze_bert_layers=6):
        super(TextFeatureExtractor, self).__init__()
        # Choose pre-trained model
        if model_type == 'bert':
            self.bert = BertModel.from_pretrained('bert-base-chinese')
            self.hidden_size = self.bert.config.hidden_size
        # ...other model types
        
        # Selective layer freezing
        if freeze_bert_layers > 0:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
            # ...freeze specified number of layers
            
        # Feature projection layer
        self.feature_projection = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.LayerNorm(512),
            nn.Dropout(0.2),
            nn.ReLU()
        )
```

### Feature Fusion and Classification Layer

We use a multi-feature fusion strategy, combining deep features and traditional text mining features:

1. **Self-Attention Mechanism**: Capture long-distance dependencies within the text
2. **Sentiment Lexicon Features**: Integrate features extracted by TF-IDF and topic models
3. **Multi-task Learning**: Main task (emotion classification) and auxiliary task (sentiment polarity prediction)

## Data Cleaning and Normalization

The text processing pipeline includes the following steps:

### Text Cleaning

```python
def clean_text(self, text):
    """Text cleaning"""
    if not isinstance(text, str):
        return ""
        
    # Remove redundant spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters, keep only Chinese, English, numbers, and basic punctuation
    text = re.sub(r'[^\w\s\u4e00-\u9fff。，！？、；：""''（）【】《》]', '', text)
    
    return text.strip()
```

### Text Normalization

1. **Character-level Standardization**: Unify full-width/half-width symbols, convert between simplified and traditional Chinese
2. **Word Segmentation and Stopword Filtering**: Use jieba for word segmentation and remove stopwords
3. **Punctuation Standardization**: Unify punctuation formats
4. **Feature Extraction**: Extract statistical features including text length, word frequency, punctuation count, etc.

## Text Augmentation Techniques

To address class imbalance and improve the model's generalization ability, we use a variety of text augmentation strategies:

### 1. Synonym Replacement Augmentation

```python
def synonym_replacement(self, text, n=2):
    """Synonym replacement augmentation"""
    words = self.preprocessor.segment(text, remove_stop_words=False)
    new_words = words.copy()
    
    # Record positions for replacement
    replaced_positions = []
    
    # Try to replace n words
    replacement_count = 0
    for i, word in enumerate(words):
        if word in self.synonyms and i not in replaced_positions and replacement_count < n:
            synonyms = self.synonyms[word]
            if synonyms:
                # Randomly select a synonym for replacement
                replacement = np.random.choice(synonyms)
                new_words[i] = replacement
                replaced_positions.append(i)
                replacement_count += 1
    
    return ''.join(new_words)
```

### 2. Back-Translation Technique

By translating the original text into an intermediate language (such as English) and then back to the original language, generate semantically similar but differently expressed text variants:

```python
def back_translation(self, text):
    """Back-translation (simulation)"""
    # Depression-related text back-translation variants
    depression_translations = {
        "我感到非常难过和无助": "我感到极度悲伤和绝望。似乎没有任何事物能够让我感到一丝希望。",
        "我最近总是感到疲惫": "近期我一直感到精疲力竭，仿佛被抽干了所有精力。",
        # ...more templates
    }
    
    # Find matching text for replacement
    for source, target in depression_translations.items():
        if source in text:
            return text.replace(source, target)
            
    return text
```

### 3. Random Insertion of Sentiment Words

```python
def random_insertion(self, text, n=1):
    """Random insertion of sentiment words"""
    words = self.preprocessor.segment(text, remove_stop_words=False)
    
    # Sentiment word bank
    depression_words = ["loneliness", "fatigue", "loss", "oppression", "pain", "sadness", "hopelessness"]
    positive_words = ["relax", "comfort", "satisfaction", "warmth", "peace", "happiness", "joy"]
    
    # Choose word bank based on sentiment tendency
    word_list = depression_words if any(word in text for word in ["sad", "helpless", "pain"]) else positive_words
        
    # Perform insertion
    for _ in range(n):
        word_to_insert = np.random.choice(word_list)
        insert_pos = np.random.randint(0, len(words) + 1)
        words.insert(insert_pos, word_to_insert)
        
    return ''.join(words)
```

### 4. Random Word Swap

```python
def random_swap(self, text, n=1):
    """Random word order swapping"""
    words = self.preprocessor.segment(text, remove_stop_words=False)
    
    if len(words) < 2:
        return text
        
    # Perform swapping
    for _ in range(n):
        idx1, idx2 = np.random.choice(len(words), 2, replace=False)
        words[idx1], words[idx2] = words[idx2], words[idx1]
        
    return ''.join(words)
```

### 5. EDA Integrated Augmentation

We use Easy Data Augmentation (EDA) strategy, integrating multiple augmentation methods:

```python
def eda_augment(self, text, alpha=0.3):
    """Integrate multiple augmentation methods"""
    augmented_text = text
    
    # Apply each augmentation with a certain probability
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

## Text Mining and Feature Engineering

In addition to deep learning features, we also integrate traditional text mining techniques:

### 1. TF-IDF Feature Extraction

Use the TF-IDF (Term Frequency-Inverse Document Frequency) algorithm to extract keyword features and capture document-specific terms.

### 2. Topic Modeling (LDA)

Use Latent Dirichlet Allocation (LDA) to extract latent topic features:

```python
def extract_features(self, text):
    """Extract text features"""
    tfidf_vector = self.tfidf_vectorizer.transform([text])
    
    # TF-IDF features
    tfidf_features = tfidf_vector.toarray().flatten()
    
    # Topic features
    topic_features = self.lda_model.transform(tfidf_vector).flatten()
    
    # Sentiment features (simplified)
    sentiment_score = self.calculate_sentiment_score(text)
    
    # Integrate all features
    all_features = np.concatenate([
        tfidf_features[:20],  # Take the first 20 TF-IDF features
        topic_features,
        np.array([sentiment_score])
    ])
    
    return all_features
```

### 3. Statistical Feature Analysis

Extract various statistical features to capture text style and sentiment characteristics:

- Text length features (number of characters, words, average word length)
- Word frequency features (number of unique words, lexical diversity)
- Punctuation features (counts of commas, periods, question marks, exclamation marks)
- Sentiment word counts and sentiment ratios

## Installation and Environment Setup

### Environment Requirements

- Python 3.7+
- PyTorch 1.9+
- Transformers 4.5+
- scikit-learn
- matplotlib
- seaborn
- pandas
- numpy

### Install Dependencies

```bash
pip install torch torchvision transformers sklearn matplotlib seaborn pandas numpy jieba
```

## Usage Instructions

### Data Preparation

The system uses data in JSON format, with one sample per line, as shown below:

```json
{"id": "sample_id", "title": "Title", "post": "Content", "text": "Title ### Content", "upvotes": 102, "date": "2022-12-19 19:50:52", "emotions": ["emptiness", "hopelessness"], "label_id": 110000}
```

The dataset needs to be split into three files:
- `train.json`: Training set
- `val.json`: Validation set
- `test.json`: Test set

### Model Training

Use our provided script to run training:

```bash
# Make the model training script executable
chmod +x run_depression_emotion_classifier.sh

# Run standard training
./run_depression_emotion_classifier.sh

# Train with Focal Loss
./run_depression_emotion_classifier.sh focal
```

Or use Python command directly:

```bash
# Training mode
python depression_emotion_classifier.py --mode train --model_name bert-base-cased --train_path Dataset/train.json --val_path Dataset/val.json --epochs 4 --batch_size 8 --max_length 256

# Test mode
python depression_emotion_classifier.py --mode test --model_name bert-base-cased --model_path best_depression_emotion_model.bin --test_path Dataset/test.json
```

### Result Analysis and Visualization

After training is complete, you can use our visualization tool to analyze model performance:

```bash
python visualize_results.py --results_file outputs/test_results.json --test_path Dataset/test.json --output_path outputs/analysis
```

This will generate the following analysis results:
- Overall model performance overview
- F1 scores for each emotion category
- Comorbidity pattern analysis of emotions
- Comprehensive performance report

## Advanced Configuration

### Model Parameter Adjustment

You can adjust model performance by modifying the following parameters:

- `--learning_rate`: Learning rate (default: 2e-5)
- `--batch_size`: Batch size (default: 8)
- `--max_length`: Maximum sequence length (default: 256)
- `--epochs`: Number of training epochs (default: 4)
- `--use_focal_loss`: Use Focal Loss instead of standard BCE loss

### Custom Pre-trained Models

You can use different pre-trained models as the backbone:

```bash
python depression_emotion_classifier.py --mode train --model_name bert-large-cased --train_path Dataset/train.json --val_path Dataset/val.json
```

Supported models include:
- bert-base-cased
- bert-large-cased
- roberta-base
- xlm-roberta-base
- distilbert-base-cased
- albert-base-v2

## Performance Example

Performance on the test set:

| Metric    | Score |
|-----------|-------|
| F1 (micro)| ~0.78 |
| F1 (macro)| ~0.69 |
| Accuracy  | ~0.82 |
| Recall    | ~0.75 |

F1 scores for different emotions:
- Sadness: ~0.87
- Loneliness: ~0.82 
- Hopelessness: ~0.79
- Anger: ~0.75
- Worthlessness: ~0.68
- Suicide intent: ~0.60
- Emptiness: ~0.56 
- Brain dysfunction: ~0.43

---
