
---

# Multi-label Depression Emotion Classification System

A multi-label depression emotion classification system based on deep learning and natural language processing techniques, capable of identifying eight different manifestations of depressive emotions from text.

## Emotion Categories

The system can identify the following eight depression-related emotions:

1. Anger
2. Cognitive dysfunction/forgetfulness
3. Emptiness
4. Hopelessness
5. Loneliness
6. Sadness
7. Suicide intent
8. Worthlessness

## System Architecture

The system mainly consists of the following components:

1. **Enhanced BERT Model**: Based on the pre-trained BERT model, with an added attention mechanism and emotional feature fusion layer.
2. **Text Augmentation Module**: Automatically augments training data to improve recognition of rare emotion categories.
3. **Evaluation and Visualization Tools**: Analyze model performance and generate visual results.

## Core Model Architecture

### BERT Model Details

Our system is based on BERT (Bidirectional Encoder Representations from Transformers), a pre-trained bidirectional transformer encoder architecture. We mainly use the following BERT variants:

- **bert-base-chinese**: BERT model optimized for Chinese text
- **bert-base-uncased**: Basic BERT model for English text processing

Advantages of BERT:

- **Bidirectional Context Understanding**: Considers both left and right context for more accurate semantic understanding.
- **Pretraining-Fine-tuning Paradigm**: Makes full use of large-scale unlabeled data for pre-trained representations.
- **Deep Semantic Encoding**: The multi-layer architecture of Transformer can capture language information at different granularities.

Our improvements on BERT include:

1. **Selective Layer Freezing**: Freeze the bottom 6 layers and only fine-tune the upper layers to reduce overfitting.
2. **Feature Projection Layer**: Add a linear projection layer to reduce BERT output from 768 to 512 dimensions.
3. **Multi-head Self-attention Layer**: Enhance attention to different emotional expressions.
4. **Emotional Feature Fusion**: Fuse BERT representations with traditional sentiment analysis features.

### Text Feature Extraction Module

The `TextFeatureExtractor` class is responsible for extracting deep semantic features from text:

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
            # ...freeze specified layers
            
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

1. **Self-attention Mechanism**: Captures long-distance dependencies within the text.
2. **Sentiment Lexicon Features**: Integrates features extracted by TF-IDF and topic models.
3. **Multi-task Learning**: Main task (emotion classification) and auxiliary task (sentiment polarity prediction).

## Data Cleaning and Normalization

The text processing pipeline includes the following steps:

### Text Cleaning

```python
def clean_text(self, text):
    """Text cleaning"""
    if not isinstance(text, str):
        return ""
        
    # Remove excess spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters, only keep Chinese, English, numbers, and basic punctuation
    text = re.sub(r'[^\w\s\u4e00-\u9fff。，！？、；：""''（）【】《》]', '', text)
    
    return text.strip()
```

### Text Normalization

1. **Character-level Standardization**: Unify full-width/half-width symbols, simplify/traditional conversion.
2. **Word Segmentation and Stopword Removal**: Use NLTK for tokenization and remove stopwords.
3. **Punctuation Normalization**: Unify punctuation formats.
4. **Feature Extraction**: Extract statistical features, including text length, word frequency, punctuation count, etc.

## Text Augmentation Techniques

To address class imbalance and improve model generalization, we use multiple text augmentation strategies:

### 1. Synonym Replacement Augmentation

```python
def synonym_replacement(self, text, n=2):
    """Synonym replacement augmentation"""
    words = self.preprocessor.segment(text, remove_stop_words=False)
    new_words = words.copy()
    
    # Record positions that can be replaced
    replaced_positions = []
    
    # Try to replace n words
    replacement_count = 0
    for i, word in enumerate(words):
        if word in self.synonyms and i not in replaced_positions and replacement_count < n:
            synonyms = self.synonyms[word]
            if synonyms:
                # Randomly choose a synonym for replacement
                replacement = np.random.choice(synonyms)
                new_words[i] = replacement
                replaced_positions.append(i)
                replacement_count += 1
    
    return ''.join(new_words)
```

### 2. Back Translation (Chinese Example)

Generate semantically similar but differently expressed text variants by translating the original text into an intermediate language (such as English) and then back to the original language:

```python
def back_translation(self, text):
    """Back translation (simulation)"""
    # Back-translation variants for depression-related text
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

### 3. Random Insertion of Sentiment Words (Chinese Example)

```python
def random_insertion(self, text, n=1):
    """Randomly insert sentiment words"""
    words = self.preprocessor.segment(text, remove_stop_words=False)
    
    # Sentiment word bank
    depression_words = ["loneliness", "fatigue", "loss", "oppression", "pain", "sadness", "despair"]
    positive_words = ["relaxed", "comfortable", "satisfied", "warm", "secure", "happy", "joy"]
    
    # Select word bank based on emotional tendency of the text
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
    """Randomly swap word order"""
    words = self.preprocessor.segment(text, remove_stop_words=False)
    
    if len(words) < 2:
        return text
        
    # Perform swaps
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
    
    # Apply each augmentation method with a certain probability
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

TF-IDF (Term Frequency-Inverse Document Frequency) algorithm is used to extract keyword features and capture document-specific terms.

### 2. Topic Modeling (LDA)

Latent Dirichlet Allocation (LDA) model is used to extract latent topic features:

```python
def extract_features(self, text):
    """Extract text features"""
    tfidf_vector = self.tfidf_vectorizer.transform([text])
    
    # TF-IDF features
    tfidf_features = tfidf_vector.toarray().flatten()
    
    # Topic features
    topic_features = self.lda_model.transform(tfidf_vector).flatten()
    
    # Sentiment feature (simplified)
    sentiment_score = self.calculate_sentiment_score(text)
    
    # Integrate all features
    all_features = np.concatenate([
        tfidf_features[:20],  # Take first 20 TF-IDF features
        topic_features,
        np.array([sentiment_score])
    ])
    
    return all_features
```

### 3. Statistical Feature Analysis

Extract a variety of statistical features to capture text style and emotional features:

- Text length features (number of characters, words, average word length)
- Word frequency features (number of unique words, vocabulary diversity)
- Punctuation features (count of commas, periods, question marks, exclamation marks)
- Sentiment word count and sentiment ratio

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

## Usage

### Data Preparation

The system uses JSON-formatted data, one sample per line, in the following format:

```json
{"id": "sample_id", "title": "Title", "post": "Content", "text": "Title ### Content", "upvotes": 102, "date": "2022-12-19 19:50:52", "emotions": ["emptiness", "hopelessness"], "label_id": 110000}
```

The dataset needs to be divided into three files:
- `train.json`: Training set
- `val.json`: Validation set
- `test.json`: Test set

### Model Training

Run the training script we provided:

```bash
# Make the training script executable
chmod +x run_depression_emotion_classifier.sh

# Run standard training
./run_depression_emotion_classifier.sh

# Train with Focal Loss
./run_depression_emotion_classifier.sh focal
```

Or use a Python command directly:

```bash
# Training mode
python depression_emotion_classifier.py --mode train --model_name bert-base-cased --train_path Dataset/train.json --val_path Dataset/val.json --epochs 4 --batch_size 8 --max_length 256

# Testing mode
python depression_emotion_classifier.py --mode test --model_name bert-base-cased --model_path best_depression_emotion_model.bin --test_path Dataset/test.json
```

### Results Analysis and Visualization

After training, you can use our visualization tool to analyze model performance:

```bash
python visualize_results.py --results_file outputs/test_results.json --test_path Dataset/test.json --output_path outputs/analysis
```

This will generate the following analysis results:
- Overview of model performance
- F1 scores for each emotion category
- Analysis of emotion comorbidity patterns
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

You can use different pre-trained models as the base:

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

## Example Performance

Performance on the test set:

| Metric         | Score |
|----------------|-------|
| F1 (micro)     | ~0.78 |
| F1 (macro)     | ~0.69 |
| Accuracy       | ~0.82 |
| Recall         | ~0.75 |

F1 scores for different emotions:
- Sadness: ~0.87
- Loneliness: ~0.82 
- Hopelessness: ~0.79
- Anger: ~0.75
- Worthlessness: ~0.68
- Suicide intent: ~0.60
- Emptiness: ~0.56 
- Cognitive dysfunction: ~0.43

---
