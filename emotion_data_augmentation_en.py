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

# Download required NLTK resources
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('punkt')

class EmotionDataAugmentation:
    """Emotion data augmentation class for multi-label classification"""
    
    def __init__(self, random_state=42):
        """
        Initialize the data augmenter
        
        Args:
            random_state: Random seed to ensure reproducibility
        """
        self.random_state = random_state
        random.seed(random_state)
        np.random.seed(random_state)
        
        # Load Chinese BERT model for synonym replacement and sentence embeddings
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.bert_model = BertModel.from_pretrained('bert-base-chinese')
        
        # Emotion categories
        self.emotion_categories = [
            "sadness", "hopelessness", "worthlessness", "loneliness", "self_hate", 
            "emptiness", "lack_of_energy", "anger", "suicidal_intent", "cognitive_dysfunction"
        ]
        
    def load_data(self, file_path):
        """
        Load emotion dataset
        
        Args:
            file_path: Path to CSV file containing text and emotion labels
            
        Returns:
            Loaded dataframe
        """
        df = pd.read_csv(file_path)
        print(f"Original dataset size: {len(df)} records")
        return df
    
    def analyze_label_distribution(self, df, label_column='feelings'):
        """
        Analyze label distribution
        
        Args:
            df: Dataframe
            label_column: Label column name
            
        Returns:
            Label distribution statistics
        """
        # Calculate occurrence count for each emotion category
        emotion_counts = {}
        
        for emotions_str in df[label_column]:
            emotions = str(emotions_str).split(',')
            for emotion in emotions:
                emotion = emotion.strip()
                if emotion:
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Convert to DataFrame for better viewing
        distribution_df = pd.DataFrame({
            'Emotion Category': list(emotion_counts.keys()),
            'Occurrence Count': list(emotion_counts.values())
        }).sort_values('Occurrence Count', ascending=False)
        
        print("Label distribution:")
        print(distribution_df)
        
        return distribution_df
    
    def _synonym_replacement(self, text, n=2):
        """
        Synonym replacement data augmentation
        
        Args:
            text: Input text
            n: Number of words to replace
            
        Returns:
            Augmented text
        """
        words = list(jieba.cut(text))
        
        # If text is too short, reduce replacement count
        n = min(n, max(1, len(words) // 3))
        
        # Randomly select n words to replace
        replace_indices = random.sample(range(len(words)), n)
        
        for i in replace_indices:
            word = words[i]
            # Use BERT model to get similar words
            if len(word) > 1:  # Only replace words longer than 1 character
                try:
                    # Use BERT to get context-sensitive synonyms
                    inputs = self.tokenizer(word, return_tensors="pt")
                    with torch.no_grad():
                        outputs = self.bert_model(**inputs)
                    
                    # Simple synonym replacement strategy - use a more complex method in actual projects
                    synonyms = ["similar_word1", "similar_word2", "similar_expression"]  # Should use real synonym dictionary
                    if synonyms:
                        words[i] = random.choice(synonyms)
                except:
                    continue
        
        return ''.join(words)
    
    def _random_insertion(self, text, n=2):
        """
        Random insertion of emotion-related words
        
        Args:
            text: Input text
            n: Number of words to insert
            
        Returns:
            Augmented text
        """
        words = list(jieba.cut(text))
        
        # Emotion-related word dictionary - should be expanded in actual applications
        emotion_words = {
            "sadness": ["sad", "heartbroken", "sorrowful", "melancholy", "grieved", "heartache"],
            "hopelessness": ["hopeless", "despairing", "desperate", "helpless", "lost"],
            "worthlessness": ["worthless", "useless", "good-for-nothing", "insignificant", "meaningless"],
            "loneliness": ["lonely", "alone", "isolated", "solitary", "abandoned"],
            "self_hate": ["self-hatred", "self-loathing", "self-disgust", "self-blame", "self-contempt"],
            "emptiness": ["empty", "hollow", "void", "meaningless", "numb"],
            "lack_of_energy": ["tired", "exhausted", "fatigued", "drained", "lethargic"],
            "anger": ["angry", "furious", "enraged", "irritated", "indignant"],
            "suicidal_intent": ["suicidal", "end it all", "death wish", "self-harm", "escape"],
            "cognitive_dysfunction": ["forgetful", "confused", "unfocused", "disoriented", "brain fog"]
        }
        
        # Randomly select an emotion category and corresponding words
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
        Random word position swapping
        
        Args:
            text: Input text
            n: Number of swaps to perform
            
        Returns:
            Augmented text
        """
        words = list(jieba.cut(text))
        
        if len(words) <= 1:
            return text
            
        n = min(n, len(words) // 2)  # Ensure not too many swaps
        
        for _ in range(n):
            i, j = random.sample(range(len(words)), 2)
            words[i], words[j] = words[j], words[i]
            
        return ''.join(words)
    
    def _back_translation(self, text):
        """
        Back-translation augmentation
        Note: This function requires external translation API, this is just a framework example
        
        Args:
            text: Input text
            
        Returns:
            Back-translated text
        """
        # Should call translation API for Chinese-English-Chinese back-translation
        # For simplification, just return original text
        try:
            # Chinese -> English -> Chinese back-translation process
            # translated_to_english = translate_api(text, source='zh', target='en')
            # back_to_chinese = translate_api(translated_to_english, source='en', target='zh')
            # return back_to_chinese
            return text + "(back-translated version)"  # Example
        except:
            return text
    
    def augment_minority_class(self, df, label_column='feelings', text_column='text', 
                              min_samples=30, methods=None):
        """
        Augment minority classes
        
        Args:
            df: Dataframe
            label_column: Label column name
            text_column: Text column name
            min_samples: Minimum sample count for each category
            methods: List of augmentation methods to use
            
        Returns:
            Augmented dataframe
        """
        if methods is None:
            methods = ['synonym', 'insertion', 'swap', 'backtrans']
        
        # Count occurrences of each emotion label
        emotion_samples = {emotion: [] for emotion in self.emotion_categories}
        
        # Group data by emotion
        for idx, row in df.iterrows():
            emotions = str(row[label_column]).split(',')
            emotions = [e.strip() for e in emotions if e.strip()]
            
            for emotion in emotions:
                if emotion in self.emotion_categories:
                    emotion_samples[emotion].append(idx)
        
        # Store new augmented samples
        augmented_samples = []
        
        # Augment each minority class emotion
        for emotion, indices in emotion_samples.items():
            current_count = len(indices)
            print(f"Emotion '{emotion}' original sample count: {current_count}")
            
            if current_count < min_samples:
                # Number of samples to add
                samples_to_add = min_samples - current_count
                
                # Randomly select from existing samples for augmentation
                selected_indices = random.choices(indices, k=samples_to_add)
                
                for idx in selected_indices:
                    original_text = df.loc[idx, text_column]
                    original_emotions = df.loc[idx, label_column]
                    
                    # Randomly select an augmentation method
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
        
        # Create dataframe of augmented samples
        if augmented_samples:
            augmented_df = pd.DataFrame(augmented_samples)
            result_df = pd.concat([df, augmented_df], ignore_index=True)
            print(f"Augmented dataset size: {len(result_df)} records")
            return result_df
        else:
            print("No data augmentation performed")
            return df
    
    def generate_balanced_samples(self, df, label_column='feelings', text_column='text'):
        """
        Generate balanced training set, combining oversampling and data augmentation
        
        Args:
            df: Dataframe
            label_column: Label column name
            text_column: Text column name
            
        Returns:
            Balanced dataframe
        """
        # Perform data augmentation on the dataset
        augmented_df = self.augment_minority_class(df, label_column, text_column)
        
        # Count samples for each emotion after augmentation
        emotion_counts = {}
        for idx, row in augmented_df.iterrows():
            emotions = str(row[label_column]).split(',')
            emotions = [e.strip() for e in emotions if e.strip()]
            
            for emotion in emotions:
                if emotion in self.emotion_categories:
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Find the sample count of the majority class
        max_samples = max(emotion_counts.values()) if emotion_counts else 0
        
        # Oversample or undersample each emotion category to balance samples
        balanced_samples = []
        
        for emotion in self.emotion_categories:
            # Get indices of samples containing this emotion
            emotion_indices = []
            for idx, row in augmented_df.iterrows():
                emotions = str(row[label_column]).split(',')
                if emotion in [e.strip() for e in emotions]:
                    emotion_indices.append(idx)
            
            # Skip if no samples for this emotion
            if not emotion_indices:
                continue
                
            # Use oversampling methods to balance samples
            if len(emotion_indices) < max_samples:
                # Oversample minority class
                sampled_indices = resample(
                    emotion_indices,
                    replace=True,
                    n_samples=max_samples,
                    random_state=self.random_state
                )
            else:
                # Undersample majority class
                sampled_indices = resample(
                    emotion_indices,
                    replace=False,
                    n_samples=max_samples,
                    random_state=self.random_state
                )
            
            for idx in sampled_indices:
                sample = augmented_df.loc[idx].copy()
                balanced_samples.append(sample)
        
        # Create balanced dataframe
        balanced_df = pd.DataFrame(balanced_samples)
        print(f"Balanced dataset size: {len(balanced_df)} records")
        
        return balanced_df
    
    def save_augmented_data(self, df, output_file):
        """
        Save augmented data
        
        Args:
            df: Dataframe
            output_file: Output file path
        """
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"Augmented data saved to: {output_file}")


# Usage example
if __name__ == "__main__":
    # Create data augmenter
    augmenter = EmotionDataAugmentation()
    
    # Load original data
    data = augmenter.load_data("emotions_dataset.csv")
    
    # Analyze label distribution
    augmenter.analyze_label_distribution(data)
    
    # Augment minority classes
    augmented_data = augmenter.augment_minority_class(
        data, 
        label_column='feelings',
        text_column='text',
        min_samples=30
    )
    
    # Generate balanced dataset
    balanced_data = augmenter.generate_balanced_samples(
        augmented_data, 
        label_column='feelings',
        text_column='text'
    )
    
    # Save augmented data
    augmenter.save_augmented_data(balanced_data, "augmented_emotions_dataset.csv") 