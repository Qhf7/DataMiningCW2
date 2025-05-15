import argparse
import json
import numpy as np
import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
import torch.nn.functional as F
import logging
import time
from datetime import datetime

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("depression_emotion_classifier.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 定义8种抑郁情绪
EMOTION_LIST = ['anger', 'brain dysfunction (forget)', 'emptiness', 'hopelessness', 
                'loneliness', 'sadness', 'suicide intent', 'worthlessness']

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"使用设备: {device}")

# 数据集类
class DepressionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

# 注意力机制类
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, hidden_states):
        # hidden_states: [batch_size, seq_len, hidden_size]
        attention_weights = self.attention(hidden_states)  # [batch_size, seq_len, 1]
        context_vector = torch.sum(attention_weights * hidden_states, dim=1)  # [batch_size, hidden_size]
        return context_vector, attention_weights

# 改进的情感分类器模型
class EnhancedDepressionEmotionClassifier(nn.Module):
    def __init__(self, n_classes, model_name='bert-base-cased', dropout=0.1):
        super(EnhancedDepressionEmotionClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.attention = AttentionLayer(self.bert.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # 情感特征融合层
        self.sentiment_feature = nn.Linear(self.bert.config.hidden_size, 64)
        self.content_feature = nn.Linear(self.bert.config.hidden_size, 64)
        
        # 最终分类层
        self.classifier = nn.Linear(self.bert.config.hidden_size + 128, n_classes)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_ids, attention_mask):
        # BERT输出
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True
        )
        
        # 使用CLS token表示
        pooled_output = outputs.pooler_output  # [batch_size, hidden_size]
        
        # 获取最后一层的隐藏状态
        last_hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # 应用注意力机制
        context_vector, attention_weights = self.attention(last_hidden_states)
        
        # 情感特征提取
        sentiment_feature = torch.tanh(self.sentiment_feature(pooled_output))
        content_feature = torch.tanh(self.content_feature(context_vector))
        
        # 特征融合
        combined_features = torch.cat([pooled_output, sentiment_feature, content_feature], dim=1)
        dropout_output = self.dropout(combined_features)
        
        # 分类
        logits = self.classifier(dropout_output)
        return self.sigmoid(logits), attention_weights

# 自定义损失函数：结合二元交叉熵和标签相关性
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, inputs, targets):
        # 二元交叉熵
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        # Focal Loss项
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        return focal_loss.mean()

# 加载数据
def load_data(data_path):
    texts = []
    labels = []
    
    logger.info(f"加载数据集: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                texts.append(data['text'])
                
                # 将标签ID转换为二进制标签
                label_id = str(data['label_id'])
                binary_label = [0] * len(EMOTION_LIST)
                
                # 将label_id转换为二进制列表
                for i, digit in enumerate(label_id.zfill(len(EMOTION_LIST))[-len(EMOTION_LIST):]):
                    binary_label[i] = int(digit)
                
                labels.append(binary_label)
            except Exception as e:
                logger.error(f"处理数据时出错: {e}")
                continue
    
    logger.info(f"加载了 {len(texts)} 条数据")
    return texts, labels

# 文本增强函数
def augment_text(text, label, augment_prob=0.2):
    """简单的文本增强方法：随机删除词"""
    import random
    
    if random.random() > augment_prob:
        return text
    
    words = text.split()
    if len(words) <= 10:  # 对于太短的文本不做增强
        return text
    
    # 随机删除10%的词
    n_delete = max(1, int(len(words) * 0.1))
    indices_to_delete = random.sample(range(len(words)), n_delete)
    augmented_words = [word for i, word in enumerate(words) if i not in indices_to_delete]
    
    return ' '.join(augmented_words)

# 创建数据加载器，支持数据增强
def create_data_loader(texts, labels, tokenizer, max_len, batch_size, is_train=False):
    if is_train:
        # 对训练数据进行增强
        augmented_texts = []
        augmented_labels = []
        
        # 对每个具有抑郁标签的样本进行增强
        for text, label in zip(texts, labels):
            augmented_texts.append(text)
            augmented_labels.append(label)
            
            # 如果样本包含至少一个抑郁情绪标签，则进行增强
            if sum(label) > 0:
                aug_text = augment_text(text, label)
                if aug_text != text:  # 确保增强后文本与原文本不同
                    augmented_texts.append(aug_text)
                    augmented_labels.append(label)
        
        logger.info(f"原始数据: {len(texts)}条, 增强后: {len(augmented_texts)}条")
        texts, labels = augmented_texts, augmented_labels
    
    dataset = DepressionDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_len=max_len
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=2,
        shuffle=is_train
    )

# 训练模型
def train_model(model, train_loader, val_loader, optimizer, scheduler, device, n_epochs, criterion):
    best_val_f1 = 0
    best_epoch = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_f1_micro': [],
        'val_f1_macro': [],
        'lr': []
    }
    
    # 保存开始时间
    start_time = time.time()
    
    for epoch in range(n_epochs):
        epoch_start_time = time.time()
        logger.info(f"第 {epoch+1}/{n_epochs} 轮训练")
        
        # 训练阶段
        model.train()
        train_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            
            if batch_idx % 50 == 0:
                logger.info(f"批次: {batch_idx}/{len(train_loader)}, 损失: {loss.item():.4f}")
        
        # 计算平均训练损失
        train_loss = train_loss / len(train_loader)
        history['train_loss'].append(train_loss)
        history['lr'].append(scheduler.get_last_lr()[0])
        
        # 验证阶段
        model.eval()
        val_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs, _ = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # 使用0.5作为阈值进行二进制分类
                preds = (outputs > 0.5).float()
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        # 计算平均验证损失
        val_loss = val_loss / len(val_loader)
        history['val_loss'].append(val_loss)
        
        # 计算评估指标
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        
        f1_micro = f1_score(true_labels, predictions, average='micro')
        f1_macro = f1_score(true_labels, predictions, average='macro')
        precision = precision_score(true_labels, predictions, average='macro', zero_division=0)
        recall = recall_score(true_labels, predictions, average='macro', zero_division=0)
        
        history['val_f1_micro'].append(f1_micro)
        history['val_f1_macro'].append(f1_macro)
        
        epoch_time = time.time() - epoch_start_time
        logger.info(f"轮次用时: {epoch_time:.2f}秒")
        logger.info(f"训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
        logger.info(f"F1 (micro): {f1_micro:.4f}, F1 (macro): {f1_macro:.4f}")
        logger.info(f"精确度: {precision:.4f}, 召回率: {recall:.4f}")
        
        # 保存最佳模型
        if f1_micro > best_val_f1:
            best_val_f1 = f1_micro
            best_epoch = epoch + 1
            torch.save(model.state_dict(), 'best_depression_emotion_model.bin')
            logger.info("保存最佳模型")
            
            # 保存每种情绪的F1分数
            f1_per_emotion = {}
            for i, emotion in enumerate(EMOTION_LIST):
                emo_f1 = f1_score(true_labels[:, i], predictions[:, i], zero_division=0)
                f1_per_emotion[emotion] = float(emo_f1)
                logger.info(f"{emotion}: F1 = {emo_f1:.4f}")
            
            # 保存验证结果
            val_results = {
                'epoch': best_epoch,
                'f1_micro': float(f1_micro),
                'f1_macro': float(f1_macro),
                'precision': float(precision),
                'recall': float(recall),
                'f1_per_emotion': f1_per_emotion
            }
            
            with open('best_validation_results.json', 'w') as f:
                json.dump(val_results, f, indent=4)
    
    total_time = time.time() - start_time
    logger.info(f"总训练时间: {total_time:.2f}秒")
    logger.info(f"最佳F1分数: {best_val_f1:.4f} (轮次 {best_epoch})")
    
    return history

# 测试模型
def test_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    predictions = []
    true_labels = []
    attention_weights_list = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs, attention_weights = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            # 使用0.5作为阈值进行二进制分类
            preds = (outputs > 0.5).float()
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            
            # 保存部分样本的注意力权重（用于可视化）
            if len(attention_weights_list) < 10:
                attention_weights_list.append(attention_weights.cpu().numpy())
    
    # 计算平均测试损失
    test_loss = test_loss / len(test_loader)
    
    # 计算评估指标
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    f1_micro = f1_score(true_labels, predictions, average='micro')
    f1_macro = f1_score(true_labels, predictions, average='macro')
    precision = precision_score(true_labels, predictions, average='macro', zero_division=0)
    recall = recall_score(true_labels, predictions, average='macro', zero_division=0)
    
    # 为每种情绪计算单独的F1分数
    f1_per_emotion = {}
    for i, emotion in enumerate(EMOTION_LIST):
        emo_f1 = f1_score(true_labels[:, i], predictions[:, i], zero_division=0)
        f1_per_emotion[emotion] = emo_f1
    
    logger.info(f"测试损失: {test_loss:.4f}")
    logger.info(f"F1 (micro): {f1_micro:.4f}, F1 (macro): {f1_macro:.4f}")
    logger.info(f"精确度: {precision:.4f}, 召回率: {recall:.4f}")
    logger.info("每种情绪的F1分数:")
    for emotion, f1 in f1_per_emotion.items():
        logger.info(f"{emotion}: {f1:.4f}")
    
    test_results = {
        'test_loss': float(test_loss),
        'f1_micro': float(f1_micro),
        'f1_macro': float(f1_macro),
        'precision': float(precision),
        'recall': float(recall),
        'f1_per_emotion': {k: float(v) for k, v in f1_per_emotion.items()}
    }
    
    return test_results, attention_weights_list

# 可视化训练历史
def plot_history(history):
    plt.figure(figsize=(15, 12))
    
    # 损失曲线
    plt.subplot(3, 1, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.title('训练和验证损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    
    # F1分数曲线
    plt.subplot(3, 1, 2)
    plt.plot(history['val_f1_micro'], label='F1 (micro)')
    plt.plot(history['val_f1_macro'], label='F1 (macro)')
    plt.title('验证集F1分数')
    plt.xlabel('轮次')
    plt.ylabel('F1分数')
    plt.legend()
    
    # 学习率曲线
    plt.subplot(3, 1, 3)
    plt.plot(history['lr'])
    plt.title('学习率变化')
    plt.xlabel('轮次')
    plt.ylabel('学习率')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    logger.info("保存训练历史图表至 training_history.png")

# 可视化注意力权重
def visualize_attention(texts, attention_weights, tokenizer, save_path="attention_visualization.png"):
    import seaborn as sns
    
    plt.figure(figsize=(10, len(texts) * 3))
    
    for i, (text, attn) in enumerate(zip(texts, attention_weights)):
        # 将注意力权重从形状 [batch_size, seq_len, 1] 转换为 [seq_len]
        attn = attn.reshape(-1)
        
        # 获取tokens
        tokens = tokenizer.encode(text, add_special_tokens=True)
        tokens = tokenizer.convert_ids_to_tokens(tokens)
        
        # 截断过长的序列
        max_len = min(len(tokens), len(attn), 50)  # 最多显示50个token
        tokens = tokens[:max_len]
        attn = attn[:max_len]
        
        plt.subplot(len(texts), 1, i+1)
        sns.heatmap([attn], cmap="YlOrRd", annot=False, xticklabels=tokens, yticklabels=False)
        plt.title(f"样本 {i+1}")
        plt.tight_layout()
    
    plt.savefig(save_path)
    logger.info(f"保存注意力可视化至 {save_path}")

def main():
    parser = argparse.ArgumentParser(description='抑郁情绪多标签分类 - 增强版BERT模型')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='训练或测试模式')
    parser.add_argument('--model_name', type=str, default='bert-base-cased', help='预训练模型名称')
    parser.add_argument('--model_path', type=str, default='best_depression_emotion_model.bin', help='保存的模型路径')
    parser.add_argument('--train_path', type=str, default='Dataset/train.json', help='训练数据路径')
    parser.add_argument('--val_path', type=str, default='Dataset/val.json', help='验证数据路径')
    parser.add_argument('--test_path', type=str, default='Dataset/test.json', help='测试数据路径')
    parser.add_argument('--epochs', type=int, default=8, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--max_length', type=int, default=256, help='最大序列长度')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='学习率')
    parser.add_argument('--use_focal_loss', action='store_true', help='是否使用Focal Loss')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs('outputs', exist_ok=True)
    
    # 记录运行时间
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"运行时间: {run_time}")
    logger.info(f"运行模式: {args.mode}")
    logger.info(f"使用设备: {device}")
    
    # 加载tokenizer
    logger.info(f"加载预训练模型: {args.model_name}")
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    
    if args.mode == 'train':
        # 加载训练和验证数据
        train_texts, train_labels = load_data(args.train_path)
        val_texts, val_labels = load_data(args.val_path)
        
        logger.info(f"加载了 {len(train_texts)} 个训练样本和 {len(val_texts)} 个验证样本")
        
        # 创建数据加载器
        train_loader = create_data_loader(train_texts, train_labels, tokenizer, args.max_length, args.batch_size, is_train=True)
        val_loader = create_data_loader(val_texts, val_labels, tokenizer, args.max_length, args.batch_size)
        
        # 初始化模型
        model = EnhancedDepressionEmotionClassifier(
            n_classes=len(EMOTION_LIST),
            model_name=args.model_name
        )
        model = model.to(device)
        
        # 设置优化器和学习率调度器
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, correct_bias=False, weight_decay=0.01)
        total_steps = len(train_loader) * args.epochs
        
        # 创建带有预热的学习率调度器
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),  # 10%的步数用于预热
            num_training_steps=total_steps
        )
        
        # 设置损失函数（多标签二分类）
        if args.use_focal_loss:
            logger.info("使用Focal Loss作为损失函数")
            criterion = FocalLoss()
        else:
            logger.info("使用二元交叉熵作为损失函数")
            criterion = nn.BCELoss()
        
        # 训练模型
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            n_epochs=args.epochs,
            criterion=criterion
        )
        
        # 可视化训练历史
        plot_history(history)
        
    elif args.mode == 'test':
        # 加载测试数据
        test_texts, test_labels = load_data(args.test_path)
        logger.info(f"加载了 {len(test_texts)} 个测试样本")
        
        # 创建测试数据加载器
        test_loader = create_data_loader(test_texts, test_labels, tokenizer, args.max_length, args.batch_size)
        
        # 初始化模型并加载权重
        model = EnhancedDepressionEmotionClassifier(
            n_classes=len(EMOTION_LIST),
            model_name=args.model_name
        )
        model.load_state_dict(torch.load(args.model_path))
        model = model.to(device)
        
        # 设置损失函数
        criterion = nn.BCELoss()
        
        # 测试模型
        test_results, attention_weights = test_model(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device
        )
        
        # 保存测试结果
        result_path = os.path.join('outputs', f'test_results_{run_time}.json')
        with open(result_path, 'w') as f:
            json.dump(test_results, f, indent=4)
        logger.info(f"保存测试结果至 {result_path}")
        
        # 可视化部分样本的注意力权重
        if attention_weights:
            sample_texts = test_texts[:len(attention_weights)]
            visualize_attention(
                sample_texts, 
                attention_weights, 
                tokenizer, 
                save_path=os.path.join('outputs', f'attention_visualization_{run_time}.png')
            )

if __name__ == "__main__":
    main() 