import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 导入自定义模块
from depression_detection.models.text_only_model import TextOnlyDepressionModel
from depression_detection.preprocess.text_preprocess import TextPreprocessor, TextAugmenter


class DepressionTextDataset(Dataset):
    """抑郁症文本数据集"""
    
    def __init__(self, csv_file, data_dir, preprocessor, max_length=512, augment=False, augmenter=None):
        """
        初始化数据集
        
        Args:
            csv_file: CSV文件路径
            data_dir: 数据目录
            preprocessor: 文本预处理器
            max_length: 最大文本长度
            augment: 是否进行数据增强
            augmenter: 文本增强器
        """
        self.df = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.preprocessor = preprocessor
        self.max_length = max_length
        self.augment = augment
        self.augmenter = augmenter if augmenter else TextAugmenter(preprocessor)
        
        # 读取所有文本
        self.texts = []
        for i, row in tqdm(self.df.iterrows(), total=len(self.df), desc="加载文本"):
            text_path = os.path.join(data_dir, row['text_path'])
            try:
                with open(text_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                self.texts.append(text)
            except Exception as e:
                print(f"无法读取文件 {text_path}: {str(e)}")
                self.texts.append("")
        
        # 提取统计特征
        self.statistical_features = []
        for text in tqdm(self.texts, desc="提取统计特征"):
            features = preprocessor.extract_statistical_features(text)
            self.statistical_features.append(list(features.values()))
            
        self.statistical_features = np.array(self.statistical_features)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.df.iloc[idx]['label']
        
        # 数据增强
        if self.augment and np.random.random() < 0.5:  # 50%概率进行增强
            text = self.augmenter.eda_augment(text)
            
        # 获取BERT输入
        input_ids, attention_mask = self.preprocessor.get_bert_tokenizer_outputs(text, max_length=self.max_length)
        
        # 统计特征
        statistical_features = torch.tensor(self.statistical_features[idx], dtype=torch.float32)
        
        return {
            'input_ids': input_ids.squeeze(0),
            'attention_mask': attention_mask.squeeze(0),
            'extra_features': statistical_features,
            'label': torch.tensor(label, dtype=torch.long)
        }


def train_model(model, train_loader, val_loader, device, args):
    """
    训练模型
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 设备
        args: 参数
        
    Returns:
        训练历史
    """
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    sentiment_criterion = nn.MSELoss()
    
    # 如果启用权重衰减，使用AdamW优化器
    if args.weight_decay > 0:
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # 使用学习率调度器
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'reduce':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    else:
        scheduler = None
    
    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_f1': [],
        'best_val_f1': 0
    }
    
    # 开始训练
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch in train_progress:
            # 准备数据
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            extra_features = batch['extra_features'].to(device)
            labels = batch['label'].to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            logits, sentiment = model(input_ids, attention_mask, extra_features)
            
            # 计算损失
            classification_loss = criterion(logits, labels)
            
            # 使用标签生成情感目标 (将0/1标签转换为-1.0/1.0情感分数)
            sentiment_targets = (labels.float() * 2 - 1).view(-1, 1)
            sentiment_loss = sentiment_criterion(sentiment, sentiment_targets)
            
            # 多任务损失
            loss = classification_loss + args.sentiment_weight * sentiment_loss
            
            # 反向传播和优化
            loss.backward()
            if args.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            
            # 更新统计信息
            train_loss += loss.item()
            
            # 获取预测结果
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(labels.cpu().numpy())
            
            # 更新进度条
            train_progress.set_postfix(loss=loss.item())
        
        # 计算训练指标
        train_acc = accuracy_score(train_labels, train_preds)
        train_loss = train_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        val_probs = []
        
        with torch.no_grad():
            val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            for batch in val_progress:
                # 准备数据
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                extra_features = batch['extra_features'].to(device)
                labels = batch['label'].to(device)
                
                # 前向传播
                logits, sentiment = model(input_ids, attention_mask, extra_features)
                
                # 计算损失
                classification_loss = criterion(logits, labels)
                
                # 使用标签生成情感目标
                sentiment_targets = (labels.float() * 2 - 1).view(-1, 1)
                sentiment_loss = sentiment_criterion(sentiment, sentiment_targets)
                
                # 多任务损失
                loss = classification_loss + args.sentiment_weight * sentiment_loss
                
                # 更新统计信息
                val_loss += loss.item()
                
                # 获取预测结果和概率
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())
                val_probs.extend(probs[:, 1])  # 保存正类的概率
                
                # 更新进度条
                val_progress.set_postfix(loss=loss.item())
        
        # 计算验证指标
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)
        val_precision = precision_score(val_labels, val_preds)
        val_recall = recall_score(val_labels, val_preds)
        val_auc = roc_auc_score(val_labels, val_probs)
        val_loss = val_loss / len(val_loader)
        
        # 更新学习率
        if scheduler is not None:
            if args.scheduler == 'cosine':
                scheduler.step()
            elif args.scheduler == 'reduce':
                scheduler.step(val_f1)
        
        # 更新训练历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # 保存最佳模型
        if val_f1 > history['best_val_f1']:
            history['best_val_f1'] = val_f1
            if args.save_dir:
                os.makedirs(args.save_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pt'))
                print(f"模型已保存到 {os.path.join(args.save_dir, 'best_model.pt')}")
        
        # 打印当前epoch的结果
        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        print(f"  Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val AUC: {val_auc:.4f}")
    
    return history


def evaluate_model(model, test_loader, device, args):
    """
    评估模型
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        device: 设备
        args: 参数
        
    Returns:
        评估指标
    """
    model.eval()
    
    test_preds = []
    test_labels = []
    test_probs = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="评估测试集"):
            # 准备数据
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            extra_features = batch['extra_features'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播
            logits, _ = model(input_ids, attention_mask, extra_features)
            
            # 获取预测结果和概率
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            test_preds.extend(preds)
            test_labels.extend(labels.cpu().numpy())
            test_probs.extend(probs[:, 1])  # 保存正类的概率
    
    # 计算评估指标
    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds)
    test_precision = precision_score(test_labels, test_preds)
    test_recall = recall_score(test_labels, test_preds)
    test_auc = roc_auc_score(test_labels, test_probs)
    
    # 计算混淆矩阵
    cm = confusion_matrix(test_labels, test_preds)
    
    # 保存评估结果
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        
        # 保存评估指标
        metrics = {
            'accuracy': test_acc,
            'f1': test_f1,
            'precision': test_precision,
            'recall': test_recall,
            'auc': test_auc
        }
        
        pd.DataFrame([metrics]).to_csv(os.path.join(args.save_dir, 'test_metrics.csv'), index=False)
        
        # 绘制混淆矩阵
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['正常', '抑郁'],
                    yticklabels=['正常', '抑郁'])
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('混淆矩阵')
        plt.tight_layout()
        plt.savefig(os.path.join(args.save_dir, 'confusion_matrix.png'))
    
    # 打印评估结果
    print("测试集评估结果:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  F1 Score: {test_f1:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    print(f"  AUC: {test_auc:.4f}")
    
    return {
        'accuracy': test_acc,
        'f1': test_f1, 
        'precision': test_precision,
        'recall': test_recall,
        'auc': test_auc,
        'confusion_matrix': cm
    }


def plot_training_history(history, save_dir=None):
    """
    绘制训练历史
    
    Args:
        history: 训练历史
        save_dir: 保存目录
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 绘制损失曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='训练损失')
    plt.plot(epochs, history['val_loss'], 'r-', label='验证损失')
    plt.title('训练和验证损失')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='训练准确率')
    plt.plot(epochs, history['val_acc'], 'r-', label='验证准确率')
    plt.plot(epochs, history['val_f1'], 'g-', label='验证F1分数')
    plt.title('训练和验证准确率')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy / F1')
    plt.legend()
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'training_history.png'))
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='训练抑郁症检测模型')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='./data', help='数据目录')
    parser.add_argument('--train_csv', type=str, default='train.csv', help='训练集CSV文件名')
    parser.add_argument('--val_csv', type=str, default='val.csv', help='验证集CSV文件名')
    parser.add_argument('--test_csv', type=str, default='test.csv', help='测试集CSV文件名')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, default='bert', choices=['bert', 'roberta', 'xlnet', 'electra'],
                        help='预训练模型类型')
    parser.add_argument('--hidden_dim', type=int, default=512, help='隐藏层维度')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout率')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=16, help='批量大小')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, help='梯度裁剪范数')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'reduce', 'none'],
                        help='学习率调度器类型')
    parser.add_argument('--sentiment_weight', type=float, default=0.2, help='情感任务权重')
    parser.add_argument('--max_length', type=int, default=512, help='最大文本长度')
    parser.add_argument('--use_augmentation', action='store_true', help='是否使用数据增强')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--save_dir', type=str, default='./results', help='保存结果目录')
    parser.add_argument('--no_cuda', action='store_true', help='不使用CUDA')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建结果目录
    if args.save_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_dir = os.path.join(args.save_dir, f"{args.model_type}_{timestamp}")
        os.makedirs(args.save_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"使用设备: {device}")
    
    # 初始化文本预处理器
    preprocessor = TextPreprocessor(bert_model_name=f"bert-base-chinese", use_cuda=not args.no_cuda)
    
    # 初始化文本增强器
    augmenter = TextAugmenter(preprocessor)
    
    # 加载数据集
    train_dataset = DepressionTextDataset(
        csv_file=os.path.join(args.data_dir, args.train_csv),
        data_dir=args.data_dir,
        preprocessor=preprocessor,
        max_length=args.max_length,
        augment=args.use_augmentation,
        augmenter=augmenter
    )
    
    val_dataset = DepressionTextDataset(
        csv_file=os.path.join(args.data_dir, args.val_csv),
        data_dir=args.data_dir,
        preprocessor=preprocessor,
        max_length=args.max_length,
        augment=False
    )
    
    test_dataset = DepressionTextDataset(
        csv_file=os.path.join(args.data_dir, args.test_csv),
        data_dir=args.data_dir,
        preprocessor=preprocessor,
        max_length=args.max_length,
        augment=False
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4
    )
    
    # 初始化模型
    model = TextOnlyDepressionModel(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_classes=2,
        model_type=args.model_type
    ).to(device)
    
    # 训练模型
    history = train_model(model, train_loader, val_loader, device, args)
    
    # 绘制训练历史
    plot_training_history(history, args.save_dir)
    
    # 加载最佳模型
    if args.save_dir and os.path.exists(os.path.join(args.save_dir, 'best_model.pt')):
        model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pt')))
    
    # 评估模型
    metrics = evaluate_model(model, test_loader, device, args)
    
    # 保存评估结果
    if args.save_dir:
        # 保存参数
        with open(os.path.join(args.save_dir, 'args.txt'), 'w') as f:
            for k, v in vars(args).items():
                f.write(f"{k}: {v}\n")
        
        # 保存模型结构
        with open(os.path.join(args.save_dir, 'model_summary.txt'), 'w') as f:
            f.write(str(model))
    
    return metrics


if __name__ == '__main__':
    main() 