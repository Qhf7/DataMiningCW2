import os
import time
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    train_loss = 0.0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch} [Training]')
    
    for batch in progress_bar:
        # 将数据移至指定设备
        video = batch['video'].to(device)
        audio = batch['audio'].to(device)
        text_input_ids = batch['text_input_ids'].to(device)
        text_attention_mask = batch['text_attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # 梯度清零
        optimizer.zero_grad()
        
        # 前向传播
        logits, _ = model(video, audio, text_input_ids, text_attention_mask)
        
        # 计算损失
        loss = criterion(logits, labels)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 参数更新
        optimizer.step()
        
        # 累计损失
        train_loss += loss.item()
        
        # 获取预测结果
        _, preds = torch.max(logits, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # 更新进度条
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # 计算平均损失和评估指标
    train_loss /= len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return train_loss, accuracy, precision, recall, f1


def validate(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Validation')
        
        for batch in progress_bar:
            # 将数据移至指定设备
            video = batch['video'].to(device)
            audio = batch['audio'].to(device)
            text_input_ids = batch['text_input_ids'].to(device)
            text_attention_mask = batch['text_attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播
            logits, _ = model(video, audio, text_input_ids, text_attention_mask)
            
            # 计算损失
            loss = criterion(logits, labels)
            
            # 累计损失
            val_loss += loss.item()
            
            # 获取预测结果
            probs = torch.softmax(logits, dim=1)
            _, preds = torch.max(logits, 1)
            
            all_probs.extend(probs[:, 1].cpu().numpy())  # 假设类别1是正例（抑郁症）
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 更新进度条
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # 计算平均损失和评估指标
    val_loss /= len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # 计算AUC
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    return val_loss, accuracy, precision, recall, f1, auc, cm


def test_model(model, test_loader, criterion, device):
    """测试模型"""
    # 测试评估与验证相同
    return validate(model, test_loader, criterion, device)


def train_model(model, train_loader, val_loader, config, device):
    """训练模型的完整流程"""
    # 设置优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # 学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 记录最佳模型
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    best_model_path = os.path.join(config['save_dir'], 'best_model.pth')
    
    # 训练循环
    for epoch in range(1, config['epochs'] + 1):
        # 训练一个epoch
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # 验证
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc, val_cm = validate(
            model, val_loader, criterion, device
        )
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 打印训练信息
        print(f'Epoch {epoch}/{config["epochs"]}:')
        print(f'  Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}')
        
        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_f1': val_f1,
                'config': config
            }, best_model_path)
            print(f'  Saved best model with F1: {val_f1:.4f}')
        
        # 保存检查点
        if epoch % config['save_interval'] == 0:
            checkpoint_path = os.path.join(config['save_dir'], f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_f1': val_f1,
                'config': config
            }, checkpoint_path)
    
    return best_val_loss, best_val_f1, best_model_path


def load_model(model, model_path, device):
    """加载训练好的模型"""
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['config']


def extract_modality_weights(model, data_loader, device):
    """提取模型对不同模态的权重"""
    model.eval()
    video_weights = []
    audio_weights = []
    text_weights = []
    labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Extracting modality weights'):
            # 将数据移至指定设备
            video = batch['video'].to(device)
            audio = batch['audio'].to(device)
            text_input_ids = batch['text_input_ids'].to(device)
            text_attention_mask = batch['text_attention_mask'].to(device)
            batch_labels = batch['label'].cpu().numpy()
            
            # 前向传播
            _, modality_weights = model(video, audio, text_input_ids, text_attention_mask)
            
            # 提取权重
            video_w, audio_w, text_w = modality_weights
            
            video_weights.extend(video_w.cpu().numpy())
            audio_weights.extend(audio_w.cpu().numpy())
            text_weights.extend(text_w.cpu().numpy())
            labels.extend(batch_labels)
    
    return np.array(video_weights), np.array(audio_weights), np.array(text_weights), np.array(labels) 