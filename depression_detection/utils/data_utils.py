import os
import cv2
import torch
import numpy as np
import pandas as pd
import librosa
import argparse
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torchvision import transforms

class MultimodalDataset(Dataset):
    """多模态抑郁症数据集加载器"""
    
    def __init__(self, data_path, csv_file, transform=None, max_frames=150, 
                 max_audio_len=30000, max_text_len=128, mode='train'):
        """
        初始化多模态数据集
        
        Args:
            data_path: 数据根目录
            csv_file: 包含文件路径和标签的CSV文件
            transform: 视频帧的预处理转换
            max_frames: 最大视频帧数
            max_audio_len: 最大音频长度
            max_text_len: 最大文本长度
            mode: 'train', 'val', 或 'test'
        """
        self.data_path = data_path
        self.df = pd.read_csv(os.path.join(data_path, csv_file))
        self.transform = transform if transform else self._get_default_transform(mode)
        self.max_frames = max_frames
        self.max_audio_len = max_audio_len
        self.max_text_len = max_text_len
        self.mode = mode
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def __len__(self):
        return len(self.df)
    
    def _get_default_transform(self, mode):
        """获取默认的视频帧转换"""
        if mode == 'train':
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])
    
    def _load_video(self, video_path):
        """加载视频帧"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        frame_count = 0
        success = True
        
        while success and frame_count < self.max_frames:
            success, frame = cap.read()
            if success:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)
                frame_count += 1
        
        cap.release()
        
        # 填充到最大帧数
        if len(frames) < self.max_frames:
            padding = [torch.zeros_like(frames[0]) for _ in range(self.max_frames - len(frames))]
            frames.extend(padding)
        
        return torch.stack(frames)
    
    def _load_audio(self, audio_path):
        """加载音频特征"""
        try:
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # 提取音频特征 - MFCC
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
            mfcc = mfcc.T  # 转置为 (time_steps, n_mfcc)
            
            # 确保长度一致
            if mfcc.shape[0] > self.max_audio_len:
                mfcc = mfcc[:self.max_audio_len, :]
            else:
                padding = np.zeros((self.max_audio_len - mfcc.shape[0], mfcc.shape[1]))
                mfcc = np.vstack((mfcc, padding))
            
            return torch.FloatTensor(mfcc)
        except Exception as e:
            print(f"Error loading audio: {e}")
            return torch.zeros((self.max_audio_len, 40))
    
    def _load_text(self, text_path):
        """加载并处理文本数据"""
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        except:
            text = ""
            
        # 使用BERT tokenizer处理文本
        encoded_text = self.tokenizer(
            text,
            max_length=self.max_text_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded_text['input_ids'].squeeze(0),
            'attention_mask': encoded_text['attention_mask'].squeeze(0)
        }
    
    def __getitem__(self, idx):
        """获取一个多模态样本"""
        row = self.df.iloc[idx]
        
        # 获取文件路径
        video_path = os.path.join(self.data_path, row['video_path'])
        audio_path = os.path.join(self.data_path, row['audio_path'])
        text_path = os.path.join(self.data_path, row['text_path'])
        
        # 加载各模态数据
        video_data = self._load_video(video_path)
        audio_data = self._load_audio(audio_path)
        text_data = self._load_text(text_path)
        
        # 标签
        label = torch.tensor(row['label'], dtype=torch.long)
        
        return {
            'video': video_data,
            'audio': audio_data,
            'text_input_ids': text_data['input_ids'],
            'text_attention_mask': text_data['attention_mask'],
            'label': label
        }

def get_dataloaders(data_path, batch_size=8, num_workers=4):
    """
    创建训练、验证和测试数据加载器
    
    Args:
        data_path: 数据目录
        batch_size: 批次大小
        num_workers: 数据加载的工作线程数
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = MultimodalDataset(data_path, 'train.csv', mode='train')
    val_dataset = MultimodalDataset(data_path, 'val.csv', mode='val')
    test_dataset = MultimodalDataset(data_path, 'test.csv', mode='test')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def create_dummy_dataset(output_dir, num_samples=10):
    """创建测试数据集"""
    print(f"正在创建示例数据集，样本数：{num_samples}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'videos'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'audios'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'texts'), exist_ok=True)
    
    data = []
    for i in range(num_samples):
        # 假设我们有视频、音频和文本文件
        video_path = f'videos/sample_{i}.mp4'
        audio_path = f'audios/sample_{i}.wav'
        text_path = f'texts/sample_{i}.txt'
        
        # 随机标签 (0: 无抑郁, 1: 抑郁)
        label = np.random.randint(0, 2)
        
        data.append({
            'video_path': video_path,
            'audio_path': audio_path,
            'text_path': text_path,
            'label': label
        })
    
    # 创建训练、验证和测试集
    df = pd.DataFrame(data)
    train_df = df.iloc[:int(0.7*num_samples)]
    val_df = df.iloc[int(0.7*num_samples):int(0.9*num_samples)]
    test_df = df.iloc[int(0.9*num_samples):]
    
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    # 创建简单的示例文本文件
    for i in range(num_samples):
        with open(os.path.join(output_dir, f'texts/sample_{i}.txt'), 'w') as f:
            if np.random.randint(0, 2) == 1:
                f.write("我感到非常难过和无助。没有什么能让我感到快乐。")
            else:
                f.write("今天是个不错的日子，我感觉很好，非常开心。")
    
    # 创建空的示例视频和音频文件（仅供演示）
    for i in range(num_samples):
        with open(os.path.join(output_dir, f'videos/sample_{i}.mp4'), 'w') as f:
            f.write("这是一个示例视频文件")
        with open(os.path.join(output_dir, f'audios/sample_{i}.wav'), 'w') as f:
            f.write("这是一个示例音频文件")
    
    print(f"示例数据集创建完成：{output_dir}")

# 添加直接运行功能
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="数据工具函数")
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # 创建示例数据集的子命令
    create_dummy_parser = subparsers.add_parser("create_dummy_dataset", help="创建示例数据集")
    create_dummy_parser.add_argument("output_dir", type=str, help="输出目录")
    create_dummy_parser.add_argument("--num_samples", type=int, default=10, help="样本数量")
    
    args = parser.parse_args()
    
    if args.command == "create_dummy_dataset":
        create_dummy_dataset(args.output_dir, args.num_samples)
    else:
        parser.print_help() 