import os
import torch
import argparse
import numpy as np
import cv2
import librosa
from transformers import BertTokenizer
from torchvision import transforms

from depression_detection.models.multimodal_model import MultimodalDepressionModel
from depression_detection.utils.train_utils import load_model


def parse_args():
    parser = argparse.ArgumentParser(description='使用多模态模型进行抑郁症检测')
    parser.add_argument('--model_path', type=str, required=True, help='预训练模型路径')
    parser.add_argument('--video_path', type=str, required=True, help='输入视频文件路径')
    parser.add_argument('--audio_path', type=str, required=True, help='输入音频文件路径')
    parser.add_argument('--text_path', type=str, required=True, help='输入文本文件路径')
    parser.add_argument('--hidden_dim', type=int, default=512, help='隐藏层维度')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout概率')
    parser.add_argument('--max_frames', type=int, default=150, help='最大视频帧数')
    parser.add_argument('--max_audio_len', type=int, default=30000, help='最大音频长度')
    parser.add_argument('--max_text_len', type=int, default=128, help='最大文本长度')
    return parser.parse_args()


def load_video(video_path, max_frames=150, transform=None):
    """加载视频帧"""
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    success = True
    
    # 默认变换
    if transform is None:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
    
    while success and frame_count < max_frames:
        success, frame = cap.read()
        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if transform:
                frame = transform(frame)
            frames.append(frame)
            frame_count += 1
    
    cap.release()
    
    # 填充到最大帧数
    if len(frames) < max_frames:
        padding = [torch.zeros_like(frames[0]) for _ in range(max_frames - len(frames))]
        frames.extend(padding)
    
    return torch.stack(frames).unsqueeze(0)  # [1, frames, channels, height, width]


def load_audio(audio_path, max_audio_len=30000):
    """加载音频特征"""
    try:
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # 提取音频特征 - MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfcc = mfcc.T  # 转置为 (time_steps, n_mfcc)
        
        # 确保长度一致
        if mfcc.shape[0] > max_audio_len:
            mfcc = mfcc[:max_audio_len, :]
        else:
            padding = np.zeros((max_audio_len - mfcc.shape[0], mfcc.shape[1]))
            mfcc = np.vstack((mfcc, padding))
        
        return torch.FloatTensor(mfcc).unsqueeze(0)  # [1, time_steps, n_mfcc]
    except Exception as e:
        print(f"Error loading audio: {e}")
        return torch.zeros((1, max_audio_len, 40))


def load_text(text_path, max_text_len=128):
    """加载并处理文本数据"""
    try:
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
    except:
        text = ""
        
    # 使用BERT tokenizer处理文本
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoded_text = tokenizer(
        text,
        max_length=max_text_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    return encoded_text['input_ids'], encoded_text['attention_mask']


def main():
    args = parse_args()
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    print("加载模型...")
    model = MultimodalDepressionModel(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_classes=2  # 二分类任务
    ).to(device)
    
    # 加载预训练模型
    model, _ = load_model(model, args.model_path, device)
    model.eval()
    
    # 加载输入数据
    print("加载输入数据...")
    video = load_video(args.video_path, args.max_frames).to(device)
    audio = load_audio(args.audio_path, args.max_audio_len).to(device)
    text_input_ids, text_attention_mask = load_text(args.text_path, args.max_text_len)
    text_input_ids = text_input_ids.to(device)
    text_attention_mask = text_attention_mask.to(device)
    
    # 模型推理
    print("进行推理...")
    with torch.no_grad():
        logits, modality_weights = model(video, audio, text_input_ids, text_attention_mask)
        
        # 获取预测概率和类别
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(logits, dim=1).item()
        depression_prob = probs[0, 1].item()
        
        # 获取各模态权重
        video_weight, audio_weight, text_weight = modality_weights
        video_weight = video_weight.cpu().numpy()[0, 0]
        audio_weight = audio_weight.cpu().numpy()[0, 0]
        text_weight = text_weight.cpu().numpy()[0, 0]
        
    # 输出结果
    print("\n抑郁症检测结果:")
    print(f"预测类别: {'抑郁' if pred_class == 1 else '正常'}")
    print(f"抑郁概率: {depression_prob:.4f}")
    print("\n各模态贡献权重:")
    print(f"视频模态: {video_weight:.4f}")
    print(f"音频模态: {audio_weight:.4f}")
    print(f"文本模态: {text_weight:.4f}")
    
    # 简单风险评估
    if depression_prob > 0.8:
        risk_level = "高风险"
    elif depression_prob > 0.5:
        risk_level = "中等风险"
    else:
        risk_level = "低风险"
    
    print(f"\n抑郁风险评估: {risk_level}")
    
    if risk_level != "低风险":
        print("\n建议: 请考虑咨询专业心理医生或精神科医生获取进一步评估和支持。")


if __name__ == '__main__':
    main() 