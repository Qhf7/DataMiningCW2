import torch
import torch.nn as nn
import torch.nn.functional as F

from depression_detection.models.encoders import VideoEncoder, AudioEncoder, TextEncoder
from depression_detection.models.fusion import AdaptiveFusion


class MultimodalDepressionModel(nn.Module):
    """多模态抑郁症检测模型"""
    
    def __init__(self, hidden_dim=512, dropout=0.3, num_classes=2):
        super(MultimodalDepressionModel, self).__init__()
        
        # 各模态编码器
        self.video_encoder = VideoEncoder(hidden_dim=hidden_dim, dropout=dropout)
        self.audio_encoder = AudioEncoder(hidden_dim=hidden_dim, dropout=dropout)
        self.text_encoder = TextEncoder(hidden_dim=hidden_dim)
        
        # 跨模态自适应融合模块
        self.fusion_module = AdaptiveFusion(hidden_dim=hidden_dim, dropout=dropout)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, video, audio, text_input_ids, text_attention_mask):
        """
        前向传播
        
        Args:
            video: 视频特征 [batch_size, frames, channels, height, width]
            audio: 音频特征 [batch_size, time_steps, n_mfcc]
            text_input_ids: 文本输入IDs [batch_size, seq_len]
            text_attention_mask: 文本注意力掩码 [batch_size, seq_len]
        
        Returns:
            logits: 分类logits [batch_size, num_classes]
            modality_weights: 各模态的权重
        """
        # 提取各模态特征
        video_features = self.video_encoder(video)
        audio_features = self.audio_encoder(audio)
        text_features = self.text_encoder(text_input_ids, text_attention_mask)
        
        # 跨模态自适应融合
        fused_features, modality_weights = self.fusion_module(
            video_features, audio_features, text_features
        )
        
        # 分类
        logits = self.classifier(fused_features)
        
        return logits, modality_weights
    
    def extract_features(self, video, audio, text_input_ids, text_attention_mask):
        """
        提取各模态特征和融合特征（用于可视化和分析）
        
        Returns:
            各模态特征和融合特征
        """
        # 提取各模态特征
        video_features = self.video_encoder(video)
        audio_features = self.audio_encoder(audio)
        text_features = self.text_encoder(text_input_ids, text_attention_mask)
        
        # 跨模态自适应融合
        fused_features, modality_weights = self.fusion_module(
            video_features, audio_features, text_features
        )
        
        return {
            'video_features': video_features,
            'audio_features': audio_features,
            'text_features': text_features,
            'fused_features': fused_features,
            'modality_weights': modality_weights
        }