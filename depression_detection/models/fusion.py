import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class CrossModalAttention(nn.Module):
    """跨模态注意力机制"""
    
    def __init__(self, hidden_dim=512, num_heads=8, dropout=0.1):
        super(CrossModalAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim必须能被num_heads整除"
        
        # 多头注意力的线性投影
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value):
        """
        跨模态注意力计算
        
        Args:
            query: 查询特征 [batch_size, hidden_dim]
            key: 键特征 [batch_size, hidden_dim]
            value: 值特征 [batch_size, hidden_dim]
            
        Returns:
            注意力加权的特征 [batch_size, hidden_dim]
        """
        batch_size = query.size(0)
        
        # 线性投影并重塑为多头形式
        q = self.q_proj(query).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, 1, D/H]
        k = self.k_proj(key).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)    # [B, H, 1, D/H]
        v = self.v_proj(value).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, 1, D/H]
        
        # 注意力得分计算
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, 1, 1]
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # 加权求和
        attn_output = torch.matmul(attn_probs, v)  # [B, H, 1, D/H]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, 1, self.hidden_dim)  # [B, 1, D]
        attn_output = self.out_proj(attn_output).squeeze(1)  # [B, D]
        
        return attn_output


class ModalityGating(nn.Module):
    """模态门控机制：调整不同模态的重要性权重"""
    
    def __init__(self, hidden_dim=512, dropout=0.1):
        super(ModalityGating, self).__init__()
        
        # 门控网络
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)  # 三个模态的权重
        )
        
    def forward(self, video_feat, audio_feat, text_feat):
        """
        计算三个模态的融合权重
        
        Args:
            video_feat: 视频特征 [batch_size, hidden_dim]
            audio_feat: 音频特征 [batch_size, hidden_dim]
            text_feat: 文本特征 [batch_size, hidden_dim]
            
        Returns:
            权重和融合特征
        """
        # 拼接所有模态特征
        combined = torch.cat([video_feat, audio_feat, text_feat], dim=1)
        
        # 计算门控权重
        gates = self.gate_net(combined)
        gates = F.softmax(gates, dim=1)  # [batch_size, 3]
        
        # 分离每个模态的权重
        video_weight = gates[:, 0].unsqueeze(1)
        audio_weight = gates[:, 1].unsqueeze(1)
        text_weight = gates[:, 2].unsqueeze(1)
        
        # 加权融合
        weighted_video = video_feat * video_weight
        weighted_audio = audio_feat * audio_weight
        weighted_text = text_feat * text_weight
        
        # 加权融合特征
        fused_features = weighted_video + weighted_audio + weighted_text
        
        return fused_features, (video_weight, audio_weight, text_weight)


class AdaptiveFusion(nn.Module):
    """自适应跨模态融合模块"""
    
    def __init__(self, hidden_dim=512, dropout=0.1):
        super(AdaptiveFusion, self).__init__()
        
        # 跨模态注意力
        self.video_to_audio_attn = CrossModalAttention(hidden_dim, num_heads=8, dropout=dropout)
        self.video_to_text_attn = CrossModalAttention(hidden_dim, num_heads=8, dropout=dropout)
        
        self.audio_to_video_attn = CrossModalAttention(hidden_dim, num_heads=8, dropout=dropout)
        self.audio_to_text_attn = CrossModalAttention(hidden_dim, num_heads=8, dropout=dropout)
        
        self.text_to_video_attn = CrossModalAttention(hidden_dim, num_heads=8, dropout=dropout)
        self.text_to_audio_attn = CrossModalAttention(hidden_dim, num_heads=8, dropout=dropout)
        
        # 视频增强特征
        self.enhanced_video_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # 音频增强特征
        self.enhanced_audio_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # 文本增强特征
        self.enhanced_text_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # 模态门控机制
        self.modality_gating = ModalityGating(hidden_dim, dropout)
        
        # 最终融合投影
        self.final_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, video_features, audio_features, text_features):
        """
        自适应跨模态融合
        
        Args:
            video_features: 视频特征 [batch_size, hidden_dim]
            audio_features: 音频特征 [batch_size, hidden_dim]
            text_features: 文本特征 [batch_size, hidden_dim]
            
        Returns:
            融合特征 [batch_size, hidden_dim]
        """
        batch_size = video_features.size(0)
        
        # 视频与其他模态的跨模态注意力
        video_attend_audio = self.video_to_audio_attn(video_features, audio_features, audio_features)
        video_attend_text = self.video_to_text_attn(video_features, text_features, text_features)
        
        # 音频与其他模态的跨模态注意力
        audio_attend_video = self.audio_to_video_attn(audio_features, video_features, video_features)
        audio_attend_text = self.audio_to_text_attn(audio_features, text_features, text_features)
        
        # 文本与其他模态的跨模态注意力
        text_attend_video = self.text_to_video_attn(text_features, video_features, video_features)
        text_attend_audio = self.text_to_audio_attn(text_features, audio_features, audio_features)
        
        # 增强的视频特征
        enhanced_video = torch.cat([
            video_features, 
            video_attend_audio,
            video_attend_text
        ], dim=1)
        enhanced_video = self.enhanced_video_proj(enhanced_video)
        
        # 增强的音频特征
        enhanced_audio = torch.cat([
            audio_features,
            audio_attend_video,
            audio_attend_text
        ], dim=1)
        enhanced_audio = self.enhanced_audio_proj(enhanced_audio)
        
        # 增强的文本特征
        enhanced_text = torch.cat([
            text_features,
            text_attend_video,
            text_attend_audio
        ], dim=1)
        enhanced_text = self.enhanced_text_proj(enhanced_text)
        
        # 通过门控机制融合增强特征
        fused_features, modality_weights = self.modality_gating(
            enhanced_video, enhanced_audio, enhanced_text
        )
        
        # 最终投影
        fused_features = self.final_proj(fused_features)
        
        return fused_features, modality_weights 