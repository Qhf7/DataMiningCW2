import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from torchvision.models import resnet50, ResNet50_Weights


class VideoEncoder(nn.Module):
    """视频编码器"""
    
    def __init__(self, hidden_dim=512, dropout=0.3):
        super(VideoEncoder, self).__init__()
        # 使用预训练的ResNet作为视频特征提取器
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # 移除最后的全连接层
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # 时序特征处理
        self.temporal_conv = nn.Conv1d(2048, hidden_dim, kernel_size=3, padding=1)
        
        # Transformer编码器层
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        
    def forward(self, x):
        """
        x: 视频帧 [batch_size, frames, channels, height, width]
        """
        batch_size, frames, c, h, w = x.shape
        
        # 重塑为2D图像批次
        x = x.view(batch_size * frames, c, h, w)
        
        # 通过ResNet提取特征
        x = self.resnet(x)  # [batch_size*frames, 2048, 1, 1]
        x = x.view(batch_size, frames, -1)  # [batch_size, frames, 2048]
        
        # 时序卷积处理
        x = x.transpose(1, 2)  # [batch_size, 2048, frames]
        x = self.temporal_conv(x)  # [batch_size, hidden_dim, frames]
        x = x.transpose(1, 2)  # [batch_size, frames, hidden_dim]
        
        # Transformer编码
        x = x.transpose(0, 1)  # [frames, batch_size, hidden_dim]
        x = self.transformer(x)
        x = x.transpose(0, 1)  # [batch_size, frames, hidden_dim]
        
        # 获取全局表示 (使用平均池化)
        x = torch.mean(x, dim=1)  # [batch_size, hidden_dim]
        
        return x


class AudioEncoder(nn.Module):
    """音频编码器"""
    
    def __init__(self, input_dim=40, hidden_dim=512, dropout=0.3):
        super(AudioEncoder, self).__init__()
        
        # 1D卷积层进行特征提取
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, hidden_dim, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        # Transformer编码器层
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        x: MFCC特征 [batch_size, time_steps, n_mfcc]
        """
        # [batch_size, time_steps, n_mfcc] -> [batch_size, n_mfcc, time_steps]
        x = x.transpose(1, 2)
        
        # 1D卷积特征提取
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn3(self.conv3(x)))
        
        # [batch_size, hidden_dim, time_steps/4] -> [batch_size, time_steps/4, hidden_dim]
        x = x.transpose(1, 2)
        
        # Transformer编码
        x = x.transpose(0, 1)  # [time_steps/4, batch_size, hidden_dim]
        x = self.transformer(x)
        x = x.transpose(0, 1)  # [batch_size, time_steps/4, hidden_dim]
        
        # 全局表示
        x = torch.mean(x, dim=1)  # [batch_size, hidden_dim]
        
        return x


class TextEncoder(nn.Module):
    """文本编码器"""
    
    def __init__(self, hidden_dim=512):
        super(TextEncoder, self).__init__()
        
        # 使用预训练的BERT模型
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # 投影层，将BERT输出维度映射到统一的隐藏维度
        self.projection = nn.Linear(self.bert.config.hidden_size, hidden_dim)
        
    def forward(self, input_ids, attention_mask):
        """
        input_ids: 文本token ids [batch_size, seq_len]
        attention_mask: 注意力掩码 [batch_size, seq_len]
        """
        # 通过BERT编码文本
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # 使用[CLS]令牌的表示作为文本的全局表示
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, bert_dim]
        
        # 投影到统一维度
        text_features = self.projection(cls_output)  # [batch_size, hidden_dim]
        
        return text_features 