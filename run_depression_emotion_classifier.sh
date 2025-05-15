#!/bin/bash

# 设置环境变量
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=0  # 如果有多个GPU，可以选择使用哪一个

# 创建输出目录
mkdir -p outputs logs

# 获取当前时间作为运行标识
RUN_ID=$(date +"%Y%m%d_%H%M%S")
echo "运行ID: $RUN_ID"

# 定义函数：记录日志
log() {
    echo "$(date +"%Y-%m-%d %H:%M:%S") - $1" | tee -a "logs/run_$RUN_ID.log"
}

# 开始训练
log "开始训练 BERT 抑郁情绪分类器"

# 运行训练模式
log "运行训练模式 - 标准BERT模型"
python depression_emotion_classifier.py \
    --mode train \
    --model_name bert-base-cased \
    --train_path Dataset/train.json \
    --val_path Dataset/val.json \
    --epochs 4 \
    --batch_size 8 \
    --max_length 256 \
    --learning_rate 2e-5 \
    2>&1 | tee logs/train_bert_${RUN_ID}.log

# 训练完成后，运行测试模式
log "运行测试模式 - 使用最佳模型"
python depression_emotion_classifier.py \
    --mode test \
    --model_name bert-base-cased \
    --model_path best_depression_emotion_model.bin \
    --test_path Dataset/test.json \
    --batch_size 16 \
    --max_length 256 \
    2>&1 | tee logs/test_bert_${RUN_ID}.log

# 运行Focal Loss训练模式（可选）
if [ "$1" == "focal" ]; then
    log "运行训练模式 - 使用Focal Loss"
    python depression_emotion_classifier.py \
        --mode train \
        --model_name bert-base-cased \
        --train_path Dataset/train.json \
        --val_path Dataset/val.json \
        --epochs 4 \
        --batch_size 8 \
        --max_length 256 \
        --learning_rate 2e-5 \
        --use_focal_loss \
        2>&1 | tee logs/train_bert_focal_${RUN_ID}.log
        
    log "运行测试模式 - Focal Loss模型"
    python depression_emotion_classifier.py \
        --mode test \
        --model_name bert-base-cased \
        --model_path best_depression_emotion_model.bin \
        --test_path Dataset/test.json \
        --batch_size 16 \
        --max_length 256 \
        2>&1 | tee logs/test_bert_focal_${RUN_ID}.log
fi

log "全部任务完成!" 