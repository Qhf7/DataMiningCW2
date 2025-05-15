import os
import sys
import argparse
from datetime import datetime

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from depression_detection.utils.data_utils import create_dummy_dataset
from depression_detection.models.multimodal_model import MultimodalDepressionModel
from depression_detection.utils.train_utils import train_model, test_model, load_model


def parse_args():
    parser = argparse.ArgumentParser(description="多模态抑郁症检测演示")
    parser.add_argument("--create_data", action="store_true", help="创建示例数据集")
    parser.add_argument("--num_samples", type=int, default=100, help="示例数据集样本数")
    parser.add_argument("--data_path", type=str, default="./data", help="数据路径")
    parser.add_argument("--save_dir", type=str, default="./saved_models", help="模型保存路径")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--hidden_dim", type=int, default=64, help="隐藏层维度")
    parser.add_argument("--only_test", action="store_true", help="仅测试不训练")
    parser.add_argument("--model_path", type=str, default=None, help="测试模式下加载的模型路径")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 创建示例数据集
    if args.create_data:
        print("\n=== 创建示例数据集 ===")
        create_dummy_dataset(args.data_path, args.num_samples)
    
    # 创建模型保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设置设备
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    
    if args.only_test and args.model_path:
        # 仅测试模式
        print(f"\n=== 加载预训练模型并测试 ===")
        model = MultimodalDepressionModel(
            hidden_dim=args.hidden_dim,
            dropout=0.3,
            num_classes=2
        ).to(device)
        
        model, config = load_model(model, args.model_path, device)
        
        # 加载数据
        from depression_detection.utils.data_utils import get_dataloaders
        _, _, test_loader = get_dataloaders(
            args.data_path, 
            batch_size=args.batch_size,
            num_workers=2
        )
        
        # 测试模型
        criterion = torch.nn.CrossEntropyLoss()
        test_loss, test_acc, test_prec, test_rec, test_f1, test_auc, test_cm = test_model(
            model, test_loader, criterion, device
        )
        
        print(f"\n测试结果:")
        print(f"Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
        print(f"Precision: {test_prec:.4f}, Recall: {test_rec:.4f}")
        print(f"F1 Score: {test_f1:.4f}, AUC: {test_auc:.4f}")
        print(f"混淆矩阵:\n{test_cm}")
    else:
        # 训练模式
        print(f"\n=== 创建并训练模型 ===")
        model = MultimodalDepressionModel(
            hidden_dim=args.hidden_dim,
            dropout=0.3,
            num_classes=2
        ).to(device)
        
        # 加载数据
        from depression_detection.utils.data_utils import get_dataloaders
        train_loader, val_loader, test_loader = get_dataloaders(
            args.data_path, 
            batch_size=args.batch_size,
            num_workers=2
        )
        
        # 设置训练配置
        config = {
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'epochs': args.epochs,
            'save_interval': max(1, args.epochs // 3),
            'save_dir': args.save_dir,
        }
        
        # 训练模型
        print(f"\n开始训练 ({args.epochs} epochs)...")
        start_time = datetime.now()
        best_val_loss, best_val_f1, best_model_path = train_model(
            model, train_loader, val_loader, config, device
        )
        end_time = datetime.now()
        
        print(f"\n训练完成！用时: {end_time - start_time}")
        print(f"最佳验证集 Loss: {best_val_loss:.4f}, F1: {best_val_f1:.4f}")
        print(f"最佳模型已保存至: {best_model_path}")
        
        # 加载最佳模型测试
        print("\n加载最佳模型进行测试...")
        model, _ = load_model(model, best_model_path, device)
        
        # 测试模型
        criterion = torch.nn.CrossEntropyLoss()
        test_loss, test_acc, test_prec, test_rec, test_f1, test_auc, test_cm = test_model(
            model, test_loader, criterion, device
        )
        
        print(f"\n测试结果:")
        print(f"Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
        print(f"Precision: {test_prec:.4f}, Recall: {test_rec:.4f}")
        print(f"F1 Score: {test_f1:.4f}, AUC: {test_auc:.4f}")
        print(f"混淆矩阵:\n{test_cm}")


if __name__ == "__main__":
    main() 