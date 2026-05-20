# run_train.py

import torch
import argparse
from src.train import train

# ============================================================
# 设备
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 完整 data 训练配置
# ============================================================


# 训练脚本

# $env:PYTHONPATH="."

# 从头训练，每5轮诊断一次
# python -m run_train --epochs 50 (((停用--diagnostic-every 5

# 继续训练，每5轮诊断一次
# python -m run_train --resume experiments/20260504_xxxxxx/checkpoint.pth --epochs 80 (((停用--diagnostic-every 5
# 接着训练的话把最后一个字段删除就可以

# 只训练，不输出诊断（默认）
# python -m run_train --epochs 50


# 手动备份checkpoint文件（如果需要）
# cp experiments/20260507_213813/checkpoint.pth experiments/20260507_213813/checkpoint_epoch40.pth


# 如果之前已经运行生成过结果分析可以采取如下操作
# 1. 运行前手动备份：
# mv evaluation_results/20260503_xxxxx evaluation_results/20260503_xxxxx_backup
# 2. 运行前删除旧结果：
# rm -rf evaluation_results/20260503_xxxxx
# 3. 手动指定不同输出目录：
# python -m src.evaluate --data_dir data --checkpoint experiments/20260503_xxxxx/checkpoint.pth --output_dir evaluation_results/new_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="聚类训练")
    parser.add_argument("--resume", type=str, default=None,
                        help="从checkpoint恢复训练, 如 experiments/20260503_123456/checkpoint.pth")
    parser.add_argument("--epochs", type=int, default=100,
                        help="训练总轮数")
    parser.add_argument("--diagnostic-every", type=int, default=None,
                        help="每N轮输出一次诊断(silhouette/UMAP等), 默认关闭")
    args = parser.parse_args()

    train(
        data_dir = "data",   # 数据集路径
        device = DEVICE,
        batch_size = 4,
        epochs = args.epochs,
        lr = 1e-5,                      # 稳定学习率
        beta = 1.0,                     # KL 权重
        gamma = 0.05,                   # entropy 权重
        log_interval = 1,               # 每个 epoch 打印
        resume_checkpoint = args.resume,  # 恢复训练
        diagnostic_every = args.diagnostic_every  # 诊断间隔
    )
