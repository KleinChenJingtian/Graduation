# src/utils.py

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


# ============================================================
# 实验目录管理
# ============================================================

def make_experiment_dir(base_dir="Graduation/experiments", exp_name=None):
    """
    创建实验目录，用时间戳区分不同实验
    """
    if exp_name is None:
        exp_name = datetime.now().strftime("%Y%m%d_%H%M%S") # strftime的作用是将时间对象转换为指定格式的字符串

    exp_dir = os.path.join(base_dir, exp_name) # 例如 Graduation/experiments/20260402_163005
    os.makedirs(exp_dir, exist_ok=True) # os.makedirs: 递归创建文件夹。如果父目录 Graduation/experiments 不存在，它会连带着一起建好。exist_ok=True: 如果文件夹已经存在，不会抛出异常。

    return exp_dir


# ============================================================
# TensorBoard 日志封装
# ============================================================

class TBLogger:
    """
    TensorBoard 日志记录工具
    """

    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir) # SummaryWriter 是 PyTorch 提供的一个类，用于将训练过程中的各种指标（如损失、准确率等）记录到 TensorBoard 中，以便可视化分析。
        # log_dir 参数指定了日志文件的保存路径，通常是一个包含时间戳的目录，以区分不同的实验。

    def log_losses(self, losses: dict, epoch: int):
        """
        losses: {
            "total": float,
            "dist": float,
            "dp": float,
            "entropy": float,
            "var": float
        }
        """
        for k, v in losses.items():  # items(): 遍历字典，把键（比如 "total"）给 k，把值（比如 0.5）给 v
            self.writer.add_scalar(f"Loss/{k}", v, epoch) # add_scalar 是 SummaryWriter 的一个方法，用于记录一个标量值（如损失）到 TensorBoard 中。第一个参数是这个标量的名字（这里我们用 f"Loss/{k}" 来构造一个名字，比如 "Loss/total"），第二个参数是这个标量的数值（v），第三个参数是这个数值对应的训练轮数（epoch）。这样，当你在 TensorBoard 中查看 Loss/total 这条曲线时，就会看到随着 epoch 的增加，total loss 的变化趋势。
            # 循环了 5 次，TensorBoard 就会为你自动生成 5 张不同的折线图

    def log_pi(self, pi, epoch: int):
        """
        记录 π_k 分布
        """
        # 假设 pi = [0.8, 0.1, 0.05, ...]
        # 第 1 轮：i = 0, v = 0.8  (第 0 个簇的权重)
        # 第 2 轮：i = 1, v = 0.1  (第 1 个簇的权重)
        pi = pi.detach().cpu().numpy()
        for i, v in enumerate(pi):  # enumerate(pi): 同时拿到索引 i（第几个簇）和数值 v（该簇的权重）
            self.writer.add_scalar(f"Pi/pi_{i}", v, epoch)

    def close(self):
        self.writer.close() # 确保缓冲区里的最后一点数据都被完整地写进硬盘
        
        
    # 离线加载 后续可能需要 待完成！！！


# ============================================================
# 模型保存 / 加载
# ============================================================

def save_checkpoint(model, optimizer, epoch, exp_dir, filename="checkpoint.pth"):
    """
    保存模型与优化器状态
    """
    path = os.path.join(exp_dir, filename)
    # 例如 Graduation/experiments/20260402_1700/checkpoint.pth
    torch.save({    # torch.save 将一个 Python 字典 序列化并写入硬盘
        "epoch": epoch,
        "model_state": model.state_dict(),   # state_dict() 是一个特殊的字典，里面存的是每一层神经网络的权重和偏置的数值。
        "optimizer_state": optimizer.state_dict()
    }, path)  # path：把这个字典存到刚才拼好的那个位置


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    加载checkpoint，恢复训练
    返回：恢复的epoch号（从下一个epoch继续）
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint.get("epoch", 0) + 1
    print(f"[INFO] 已恢复 checkpoint，epoch {checkpoint['epoch']} → 从 {start_epoch} 继续")
    return start_epoch



# ============================================================
# π 分布静态图保存
# ============================================================

def save_pi_distribution(pi, epoch, exp_dir):
    """
    保存 π_k 分布柱状图
    """
    pi = pi.detach().cpu().numpy()

    plt.figure(figsize=(6, 4))  # 长 6 英寸、宽 4 英寸
    plt.bar(np.arange(len(pi)), pi)  # np.arange(len(pi)) 生成一个从0到len(pi)-1的整数数组，作为x轴坐标；pi作为y轴高度
    plt.xlabel("Cluster index k")
    plt.ylabel("π_k")
    plt.title(f"Cluster weight distribution (Epoch {epoch})")
    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域

    path = os.path.join(exp_dir, f"pi_epoch_{epoch:03d}.png")  # 03d 是一个格式化指令：d：表示这是一个整数，03：表示这个整数至少占3位，不足的部分用0填充。例如，如果epoch=5，那么{epoch:03d}会被格式化为"005"。
    # 例如 Graduation/experiments/20260402_1700/pi_epoch_005.png
    plt.savefig(path)   # 将图片保存到之前创建的实验目录下
    plt.close()  # 每次绘图后必须关闭画布


def count_effective_clusters(pi, threshold=1e-3):
    """
    统计 π_k > threshold 的有效簇数
    """
    pi = pi.detach().cpu()
    return int((pi > threshold).sum().item())

