# compare/simclr_pretrain.py
# SimCLR 对比学习预训练基线（方案A），支持断点续训
#
# 使用方法：
#   $env:PYTHONPATH="."
#
#   从头训练（推荐 batch_size=4，负样本充足）：
#   python -m compare.simclr_pretrain --epochs 100
#
#   分段训练（如先训70轮，再续训到150轮）：
#   python -m compare.simclr_pretrain --epochs 70 --output_dir experiments/simclr_v1
#   python -m compare.simclr_pretrain --epochs 150 \
#       --resume experiments/simclr_v1/checkpoint_epoch70.pth \
#       --output_dir experiments/simclr_v1
#
#   每10轮自动保存完整 checkpoint（encoder + projector + optimizer + scheduler），
#   最终输出 checkpoint.pth（仅 encoder 权重，供 comparison.py 使用）

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from datetime import datetime

from src.dataset import GBMDataset
from src.model import MultiViewEncoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 3D MRI 数据增强（双视图生成）
# ============================================================

class MRI3DBatchedAugmentation:
    """
    全 Batch 向量化并行的 3D MRI 增强（GPU 友好，无 for 循环）。
    输入 [B, C, H, W, D]，返回两个增强视图 v1, v2。
    空间变换在 batch 内共享（通过两次独立调用实现 v1≠v2），
    强度变换通过广播独立作用于每个样本。
    """
    def __init__(self, flip_prob=0.5, scale_range=(0.95, 1.05),
                 noise_std=0.01, intensity_range=(0.95, 1.05)):
        self.flip_prob = flip_prob
        self.scale_range = scale_range
        self.noise_std = noise_std
        self.intensity_range = intensity_range

    def _augment(self, x):
        """内部增强逻辑"""
        B, C, H, W, D = x.shape

        # 1. 随机翻转（全 Batch 并行）
        r = torch.rand(3, device=x.device)
        if r[0] < self.flip_prob:
            x = torch.flip(x, dims=[2])
        if r[1] < self.flip_prob:
            x = torch.flip(x, dims=[3])
        if r[2] < self.flip_prob:
            x = torch.flip(x, dims=[4])

        # 2. 随机缩放（统一作用于全 Batch，跳过微小变化避免不必要插值）
        scale = self.scale_range[0] + torch.rand(1, device=x.device).item() * (self.scale_range[1] - self.scale_range[0])
        if abs(scale - 1.0) > 0.01:
            nH, nW, nD = int(H * scale), int(W * scale), int(D * scale)
            xs = F.interpolate(x, size=(nH, nW, nD), mode='trilinear', align_corners=False)
            if scale < 1.0:
                sh, sw, sd = (H - nH) // 2, (W - nW) // 2, (D - nD) // 2
                out = torch.zeros_like(x)
                out[:, :, sh:sh+nH, sw:sw+nW, sd:sd+nD] = xs
            else:
                sh, sw, sd = (nH - H) // 2, (nW - W) // 2, (nD - D) // 2
                out = xs[:, :, sh:sh+H, sw:sw+W, sd:sd+D]
            x = out

        # 3. 强度扰动（广播机制，无 for 循环）
        noise = torch.randn_like(x) * self.noise_std * (0.5 + torch.rand(1, device=x.device).item())
        scale_factors = torch.empty(B, 1, 1, 1, 1, device=x.device).uniform_(*self.intensity_range)
        return x + noise * scale_factors

    def __call__(self, x):
        """x: [B, C, H, W, D] on GPU，返回两个独立增强视图"""
        v1 = self._augment(x)
        v2 = self._augment(x)
        return v1, v2


# ============================================================
# SimCLR Projection Head
# ============================================================

class ProjectionHead(nn.Module):
    """SimCLR 标准 projection head: encoder_dim → hidden → output"""
    def __init__(self, input_dim=256, hidden_dim=128, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z):
        return F.normalize(self.net(z), p=2, dim=1)


# ============================================================
# NT-Xent Loss (InfoNCE)
# ============================================================

def nt_xent_loss(z1, z2, temperature=0.5):
    """
    归一化温度缩放交叉熵损失
    z1, z2: [B, D] — 两个视图的投影特征（已L2归一化）
    """
    B = z1.shape[0]
    # 拼接 → [2B, D]
    z = torch.cat([z1, z2], dim=0)
    # 相似度矩阵 [2B, 2B]
    sim = torch.mm(z, z.t()) / temperature
    # 正样本对索引
    pos_idx = torch.cat([torch.arange(B) + B, torch.arange(B)]).to(z.device)
    # 去掉自己和正样本的 mask
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, float('-inf'))
    # 交叉熵
    labels = pos_idx
    loss = F.cross_entropy(sim, labels)
    return loss


# ============================================================
# 训练主函数
# ============================================================

def train_simclr(data_dir="data",
                 output_dir=None,
                 epochs=100,
                 batch_size=4,
                 lr=1e-4,
                 temperature=0.5,
                 feature_dim=256,
                 resume_from=None,
                 log_interval=10):
    """SimCLR 对比预训练，支持断点续训"""

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("experiments", f"simclr_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    dataset = GBMDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=2 if batch_size >= 2 else 0,
                        pin_memory=True)

    # 构建模型
    encoder = MultiViewEncoder(num_modalities=4, feature_dim=feature_dim).to(DEVICE)
    projector = ProjectionHead(input_dim=feature_dim, hidden_dim=128, output_dim=64).to(DEVICE)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(projector.parameters()),
        lr=lr, weight_decay=1e-5
    )

    # 数据增强（医学 MRI 专用轻量增强：flip + 缩放 + 小幅平移 + 强度扰动）
    # 注意：不使用大角度旋转（破坏解剖结构语义）
    augment = MRI3DBatchedAugmentation(
        flip_prob=0.5,
        scale_range=(0.95, 1.05),   # 医学图像不宜过大缩放
        noise_std=0.01,             # MRI 本身有噪声，不宜再加过多噪声
        intensity_range=(0.95, 1.05),  # MRI 灰度统计不宜被过度扰动
    )

    start_epoch = 0
    best_loss = float('inf')

    # ---- 断点续训 ----
    torch.backends.cudnn.benchmark = True  # cuDNN自动选择最优卷积算法
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler('cuda')  # AMP 混合精度

    if resume_from is not None and os.path.exists(resume_from):
        print(f"\n[断点续训] 加载: {resume_from}")
        ckpt = torch.load(resume_from, map_location=DEVICE, weights_only=False)
        encoder.load_state_dict(ckpt["encoder_state"])
        if "projector_state" in ckpt:
            projector.load_state_dict(ckpt["projector_state"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        else:
            # 兼容旧 checkpoint：手动步进
            for _ in range(start_epoch):
                scheduler.step()
        start_epoch = ckpt.get("epoch", -1) + 1
        best_loss = ckpt.get("loss", float('inf'))
        print(f"  恢复自 epoch {ckpt.get('epoch', '?')}，将从 epoch {start_epoch} 继续")
    elif resume_from is not None:
        print(f"[WARNING] resume 路径不存在: {resume_from}，从头训练")

    print("=" * 60)
    print("SimCLR 对比预训练")
    print(f"  数据: {len(dataset)} 患者 | Epochs: {epochs} | 起始: {start_epoch}")
    print(f"  Batch: {batch_size}, LR: {lr}, Temp: {temperature}")
    print(f"  输出: {output_dir} | 设备: {DEVICE}")
    print("=" * 60)

    total_batches = len(loader)

    # 训练
    for epoch in range(start_epoch, epochs):
        encoder.train()
        projector.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_i, (x, mask, modality_mask, pids) in enumerate(loader):
            B = x.shape[0]
            if B < 2:
                continue  # NT-Xent 需要至少 batch_size≥2

            # 增强在 GPU 上做（全 Batch 向量化并行，无 for 循环）
            x = x.to(DEVICE)
            v1, v2 = augment(x)

            modality_mask = modality_mask.to(DEVICE)

            # 前向（AMP 混合精度）
            with autocast('cuda'):
                z1 = encoder(v1, modality_mask)
                z2 = encoder(v2, modality_mask)
                h1 = projector(z1)
                h2 = projector(z2)
                loss = nt_xent_loss(h1, h2, temperature=temperature)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(projector.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            del v1, v2, z1, z2, h1, h2

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        print(f"  Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        # 保存最佳模型（含完整状态，可续训）
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "encoder_state": encoder.state_dict(),
                "projector_state": projector.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "loss": avg_loss,
            }, os.path.join(output_dir, "checkpoint_best.pth"))

        # 定期保存完整 checkpoint（每10轮，可续训）
        if epoch % 10 == 0 or epoch == epochs - 1:
            torch.save({
                "encoder_state": encoder.state_dict(),
                "projector_state": projector.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "loss": avg_loss,
            }, os.path.join(output_dir, f"checkpoint_epoch{epoch}.pth"))

    # 保存最终 checkpoint
    final_path = os.path.join(output_dir, "checkpoint.pth")
    torch.save({
        "encoder_state": encoder.state_dict(),
        "epoch": epochs - 1,
        "loss": avg_loss,
    }, final_path)

    print(f"\n[完成] SimCLR 预训练结束")
    print(f"  最佳损失: {best_loss:.4f}")
    print(f"  Encoder 已保存: {final_path}")

    return output_dir


# ============================================================
# 命令行入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SimCLR 对比学习预训练（MRI多模态，支持断点续训）")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--feature_dim", type=int, default=256)
    parser.add_argument("--resume", type=str, default=None,
                        help="从指定 checkpoint 续训（如 checkpoint_epoch70.pth）")

    args = parser.parse_args()

    train_simclr(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        temperature=args.temperature,
        feature_dim=args.feature_dim,
        resume_from=args.resume,
    )
