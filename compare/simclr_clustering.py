# compare/simclr_clustering.py
# SimCLR + 聚类头 二阶段方案（阶段2）
#
# 使用方法：
#   $env:PYTHONPATH="."
#
#   从头训练（需要先跑完 simclr_pretrain.py）：
#   python -m compare.simclr_clustering \
#       --simclr_ckpt experiments/simclr_xxx/checkpoint.pth \
#       --epochs 100
#
#   从聚类checkpoint续训：
#   python -m compare.simclr_clustering \
#       --simclr_ckpt experiments/simclr_v1/checkpoint.pth \
#       --resume experiments/simclr_v1_cluster/cluster_checkpoint_epoch50.pth \
#       --epochs 150
#
#   每10轮自动保存完整 checkpoint，支持断点续训。
#   最终输出 cluster_final.pth（聚类头权重），供 comparison.py 评估。

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from datetime import datetime

from src.dataset import GBMDataset
from src.model import MultiViewEncoder, ClusteringHead
from src.utils import (
    make_experiment_dir,
    TBLogger,
    count_effective_clusters,
    save_pi_distribution,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 诊断函数
# ============================================================

def run_cluster_diagnostic(model, encoder, loader, device, epoch, output_dir):
    """在训练过程中对聚类结果做诊断"""
    try:
        from umap import UMAP
        HAS_UMAP = True
    except ImportError:
        HAS_UMAP = False

    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    from sklearn.decomposition import PCA
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import json

    model.eval()
    encoder.eval()
    z_list, q_list = [], []

    with torch.no_grad():
        for x, mask, modality_mask, pids in loader:
            x = x.to(device)
            modality_mask = modality_mask.to(device)
            z = encoder(x, modality_mask)
            q, _, _, _, _, _, _ = model(z, epoch=999)
            z_list.append(z.cpu().numpy())
            q_list.append(q.cpu().numpy())

    z_all = np.concatenate(z_list, axis=0)
    q_all = np.concatenate(q_list, axis=0)
    labels = q_all.argmax(axis=1)

    epoch_dir = os.path.join(output_dir, f"epoch_{epoch:03d}")
    os.makedirs(epoch_dir, exist_ok=True)

    metrics = {}
    try:
        metrics["silhouette"] = float(silhouette_score(z_all, labels))
        metrics["davies_bouldin"] = float(davies_bouldin_score(z_all, labels))
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(z_all, labels))
    except Exception:
        pass

    confidence = q_all.max(axis=1)
    metrics["confidence_mean"] = float(confidence.mean())
    metrics["confidence_std"] = float(confidence.std())

    unique, counts = np.unique(labels, return_counts=True)
    metrics["n_clusters"] = int(len(unique))
    metrics["cluster_distribution"] = {int(u): int(c) for u, c in zip(unique, counts)}

    with open(os.path.join(epoch_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"[Diagnostic @ Epoch {epoch:03d}] clusters={metrics['n_clusters']}, "
          f"sil={metrics.get('silhouette', 'N/A')}, conf={metrics['confidence_mean']:.4f}")

    # PCA
    z_2d = PCA(n_components=2).fit_transform(z_all)
    plt.figure(figsize=(8, 6))
    plt.scatter(z_2d[:, 0], z_2d[:, 1], s=5, c=labels, cmap='tab10')
    plt.title(f"SimCLR+Cluster PCA (epoch {epoch})")
    plt.colorbar(label="Cluster")
    plt.tight_layout()
    plt.savefig(os.path.join(epoch_dir, "pca_z.png"), dpi=150)
    plt.close()

    # UMAP
    if HAS_UMAP:
        try:
            reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
            z_umap = reducer.fit_transform(z_all)
            plt.figure(figsize=(8, 6))
            plt.scatter(z_umap[:, 0], z_umap[:, 1], s=5, c=labels, cmap='tab10')
            plt.title(f"SimCLR+Cluster UMAP (epoch {epoch})")
            plt.colorbar(label="Cluster")
            plt.tight_layout()
            plt.savefig(os.path.join(epoch_dir, "umap_z.png"), dpi=150)
            plt.close()
        except Exception:
            pass

    model.train()


# ============================================================
# 检查点保存 / 加载
# ============================================================

def save_cluster_checkpoint(model, optimizer, epoch, exp_dir, filename):
    """保存聚类头 + 优化器状态"""
    path = os.path.join(exp_dir, filename)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, path)


def load_cluster_checkpoint(checkpoint_path, model, optimizer=None):
    """加载聚类检查点，返回恢复的epoch号"""
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    start_epoch = ckpt.get("epoch", 0) + 1
    print(f"[INFO] 恢复聚类检查点，epoch {ckpt['epoch']} → 从 {start_epoch} 继续")
    return start_epoch


# ============================================================
# 训练主函数
# ============================================================

def train_clustering(
    simclr_checkpoint,
    data_dir="data",
    output_dir=None,
    epochs=100,
    batch_size=8,
    lr=1e-3,
    beta=1.0,
    resume_checkpoint=None,
    log_interval=1,
    diagnostic_every=None,
):
    """
    SimCLR + 聚类头 二阶段训练（阶段2）

    参数:
        simclr_checkpoint: SimCLR预训练的encoder checkpoint路径
        data_dir:          数据目录
        output_dir:        输出目录（默认自动生成）
        epochs:            训练轮数
        batch_size:        批次大小
        lr:                聚类头学习率
        beta:              L_ac 权重
        resume_checkpoint: 聚类头检查点路径（续训用）
        log_interval:      日志输出间隔
        diagnostic_every:  诊断输出间隔（None=关闭）
    """
    print("=" * 60)
    print("SimCLR + 聚类头 二阶段训练")
    print(f"  设备: {DEVICE}")
    print(f"  SimCLR checkpoint: {simclr_checkpoint}")
    print(f"  Epochs: {epochs}, Batch: {batch_size}, LR: {lr}, Beta: {beta}")
    print("=" * 60)

    # ---- 1. 加载数据 ----
    dataset = GBMDataset(data_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # ---- 2. 加载冻结的 SimCLR Encoder ----
    encoder = MultiViewEncoder(num_modalities=4, feature_dim=256).to(DEVICE)
    ckpt = torch.load(simclr_checkpoint, map_location=DEVICE, weights_only=False)
    encoder.load_state_dict(ckpt["encoder_state"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    print(f"[OK] SimCLR Encoder 已加载并冻结 (feature_dim=256)")

    # ---- 3. 创建聚类头 ----
    model = ClusteringHead(feature_dim=256, max_K=10, beta=beta).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ---- 4. 实验目录 & 日志 ----
    start_epoch = 0
    if resume_checkpoint is not None and os.path.exists(resume_checkpoint):
        start_epoch = load_cluster_checkpoint(resume_checkpoint, model, optimizer)
        output_dir = os.path.dirname(resume_checkpoint)

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 从 simclr_ckpt 路径提取SimCLR实验名
        simclr_name = os.path.basename(os.path.dirname(simclr_checkpoint))
        output_dir = os.path.join("experiments", f"{simclr_name}_cluster_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    tb_logger = TBLogger(log_dir=output_dir)
    scaler = GradScaler('cuda')

    # 诊断目录
    diag_base_dir = os.path.join("diagnostics", os.path.basename(output_dir))
    os.makedirs(diag_base_dir, exist_ok=True)

    print(f"  输出目录: {output_dir}")
    print(f"  起始 epoch: {start_epoch}")
    print("=" * 60)

    # ---- 5. 训练循环 ----
    for epoch in range(start_epoch, epochs):
        model.train()

        total = {"total": 0.0, "intra": 0.0, "inter": 0.0, "ac": 0.0, "entropy": 0.0}

        # τ 退火
        model.nonparamK.anneal_tau(epoch)

        for x, mask, modality_mask, pids in loader:
            x = x.to(DEVICE)
            modality_mask = modality_mask.to(DEVICE)

            # 冻结Encoder提取特征（不计算梯度，节省显存）
            with torch.no_grad():
                z = encoder(x, modality_mask)

            # 聚类头前向
            with autocast('cuda'):
                q, clean_pi, loss, L_intra, L_inter, L_ac, L_entropy = model(z, epoch)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            total["total"] += loss.item()
            total["intra"] += L_intra.item()
            total["inter"] += L_inter.item()
            total["ac"] += L_ac.item()
            total["entropy"] += L_entropy.item()

        # 平均
        for k in total:
            total[k] /= len(loader)

        # TensorBoard
        tb_logger.log_losses(total, epoch)

        eff_k = count_effective_clusters(clean_pi)
        tb_logger.writer.add_scalar("Cluster/effective_K", eff_k, epoch)
        tb_logger.log_pi(clean_pi, epoch)

        # π 分布图
        if epoch % 5 == 0 or epoch == epochs - 1:
            save_pi_distribution(clean_pi, epoch, output_dir)

        # Checkpoint
        if epoch % 10 == 0 or epoch == epochs - 1:
            save_cluster_checkpoint(model, optimizer, epoch, output_dir,
                                    filename="cluster_checkpoint.pth")
        if epoch >= 40 and epoch % 10 == 0:
            save_cluster_checkpoint(model, optimizer, epoch, output_dir,
                                    filename=f"cluster_checkpoint_epoch{epoch}.pth")

        # 诊断
        if diagnostic_every is not None and (epoch % diagnostic_every == 0 or epoch == epochs - 1):
            run_cluster_diagnostic(model, encoder, loader, DEVICE, epoch, diag_base_dir)

        # 控制台
        if epoch % log_interval == 0:
            print(
                f"[Epoch {epoch:03d}] "
                f"L_total={total['total']:.4f} | "
                f"L_intra={total['intra']:.4f} | "
                f"L_inter={total['inter']:.4f} | "
                f"L_ac={total['ac']:.4f} | "
                f"L_ent={total['entropy']:.4f} | "
                f"K_eff={eff_k}"
            )

    # ---- 6. 保存最终模型 ----
    final_path = os.path.join(output_dir, "cluster_final.pth")
    torch.save({
        "model_state": model.state_dict(),
        "epoch": epochs - 1,
    }, final_path)

    tb_logger.close()
    print(f"\n[完成] 聚类头训练结束")
    print(f"  最终模型: {final_path}")
    print(f"  输出目录: {output_dir}")

    return output_dir


# ============================================================
# 命令行入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR + 聚类头 二阶段训练")
    parser.add_argument("--simclr_ckpt", type=str, required=True,
                        help="SimCLR预训练的encoder checkpoint路径")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--resume", type=str, default=None,
                        help="从聚类头checkpoint续训")
    parser.add_argument("--diagnostic_every", type=int, default=None)

    args = parser.parse_args()

    train_clustering(
        simclr_checkpoint=args.simclr_ckpt,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        beta=args.beta,
        resume_checkpoint=args.resume,
        diagnostic_every=args.diagnostic_every,
    )
