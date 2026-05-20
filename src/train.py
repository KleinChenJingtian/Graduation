# 训练脚本

# src/train.py

import torch
import os
import json
import random
import numpy as np
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from datetime import datetime

from src.dataset import GBMDataset
from src.model import DeepClusteringModel
from src.utils import (
    make_experiment_dir,
    TBLogger,
    save_checkpoint,
    save_pi_distribution,
    count_effective_clusters,
    load_checkpoint
)


# ============================================================
# 可复现性设置
# ============================================================

def set_seed(seed=42):
    """设置随机种子，最大程度保证训练可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 训练中诊断函数（每 N 轮输出 silhouette/UMAP 等）
# ============================================================

def run_diagnostic(model, loader, device, epoch, output_dir):
    """在训练过程中对当前模型做诊断：提取特征 + 计算指标 + 保存图片"""
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

    model.eval()
    z_list, q_list = [], []

    with torch.no_grad():
        for x, mask, modality_mask, pids in loader:
            x = x.to(device)
            modality_mask = modality_mask.to(device)
            z, q, _, *_ = model(x, modality_mask, epoch=999)
            z_list.append(z.cpu().numpy())
            q_list.append(q.cpu().numpy())

    z_all = np.concatenate(z_list, axis=0)
    q_all = np.concatenate(q_list, axis=0)
    labels = q_all.argmax(axis=1)

    # 创建当前epoch目录
    epoch_dir = os.path.join(output_dir, f"epoch_{epoch:03d}")
    os.makedirs(epoch_dir, exist_ok=True)

    # 1. 指标计算
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

    # 保存 metrics.json
    with open(os.path.join(epoch_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"[Diagnostic @ Epoch {epoch:03d}] clusters={metrics['n_clusters']}, "
          f"sil={metrics.get('silhouette', 'N/A')}, conf={metrics['confidence_mean']:.4f}")

    # 2. PCA 可视化
    z_2d = PCA(n_components=2).fit_transform(z_all)
    plt.figure(figsize=(8, 6))
    plt.scatter(z_2d[:, 0], z_2d[:, 1], s=5, c=labels, cmap='tab10')
    plt.title(f"PCA (epoch {epoch})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(label="Cluster")
    plt.tight_layout()
    plt.savefig(os.path.join(epoch_dir, "pca_z.png"), dpi=150)
    plt.close()

    # 3. UMAP 可视化
    if HAS_UMAP:
        try:
            reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
            z_umap = reducer.fit_transform(z_all)
            plt.figure(figsize=(8, 6))
            plt.scatter(z_umap[:, 0], z_umap[:, 1], s=5, c=labels, cmap='tab10')
            plt.title(f"UMAP (epoch {epoch})")
            plt.xlabel("UMAP1")
            plt.ylabel("UMAP2")
            plt.colorbar(label="Cluster")
            plt.tight_layout()
            plt.savefig(os.path.join(epoch_dir, "umap_z.png"), dpi=150)
            plt.close()
        except Exception:
            pass

    model.train()


def train(
    data_dir,
    device,
    batch_size=4,
    epochs=50,
    lr=1e-5,
    beta=1.0,
    gamma=0.1,
    log_interval=1,  # 每隔多少轮输出一次日志
    resume_checkpoint=None,  # 可选，从checkpoint恢复训练
    diagnostic_every=None  # 每N轮输出诊断，默认None关闭
):
    # ============================================================
    # 设备信息
    # ============================================================
    print("Using device:", device)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    # 【新增】：固定随机种子，最大程度保证可复现
    set_seed(42)

    # ============================================================
    # 准备数据
    # ============================================================

    dataset = GBMDataset(data_dir)
    loader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 2,  # 使用 2 个子进程来加载数据，可以加快数据预处理的速度，特别是当数据集较大时。
        pin_memory = True # 这个参数告诉 DataLoader 在将数据加载到 GPU 之前，先把数据放到锁页内存中。这样可以加速数据从 CPU 到 GPU 的传输，尤其是在使用 CUDA 时。
    )

    # ============================================================
    # 实验目录 & 日志
    # ============================================================

    # 不在这里创建，等resume逻辑确定路径后再说
    exp_dir = None

    # ============================================================
    # 模型 & 优化器
    # ============================================================

    model = DeepClusteringModel(
        num_modalities = 4,
        feature_dim = 256,
        max_K = 10,
        beta = beta,
        gamma = gamma
    ).to(device)

    # 【修复】：为不同参数组设置不同学习率
    # a_k 和 logits_temp 使用更大的学习率，能迅速跟上特征学习
    # CNN/Transformer 使用较小学习率，保持稳定更新
    params_ak = [model.nonparamK.a_k]
    params_temp = [model.logits_temp]
    special_param_ids = set(map(id, params_ak + params_temp))
    base_params = [p for p in model.parameters() if id(p) not in special_param_ids]

    optimizer = torch.optim.Adam([
        {'params': base_params, 'lr': lr},         # 1e-5 CNN/Transformer
        {'params': params_ak, 'lr': 1e-2},          # a_k 独立学习率（1e-2大步追赶）
        {'params': params_temp, 'lr': 1e-2}          # logits_temp 独立学习率（提升）
    ])

    # 【新增】：从checkpoint恢复训练
    start_epoch = 0
    if resume_checkpoint is not None:
        start_epoch = load_checkpoint(resume_checkpoint, model, optimizer)
        exp_dir = os.path.dirname(resume_checkpoint)  # 恢复时用同一个目录

    # 【修复】：非resume训练时，exp_dir为None，需要创建
    if exp_dir is None:
        exp_dir = make_experiment_dir()

    # 【修复】：resume后exp_dir才确定，必须在这里初始化TBLogger
    tb_logger = TBLogger(log_dir=exp_dir)

    # 【新增】：AMP 混合精度
    scaler = GradScaler('cuda')

    # 【新增】：诊断目录（每个exp独立）
    diag_base_dir = os.path.join("Graduation", "diagnostics", os.path.basename(exp_dir))
    os.makedirs(diag_base_dir, exist_ok=True)

    # ============================================================
    # 训练循环
    # ============================================================

    for epoch in range(start_epoch, epochs):
        model.train()

        total = {
            "total": 0.0,
            "intra": 0.0,
            "inter": 0.0,
            "ac": 0.0,
            "var": 0.0,
            "vicreg": 0.0
        }

        # 【改进】：使用 sigmoid 平滑预热 alpha_inter（每轮调用）
        model.get_alpha_inter(epoch)

        # τ 退火：anneal_tau 内部检查 epoch >= tau_anneal_start 才实际退火
        model.nonparamK.anneal_tau(epoch)

        for x, mask, modality_mask, pid in loader:
            x = x.to(device)
            modality_mask = modality_mask.to(device)

            # 【AMP】：使用 autocast 减少显存
            with torch.amp.autocast('cuda'):
                z, q, clean_pi, loss, L_intra, L_inter, L_ac, L_var, L_vicreg = model(x, modality_mask, epoch)

            optimizer.zero_grad() # 在计算新的一批病人前，必须把之前的梯度（误差方向）清零
            scaler.scale(loss).backward()   # 计算 loss 对模型每一个参数（权重）的梯度
            scaler.unscale_(optimizer)    
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0) # 如果某个参数的梯度特别大（比如突然跳变到几千），它会强行把梯度压缩到 5.0 以内。这防止了模型更新太猛导致”扯着跨”（权重爆炸）
            scaler.step(optimizer)   # 根据计算得到的梯度更新模型参数
            scaler.update()

            total["total"] += loss.item() 
            total["intra"] += L_intra.item()
            total["inter"] += L_inter.item()
            total["ac"] += L_ac.item()
            total["var"] += L_var.item()
            total["vicreg"] += L_vicreg.item()

        # 平均 loss
        for k in total:
            total[k] /= len(loader)


        # TensorBoard 记录 loss
        tb_logger.log_losses(total, epoch)

        # 有效簇数 & π_k 分布（使用model返回的clean_pi，无噪声）
        eff_k = count_effective_clusters(clean_pi)
        tb_logger.writer.add_scalar("Cluster/effective_K", eff_k, epoch)
        tb_logger.log_pi(clean_pi, epoch)  # 用clean_pi记录，避免Gumbel噪声干扰曲线


        # 保存 π 分布图（每5轮+最后1轮）
        if epoch % 5 == 0 or epoch == epochs - 1:
            save_pi_distribution(clean_pi, epoch, exp_dir)

        # 保存 checkpoint（每10轮+最后1轮）
        if epoch % 10 == 0 or epoch == epochs - 1:
            save_checkpoint(model, optimizer, epoch, exp_dir)

        # 【新增】：诊断输出（每 diagnostic_every 轮）
        # 暂时禁用：Windows上 DataLoader+UMAP+multiprocessing 容易死锁
        # 训练完成后用 python -m compare.diagnostic 手动诊断
        # if diagnostic_every is not None and (epoch % diagnostic_every == 0 or epoch == epochs - 1):
        #     run_diagnostic(model, loader, device, epoch, diag_base_dir)

        # 控制台输出
        if epoch % log_interval == 0:            
            print(
                f"[Epoch {epoch:03d}] "
                f"L_total={total['total']:.4f} | "
                f"L_intra={total['intra']:.4f} | "
                f"L_inter={total['inter']:.4f} | "
                f"L_ac={total['ac']:.4f} | "
                f"L_var={total['var']:.4f} | "
                f"L_vic={total['vicreg']:.4f} | "
                f"K_eff={eff_k}"
            )

    tb_logger.close()
    
