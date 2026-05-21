# compare/diagnostic.py
# 分阶段诊断实验：自动保存到 compare_results/时间戳/ 目录

# 使用方法：
# 1. 先修改 checkpoint 路径
# 2. 设置 PYTHONPATH：$env:PYTHONPATH="."
# 3. 运行：python -m compare.diagnostic

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from umap import UMAP

from src.dataset import GBMDataset
from src.model import DeepClusteringModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_diagnostic(checkpoint_path, output_dir):
    """运行所有诊断检查"""
    print("加载模型和数据...")
    dataset = GBMDataset("data")
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    model = DeepClusteringModel(num_modalities=4, feature_dim=256, max_K=10).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    epoch = checkpoint.get('epoch', '?')
    print(f"[OK] 模型加载完成: epoch {epoch}")

    # 提取特征
    print("提取特征...")
    z_list, q_list = [], []

    with torch.no_grad():
        for x, mask, modality_mask, pids in loader:
            x = x.to(DEVICE)
            modality_mask = modality_mask.to(DEVICE)
            z, q, _, *_ = model(x, modality_mask, epoch=999)
            z_list.append(z.cpu().numpy())
            q_list.append(q.cpu().numpy())

    z_all = np.concatenate(z_list, axis=0)
    q_all = np.concatenate(q_list, axis=0)
    labels = q_all.argmax(axis=1)
    print(f"[OK] 提取了 {len(z_all)} 个患者的特征")

    # ============ 诊断1: PCA可视化 ============
    print("\n[诊断1a] PCA可视化...")
    z_2d = PCA(n_components=2).fit_transform(z_all)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], s=5, c=labels, cmap='tab10')
    plt.title(f"PCA of z_all (epoch {epoch})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pca_z.png"), dpi=150)
    plt.close()
    print(f"[OK] 已保存: {output_dir}/pca_z.png")

    # ============ 诊断1b: UMAP可视化 ============
    print("\n[诊断1b] UMAP可视化...")
    try:
        reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        z_umap = reducer.fit_transform(z_all)

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(z_umap[:, 0], z_umap[:, 1], s=5, c=labels, cmap='tab10')
        plt.title(f"UMAP of z_all (epoch {epoch})")
        plt.xlabel("UMAP1")
        plt.ylabel("UMAP2")
        plt.colorbar(scatter, label="Cluster")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "umap_z.png"), dpi=150)
        plt.close()
        print(f"[OK] 已保存: {output_dir}/umap_z.png")
    except Exception as e:
        print(f"[WARNING] UMAP失败: {e}")

    # ============ 诊断2: Cluster Center 距离 ============
    print("\n[诊断2] Cluster Center 距离...")
    mu = model.cluster.mu.detach().cpu().numpy()  # [K, 256]
    K = mu.shape[0]

    # 计算两两距离矩阵
    dist_matrix = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            dist_matrix[i, j] = np.linalg.norm(mu[i] - mu[j])

    # 保存距离矩阵为文本
    with open(os.path.join(output_dir, "mu_distances.txt"), "w", encoding="utf-8") as f:
        f.write(f"Cluster Center 两两距离矩阵 (L2 norm)\n")
        f.write(f"Model epoch: {epoch}\n\n")
        header = "i/j    " + "".join([f"{j:>10}" for j in range(K)])
        f.write(header + "\n")
        for i in range(K):
            row = f"{i:4}  " + "".join([f"{dist_matrix[i,j]:>10.4f}" for j in range(K)])
            f.write(row + "\n")

    # 统计量
    dists = [dist_matrix[i, j] for i in range(K) for j in range(i+1, K)]
    stats = f"\n距离统计（排除对角线）:\n  最小: {min(dists):.4f}\n  最大: {max(dists):.4f}\n  均值: {np.mean(dists):.4f}\n  标准差: {np.std(dists):.4f}\n"
    print(stats)
    with open(os.path.join(output_dir, "mu_distances.txt"), "a", encoding="utf-8") as f:
        f.write(stats)

    # 检查是否有太接近的cluster center
    too_close = [(i, j) for i in range(K) for j in range(i+1, K) if dist_matrix[i, j] < 1.0]
    if too_close:
        msg = f"\n[WARNING] 有 {len(too_close)} 对 cluster center 距离 < 1.0:\n"
        for i, j in too_close:
            msg += f"  簇 {i} vs 簇 {j}: {dist_matrix[i,j]:.4f}\n"
        print(msg)
        with open(os.path.join(output_dir, "mu_distances.txt"), "a", encoding="utf-8") as f:
            f.write(msg)
    else:
        print("[OK] 所有 cluster center 距离都 >= 1.0")

    # ============ 诊断3: Q 置信度 ============
    print("\n[诊断3] Q 置信度分析...")
    confidence = q_all.max(axis=1)

    stats = f"""===== Q 置信度分析 =====
置信度均值: {confidence.mean():.4f}
置信度标准差: {confidence.std():.4f}
置信度最小值: {confidence.min():.4f}
置信度最大值: {confidence.max():.4f}

分布统计:
"""
    print(stats)
    with open(os.path.join(output_dir, "q_confidence.txt"), "w", encoding="utf-8") as f:
        f.write(f"Model epoch: {epoch}\n")
        f.write(stats)

        bins = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
        for i in range(len(bins) - 1):
            count = ((confidence >= bins[i]) & (confidence < bins[i+1])).sum()
            pct = count / len(confidence) * 100
            line = f"  [{bins[i]:.1f}, {bins[i+1]:.1f}): {count:4d} ({pct:.1f}%)\n"
            print(line)
            f.write(line)

        high_conf = (confidence > 0.9).sum()
        low_conf = (confidence < 0.5).sum()
        summary = f"\n高置信度 (>0.9): {high_conf} ({high_conf/len(confidence)*100:.1f}%)\n低置信度 (<0.5): {low_conf} ({low_conf/len(confidence)*100:.1f}%)\n"
        print(summary)
        f.write(summary)

    # ============ 诊断4: 聚类分布概览 ============
    print("\n[诊断4] 聚类分布概览...")
    unique_labels = np.unique(labels)
    overview = f"===== 聚类结果 =====\n总患者数: {len(labels)}\n有效簇数: {len(unique_labels)}\n"
    print(overview)
    with open(os.path.join(output_dir, "cluster_overview.txt"), "w", encoding="utf-8") as f:
        f.write(f"Model epoch: {epoch}\n")
        f.write(overview)
        for l in unique_labels:
            count = (labels == l).sum()
            line = f"  簇 {l}: {count} 人 ({count/len(labels)*100:.1f}%)\n"
            print(line)
            f.write(line)

    print(f"\n[完成] 所有诊断结果已保存到: {output_dir}")


if __name__ == "__main__":
    # 【需要修改】checkpoint路径
    checkpoint_path = "experiments/20260509_223937/checkpoint.pth"

    # 从checkpoint路径中提取时间戳作为输出目录名
    # 格式：experiments/20260509_171534/checkpoint.pth → 20260509_171534
    checkpoint_dir = os.path.dirname(checkpoint_path)  # .../experiments/20260509_171534
    exp_name = os.path.basename(checkpoint_dir)        # 20260509_171534
    output_dir = os.path.join("Graduation", "compare_results", exp_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"输出目录: {output_dir}")
    run_diagnostic(checkpoint_path, output_dir)