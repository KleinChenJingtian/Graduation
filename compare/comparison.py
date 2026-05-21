# compare/comparison.py
# 聚类方法对比（SimCLR + 聚类头 二阶段方案）
# 对比：Ours / DEC / IDEC-Var / K-Means / GMM / SimCLR+KMeans
# 输出：内部指标 + Bootstrap稳定性 + kNN一致性 + π分布 + UMAP可视化 + KM曲线
#
# ============================================================
# 完整流程（在项目根目录下执行）
# ============================================================
#
# ---- 第〇步：生成临床数据（只需执行一次）----
#   python match_clinical.py
#
# ---- 第一步：SimCLR 对比预训练（阶段1）----
#   $env:PYTHONPATH="."
#   python -m compare.simclr_pretrain --epochs 100 --batch_size 4
#
# ---- 第二步：聚类头训练（阶段2）----
#   python -m compare.simclr_clustering `
#       --simclr_ckpt experiments/simclr_xxx/checkpoint.pth `
#       --epochs 100 --lr 1e-3
#
# ---- 第三步：全方法对比评估 ----
#   （1）仅聚类对比（无临床数据）：
#   python -m compare.comparison `
#       --simclr_checkpoint experiments/simclr_xxx/checkpoint.pth `
#       --simclr_cluster_ckpt experiments/simclr_xxx_cluster_20260521_xxx/cluster_final.pth
#
#   （2）聚类对比 + 生存分析：
#   python -m compare.comparison `
#       --simclr_checkpoint experiments/simclr_xxx/checkpoint.pth `
#       --simclr_cluster_ckpt experiments/simclr_xxx_cluster_20260521_xxx/cluster_final.pth `
#       --clinical clinical_eval.csv
#
# ---- 评估输出（保存在 compare_results/<实验名>/full_comparison/ 下）----
#   comparison_report.json        完整对比报告
#   umap_comparison.png           UMAP 可视化（所有方法并列）
#   cluster_size_distribution.png 簇分布对比柱状图
#   feature_variance_boxplot.png  特征空间方差箱线图
#   pi_vs_q_bar.png              π vs q̄ 分布
#   km_<Method>.png              各方法的 KM 生存曲线

import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    normalized_mutual_info_score, adjusted_rand_score,
)
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

from src.dataset import GBMDataset
from src.model import MultiViewEncoder, ClusteringHead


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 辅助函数
# ============================================================

def align_labels(labels_ref, labels_boot):
    """
    将 labels_boot 对齐到 labels_ref（标签可能相差一个 permutation）
    使用 Hungarian algorithm 最优匹配。
    """
    classes_ref = np.unique(labels_ref)
    classes_boot = np.unique(labels_boot)
    k_ref, k_boot = len(classes_ref), len(classes_boot)

    if k_ref == 0 or k_boot == 0:
        return labels_boot.copy()

    k = max(k_ref, k_boot)
    cost = np.zeros((k, k))
    for i, c1 in enumerate(classes_ref):
        for j, c2 in enumerate(classes_boot):
            cost[i, j] = -np.sum((labels_ref == c1) & (labels_boot == c2))
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {}
    for ri, ci in zip(row_ind, col_ind):
        if ri < k_ref and ci < k_boot:
            mapping[classes_boot[ci]] = classes_ref[ri]
    aligned = np.array([mapping.get(l, l) for l in labels_boot])
    return aligned


def compute_knn_consistency(z_all, labels, k=10):
    """
    计算 kNN 一致性：在特征空间中找到每个点的 k 个最近邻，
    统计邻居中与该点同簇的比例，最终取平均
    """
    nbrs = NearestNeighbors(n_neighbors=k+1, metric="euclidean").fit(z_all)
    _, indices = nbrs.kneighbors(z_all)
    indices = indices[:, 1:]  # 去掉自己
    total = 0.0
    for i in range(len(z_all)):
        neighbor_labels = labels[indices[i]]
        total += np.mean(neighbor_labels == labels[i])
    return total / len(z_all)


def align_pi_q(pi, q_mean):
    """将 pi 和 q_mean 按概率大小排序对齐"""
    pi_sorted_idx = np.argsort(pi)[::-1]
    q_sorted_idx = np.argsort(q_mean)[::-1]
    return pi[pi_sorted_idx], q_mean[q_sorted_idx]


def run_dec(z_all, n_clusters, epochs=30, alpha=1.0, lr=0.01, device=DEVICE):
    """DEC 优化（可学习簇中心，PyTorch版本）"""
    z_tensor = torch.tensor(z_all, dtype=torch.float32, device=device)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(z_all)
    centers = torch.nn.Parameter(torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam([centers], lr=lr)

    for _ in range(epochs):
        optimizer.zero_grad()
        dist2 = torch.sum((z_tensor.unsqueeze(1) - centers.unsqueeze(0)) ** 2, dim=2)
        q = (1.0 + dist2 / alpha) ** (-(alpha + 1.0) / 2.0)
        q = q / torch.sum(q, dim=1, keepdim=True)
        with torch.no_grad():
            p = q ** 2 / torch.sum(q, dim=0, keepdim=True)
            p = p / torch.sum(p, dim=1, keepdim=True)
        loss = torch.mean(torch.sum(p * torch.log(p / (q + 1e-10)), dim=1))
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        dist2 = torch.sum((z_tensor.unsqueeze(1) - centers.unsqueeze(0)) ** 2, dim=2)
        q = (1.0 + dist2 / alpha) ** (-(alpha + 1.0) / 2.0)
        q = q / torch.sum(q, dim=1, keepdim=True)
        labels = q.argmax(dim=1).cpu().numpy()
    return labels


def run_idec_var(z_all, n_clusters, epochs=30, alpha=1.0, lr=0.01, var_weight=0.1, device=DEVICE):
    """
    IDEC 方差保留变体：
    - 在 DEC 优化目标中加入特征空间方差损失
    - 适配无解码器架构（用 L_var 替代解码器重构损失）
    """
    z_tensor = torch.tensor(z_all, dtype=torch.float32, device=device)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(z_all)
    centers = torch.nn.Parameter(torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam([centers], lr=lr)

    for _ in range(epochs):
        optimizer.zero_grad()
        dist2 = torch.sum((z_tensor.unsqueeze(1) - centers.unsqueeze(0)) ** 2, dim=2)
        q = (1.0 + dist2 / alpha) ** (-(alpha + 1.0) / 2.0)
        q = q / torch.sum(q, dim=1, keepdim=True)
        with torch.no_grad():
            p = q ** 2 / torch.sum(q, dim=0, keepdim=True)
            p = p / torch.sum(p, dim=1, keepdim=True)
        loss_kl = torch.mean(torch.sum(p * torch.log(p / (q + 1e-10)), dim=1))
        # 方差保留项
        var_loss = torch.mean(torch.var(z_tensor, dim=0))
        loss = loss_kl + var_weight * var_loss
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        dist2 = torch.sum((z_tensor.unsqueeze(1) - centers.unsqueeze(0)) ** 2, dim=2)
        q = (1.0 + dist2 / alpha) ** (-(alpha + 1.0) / 2.0)
        q = q / torch.sum(q, dim=1, keepdim=True)
        labels = q.argmax(dim=1).cpu().numpy()
    return labels


# ============================================================
# 主函数
# ============================================================

def run_comparison(
    simclr_checkpoint,
    simclr_cluster_ckpt,
    data_dir="data",
    clinical_path=None,
    output_dir=None,
    n_clusters=None,
    n_bootstrap=5,
    bootstrap_ratio=0.8,
    k_neighbors=10,
    random_seed=42
):
    """
    SimCLR 二阶段方案对比：Ours (Two-Stage) vs DEC / IDEC / KMeans / GMM / SimCLR+KMeans
    所有方法统一使用 SimCLR 特征空间 z_simclr。
    """
    np.random.seed(random_seed)

    # --- 输出目录 ---
    exp_name = os.path.basename(os.path.dirname(simclr_cluster_ckpt))
    if output_dir is None:
        output_dir = os.path.join("compare_results", exp_name, "full_comparison")
    os.makedirs(output_dir, exist_ok=True)

    # --- 加载数据 ---
    print("=" * 60)
    print("加载数据...")
    dataset = GBMDataset(data_dir)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    # --- 加载 SimCLR Encoder（冻结）---
    print("\n加载 SimCLR 预训练编码器...")
    simclr_encoder = MultiViewEncoder(num_modalities=4, feature_dim=256).to(DEVICE)
    simclr_ckpt = torch.load(simclr_checkpoint, map_location=DEVICE, weights_only=False)
    simclr_encoder.load_state_dict(simclr_ckpt["encoder_state"])
    simclr_encoder.eval()
    print(f"[OK] SimCLR 编码器加载完成: epoch {simclr_ckpt.get('epoch', '?')}")

    # --- 加载 ClusteringHead ---
    print("\n加载聚类头...")
    cluster_head = ClusteringHead(feature_dim=256, max_K=10).to(DEVICE)
    ckpt_cl = torch.load(simclr_cluster_ckpt, map_location=DEVICE, weights_only=False)
    cluster_head.load_state_dict(ckpt_cl["model_state"])
    cluster_head.eval()
    cluster_epoch = ckpt_cl.get("epoch", "?")
    print(f"[OK] 聚类头加载完成: epoch {cluster_epoch}")

    # --- 提取 SimCLR 特征 ---
    print("\n提取 SimCLR 特征...")
    z_list, pid_list = [], []
    with torch.no_grad():
        for x, mask, modality_mask, pids in loader:
            x = x.to(DEVICE)
            modality_mask = modality_mask.to(DEVICE)
            z = simclr_encoder(x, modality_mask)
            z_list.append(z.cpu().numpy())
            pid_list.extend(pids)
    z_simclr = np.concatenate(z_list, axis=0)
    print(f"[OK] SimCLR 特征提取完成: {z_simclr.shape}")

    # --- Ours：用聚类头推理 ---
    print("\n推理 Ours (Two-Stage)...")
    with torch.no_grad():
        z_tensor = torch.tensor(z_simclr, dtype=torch.float32, device=DEVICE)
        q_ours, clean_pi, *_ = cluster_head(z_tensor, epoch=999)
        labels_ours = q_ours.argmax(dim=1).cpu().numpy()
        q_mean_ours = q_ours.cpu().numpy().mean(axis=0)
        clean_pi = clean_pi.cpu().numpy()
    print(f"[OK] Ours: {len(np.unique(labels_ours))} 个簇")

    # --- 确定聚类数 ---
    if n_clusters is None:
        n_clusters = len(np.unique(labels_ours))
    print(f"使用簇数 K={n_clusters}")

    # ============================================================
    # 1. 各方法硬标签（统一使用 z_simclr 特征空间）
    # ============================================================
    print("\n" + "=" * 60)
    print("各方法聚类...")

    labels_dec = run_dec(z_simclr, n_clusters, epochs=30)
    labels_idec = run_idec_var(z_simclr, n_clusters, epochs=30, var_weight=0.1)
    labels_kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(z_simclr)
    labels_gmm = GaussianMixture(n_components=n_clusters, random_state=42, covariance_type="full", n_init=5).fit_predict(z_simclr)
    labels_simclr_km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(z_simclr)

    all_labels = {
        "Ours": labels_ours,
        "SimCLR+KMeans": labels_simclr_km,
        "KMeans": labels_kmeans,
        "GMM": labels_gmm,
        "DEC": labels_dec,
        "IDEC-Var": labels_idec,
    }

    for name, labels in all_labels.items():
        unique, counts = np.unique(labels, return_counts=True)
        print(f"  {name}: K={len(unique)}, 分布={dict(zip(unique, counts))}")

    # ============================================================
    # 2. 内部评价指标
    # ============================================================
    print("\n" + "=" * 60)
    print("内部评价指标...")

    internal_results = {}
    for name, labels in all_labels.items():
        try:
            sil = silhouette_score(z_simclr, labels)
            dbi = davies_bouldin_score(z_simclr, labels)
            chi = calinski_harabasz_score(z_simclr, labels)
            internal_results[name] = {"silhouette": float(sil), "davies_bouldin": float(dbi), "calinski_harabasz": float(chi)}
            print(f"  {name}: Sil={sil:.4f}, DBI={dbi:.4f}, CHI={chi:.1f}")
        except Exception as e:
            print(f"  {name}: 计算失败 ({e})")
            internal_results[name] = {"error": str(e)}

    # ============================================================
    # 3. 聚类塌缩现象分析
    # ============================================================
    print("\n" + "=" * 60)
    print("聚类塌缩分析...")

    collapse_results = {}

    def gini_coefficient(counts):
        sorted_counts = np.sort(counts)
        n = len(sorted_counts)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * sorted_counts)) / (n * np.sum(sorted_counts)) - (n + 1) / n

    def cluster_entropy(counts):
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        return entropy / np.log(len(counts))

    def feature_variance_per_dim(z):
        return np.var(z, axis=0)

    overall_feature_var = np.mean(feature_variance_per_dim(z_simclr))
    print(f"  整体特征空间方差: {overall_feature_var:.4f}")

    cluster_labels_for_plot = {}
    cluster_sizes_for_plot = {}

    for name, labels in all_labels.items():
        unique, counts = np.unique(labels, return_counts=True)
        n_cl = len(unique)
        gini = gini_coefficient(counts)
        entropy_norm = cluster_entropy(counts)

        within_cluster_vars = []
        for c in unique:
            mask = labels == c
            if mask.sum() > 1:
                var_c = np.mean(feature_variance_per_dim(z_simclr[mask]))
                within_cluster_vars.append(var_c)
        mean_within_var = np.mean(within_cluster_vars) if within_cluster_vars else 0.0

        max_cluster_ratio = counts.max() / counts.sum()

        collapse_results[name] = {
            "n_clusters": int(n_cl),
            "cluster_sizes": {int(k): int(v) for k, v in zip(unique, counts)},
            "gini_coefficient": float(gini),
            "normalized_entropy": float(entropy_norm),
            "max_cluster_ratio": float(max_cluster_ratio),
            "mean_within_cluster_variance": float(mean_within_var),
        }

        print(f"  {name}: K={n_cl}, 最大簇占比={max_cluster_ratio:.3f}, "
              f"基尼={gini:.4f}, 归一化熵={entropy_norm:.4f}, "
              f"簇内平均方差={mean_within_var:.4f}")
        cluster_labels_for_plot[name] = unique
        cluster_sizes_for_plot[name] = counts

    # ---- 簇样本分布对比柱状图 ----
    n_methods = len(all_labels)
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5))
    if n_methods == 1:
        axes = [axes]
    for ax, name in zip(axes, all_labels.keys()):
        sizes = cluster_sizes_for_plot[name]
        labels_c = cluster_labels_for_plot[name]
        colors = plt.cm.tab10(np.linspace(0, 1, len(sizes)))
        ax.bar(labels_c, sizes, color=colors, alpha=0.8)
        ax.set_title(f"{name}\n(Gini={collapse_results[name]['gini_coefficient']:.3f}, "
                     f"H={collapse_results[name]['normalized_entropy']:.3f})")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Count")
        ax.set_xticks(labels_c)
    plt.suptitle("Cluster Size Distribution Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cluster_size_distribution.png"), dpi=150)
    plt.close()
    print(f"[OK] 簇分布对比图已保存: {output_dir}/cluster_size_distribution.png")

    # ---- 特征空间方差箱线图 ----
    fig, ax = plt.subplots(figsize=(10, 5))
    per_dim_var = feature_variance_per_dim(z_simclr)
    ax.boxplot(per_dim_var, vert=True, showfliers=False)
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="VICReg target (var=1)")
    ax.set_title(f"Feature Space Per-Dimension Variance\n(Overall mean={overall_feature_var:.4f})")
    ax.set_ylabel("Variance")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_variance_boxplot.png"), dpi=150)
    plt.close()
    print(f"[OK] 特征方差箱线图已保存: {output_dir}/feature_variance_boxplot.png")

    # ============================================================
    # 4. Bootstrap 稳定性 (NMI / ARI)
    # ============================================================
    print("\n" + "=" * 60)
    print(f"Bootstrap 稳定性 (NMI/ARI, {n_bootstrap}次, {bootstrap_ratio*100:.0f}%采样)...")

    stability_results = {}
    n_samples = len(z_simclr)
    boot_indices = [np.random.choice(n_samples, int(n_samples * bootstrap_ratio), replace=False)
                    for _ in range(n_bootstrap)]

    for name in all_labels:
        nmi_list, ari_list = [], []
        for idx in boot_indices:
            z_boot = z_simclr[idx]
            labels_ref = all_labels[name][idx]

            if name == "Ours":
                # 聚类头在 bootstrap 特征上推理
                with torch.no_grad():
                    z_t = torch.tensor(z_boot, dtype=torch.float32, device=DEVICE)
                    q_boot, *_ = cluster_head(z_t, epoch=999)
                    labels_boot = q_boot.argmax(dim=1).cpu().numpy()
            elif name == "SimCLR+KMeans":
                labels_boot = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(z_boot)
            elif name == "KMeans":
                labels_boot = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(z_boot)
            elif name == "GMM":
                labels_boot = GaussianMixture(n_components=n_clusters, random_state=42, covariance_type="full", n_init=5).fit_predict(z_boot)
            elif name == "DEC":
                labels_boot = run_dec(z_boot, n_clusters, epochs=30)
            elif name == "IDEC-Var":
                labels_boot = run_idec_var(z_boot, n_clusters, epochs=30, var_weight=0.1)
            else:
                continue

            labels_boot_aligned = align_labels(labels_ref, labels_boot)
            nmi_list.append(normalized_mutual_info_score(labels_ref, labels_boot_aligned))
            ari_list.append(adjusted_rand_score(labels_ref, labels_boot_aligned))

        stability_results[name] = {
            "nmi_mean": float(np.mean(nmi_list)), "nmi_std": float(np.std(nmi_list)),
            "ari_mean": float(np.mean(ari_list)), "ari_std": float(np.std(ari_list)),
        }
        print(f"  {name}: NMI={np.mean(nmi_list):.4f}±{np.std(nmi_list):.4f}, "
              f"ARI={np.mean(ari_list):.4f}±{np.std(ari_list):.4f}")

    # ============================================================
    # 5. kNN 一致性
    # ============================================================
    print("\n" + "=" * 60)
    print(f"kNN 一致性 (k={k_neighbors})...")

    knn_results = {}
    for name, labels in all_labels.items():
        try:
            consistency = compute_knn_consistency(z_simclr, labels, k=k_neighbors)
            knn_results[name] = {"kNN_consistency": float(consistency)}
            print(f"  {name}: kNN一致率={consistency:.4f}")
        except Exception as e:
            print(f"  {name}: kNN计算失败 ({e})")
            knn_results[name] = {"error": str(e)}

    # ============================================================
    # 6. π vs q̄ 分布图 (Ours)
    # ============================================================
    print("\n" + "=" * 60)
    print("π vs q̄ 分布对比 (Ours)...")

    pi_aligned, q_aligned = align_pi_q(clean_pi, q_mean_ours)
    K_eff = np.sum(pi_aligned > 0.01)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(pi_aligned))
    width = 0.35
    ax.bar(x - width/2, pi_aligned, width, label="π (Learned)", color="steelblue", alpha=0.8)
    ax.bar(x + width/2, q_aligned, width, label="q̄ (Average soft assignment)", color="coral", alpha=0.8)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Probability")
    ax.set_title(f"Ours (Two-Stage): π vs q̄ (K_eff={K_eff})")
    ax.legend()
    ax.set_xticks(x)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pi_vs_q_bar.png"), dpi=150)
    plt.close()
    print(f"[OK] π vs q̄ 图已保存: {output_dir}/pi_vs_q_bar.png")

    # ============================================================
    # 7. UMAP 可视化 (所有方法对比，统一 SimCLR 特征空间)
    # ============================================================
    print("\n" + "=" * 60)
    print("UMAP 可视化...")

    try:
        from umap import UMAP
        reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        z_2d = reducer.fit_transform(z_simclr)
        has_umap = True
    except Exception as e:
        print(f"[WARNING] UMAP 失败，降级到 PCA: {e}")
        z_2d = PCA(n_components=2).fit_transform(z_simclr)
        has_umap = False

    n_methods = len(all_labels)
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))
    if n_methods == 1:
        axes = [axes]

    for ax, (name, labels) in zip(axes, all_labels.items()):
        sc = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, s=5, cmap="tab10")
        ax.set_title(name)
        ax.set_xlabel("UMAP1" if has_umap else "PC1")
        ax.set_ylabel("UMAP2" if has_umap else "PC2")
        plt.colorbar(sc, ax=ax, label="Cluster")
    plt.suptitle(f"Clustering Comparison (K={n_clusters}) — SimCLR Feature Space", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "umap_comparison.png"), dpi=150)
    plt.close()
    print(f"[OK] UMAP对比图已保存: {output_dir}/umap_comparison.png")

    # ============================================================
    # 8. KM 生存分析 (所有方法)
    # ============================================================
    km_results = {}
    if clinical_path and os.path.exists(clinical_path):
        print("\n" + "=" * 60)
        print("KM 生存分析...")

        import pandas as pd
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test, multivariate_logrank_test

        clinical_df = pd.read_csv(clinical_path)
        clinical_df["patient_id"] = clinical_df["patient_id"].astype(str)

        for name, labels in all_labels.items():
            label_df = pd.DataFrame({"patient_id": pid_list, "cluster": labels})
            merged = pd.merge(label_df, clinical_df, on="patient_id", how="inner")
            merged = merged.dropna(subset=["time", "event"])

            if len(merged) < 2:
                print(f"  {name}: 临床数据匹配少于2人，跳过")
                continue
            clusters = sorted(merged["cluster"].unique())
            if len(clusters) < 2:
                print(f"  {name}: 有效亚型少于2个，跳过")
                continue

            # Kaplan-Meier 曲线
            fig_k, ax_k = plt.subplots(figsize=(8, 6))
            for c in clusters:
                sub = merged[merged["cluster"] == c]
                kmf = KaplanMeierFitter()
                kmf.fit(sub["time"], sub["event"], label=f"S{c} (n={len(sub)})")
                kmf.plot_survival_function(ax=ax_k)
            ax_k.set_title(f"KM Curve - {name}")
            ax_k.set_xlabel("Time (months)")
            ax_k.set_ylabel("Survival Probability")
            ax_k.legend(loc="best")
            ax_k.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"km_{name}.png"), dpi=150)
            plt.close()

            # Log-rank 检验
            if len(clusters) == 2:
                c1, c2 = clusters
                sub1, sub2 = merged[merged["cluster"] == c1], merged[merged["cluster"] == c2]
                if len(sub1) > 0 and len(sub2) > 0:
                    test = logrank_test(sub1["time"], sub2["time"], sub1["event"], sub2["event"])
                    km_results[name] = {"logrank_p": float(test.p_value), "logrank_stat": float(test.test_statistic), "n_clusters": len(clusters)}
                    print(f"  {name}: Log-rank p={test.p_value:.4f}")
            else:
                mtest = multivariate_logrank_test(merged["time"], merged["cluster"], merged["event"])
                pairwise = {}
                for i, c1 in enumerate(clusters):
                    for c2 in clusters[i+1:]:
                        s1, s2 = merged[merged["cluster"] == c1], merged[merged["cluster"] == c2]
                        if len(s1) > 0 and len(s2) > 0:
                            t = logrank_test(s1["time"], s2["time"], s1["event"], s2["event"])
                            pairwise[f"{c1}_vs_{c2}"] = {"p": float(t.p_value), "stat": float(t.test_statistic)}
                km_results[name] = {
                    "multivariate_logrank_p": float(mtest.p_value),
                    "multivariate_logrank_stat": float(mtest.test_statistic),
                    "pairwise": pairwise,
                    "n_clusters": len(clusters),
                }
                print(f"  {name}: Multivariate Log-rank p={mtest.p_value:.4f}")
    else:
        print("[INFO] 未提供临床数据，跳过 KM 分析")

    # ============================================================
    # 9. 保存完整报告
    # ============================================================
    report = {
        "experiment": exp_name,
        "cluster_epoch": cluster_epoch,
        "simclr_epoch": simclr_ckpt.get("epoch", "?"),
        "n_patients": len(z_simclr),
        "n_clusters": n_clusters,
        "internal_metrics": internal_results,
        "collapse_analysis": collapse_results,
        "bootstrap_stability": stability_results,
        "kNN_consistency": knn_results,
        "survival_analysis": km_results,
        "cluster_distribution": {name: {int(u): int(c) for u, c in zip(*np.unique(labels, return_counts=True))}
                               for name, labels in all_labels.items()},
    }

    report_path = os.path.join(output_dir, "comparison_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] 完整报告已保存: {report_path}")

    # --- 打印汇总表 ---
    print("\n" + "=" * 60)
    print("===== 最终汇总 =====")
    print(f"{'方法':<16} {'Silhouette':>10} {'DBI':>8} {'CHI':>8} {'NMI':>8} {'ARI':>8} {'kNN':>8}")
    print("-" * 75)
    for name in all_labels:
        ir = internal_results.get(name, {})
        sr = stability_results.get(name, {})
        kr = knn_results.get(name, {})
        sil = ir.get("silhouette", None)
        dbi = ir.get("davies_bouldin", None)
        chi = ir.get("calinski_harabasz", None)
        nmi = sr.get("nmi_mean", None)
        ari = sr.get("ari_mean", None)
        knn = kr.get("kNN_consistency", None)
        def fmt(v): return f"{v:.4f}" if v is not None else "N/A"
        def fmt_chi(v): return f"{v:.1f}" if v is not None else "N/A"
        print(f"{name:<16} {fmt(sil):>10} {fmt(dbi):>8} {fmt_chi(chi):>8} {fmt(nmi):>8} {fmt(ari):>8} {fmt(knn):>8}")

    return report


# ============================================================
# 命令行入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="聚类方法对比（SimCLR 二阶段方案）")
    parser.add_argument("--simclr_checkpoint", type=str, required=True,
                        help="SimCLR 预训练编码器路径（必填）")
    parser.add_argument("--simclr_cluster_ckpt", type=str, required=True,
                        help="聚类头 checkpoint 路径（必填）")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="数据目录")
    parser.add_argument("--clinical", type=str, default=None,
                        help="临床信息CSV（含 patient_id, time, event 列）")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录")
    parser.add_argument("--K", type=int, default=None,
                        help="强制指定聚类数（默认从 Ours 自动推断）")
    parser.add_argument("--n_bootstrap", type=int, default=5,
                        help="Bootstrap 轮数")
    parser.add_argument("--k_knn", type=int, default=10,
                        help="kNN 一致性的 k 值")

    args = parser.parse_args()

    run_comparison(
        simclr_checkpoint=args.simclr_checkpoint,
        simclr_cluster_ckpt=args.simclr_cluster_ckpt,
        data_dir=args.data_dir,
        clinical_path=args.clinical,
        output_dir=args.output_dir,
        n_clusters=args.K,
        n_bootstrap=args.n_bootstrap,
        k_neighbors=args.k_knn,
    )