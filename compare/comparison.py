# compare/comparison.py
# 全方位聚类方法对比：YourMethod / DEC / IDEC-Var / K-Means / GMM
# 输出：内部指标 + Bootstrap稳定性 + kNN一致性 + π分布 + UMAP可视化 + KM曲线
#
# ============================================================
# 运行方式（在项目根目录 D:\codeC\VsCodeP 下执行）
# ============================================================
#
# ---- 第一步：生成临床数据（只需执行一次）----
#   python match_clinical.py
#   输出：clinical_eval.csv（评估专用，313 行 × 7 列）
#
# ---- 第二步：设置环境变量 ----
#   $env:PYTHONPATH="."
#
# ---- 第三步：运行对比实验 ----
# （1）仅聚类对比（无临床数据）：
#   python -m Graduation.compare.comparison `
#       --checkpoint Graduation/experiments/20260509_223937/checkpoint.pth
#
# （2）聚类对比 + 生存分析：
#   python -m Graduation.compare.comparison `
#       --checkpoint Graduation/experiments/20260509_223937/checkpoint.pth `
#       --clinical Graduation/clinical_eval.csv
#
# （3）含 SimCLR baseline（需先运行 simclr_pretrain.py）：
#   python -m Graduation.compare.comparison `
#       --checkpoint Graduation/experiments/20260509_223937/checkpoint.pth `
#       --simclr_checkpoint Graduation/experiments/simclr_xxx/checkpoint.pth `
#       --clinical Graduation/clinical_eval.csv
#
#   参数说明：
#     --checkpoint         YourMethod 模型权重路径（必填）
#     --clinical           临床数据 CSV（可选，提供则做 KM 生存分析）
#     --simclr_checkpoint  SimCLR 预训练编码器路径（可选）
#     --data_dir           MRI 数据目录（默认 Graduation/data）
#     --K                  强制指定聚类数（默认从模型自动推断）
#     --n_bootstrap        Bootstrap 轮数（默认 5）
#
# ---- 评估输出（保存在 compare_results/<实验名>/full_comparison/ 下）----
#   comparison_report.json        完整对比报告
#   umap_comparison.png           UMAP 可视化（所有方法并列）
#   cluster_size_distribution.png 簇分布对比柱状图
#   feature_variance_boxplot.png  特征空间方差箱线图
#   pi_vs_q_bar.png              π vs q̄ 分布对比
#   km_<Method>.png              各方法的 KM 生存曲线

import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
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

from Graduation.src.dataset import GBMDataset
from Graduation.src.model import DeepClusteringModel, MultiViewEncoder


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 辅助函数
# ============================================================

def align_labels(labels_ref, labels_boot):
    """
    将 labels_boot 对齐到 labels_ref（标签可能相差一个 permutation）
    使用 Hungarian algorithm 最优匹配。
    当 bootstrap 簇数少于 ref 时，未匹配的簇保留原标签。
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
    - 论文中称为 "IDEC (Variance-Preserved Variant)"
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
        # 方差保留项：特征在每个维度上的方差不能过小（避免坍缩）
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


def run_yourmethod_on_subset(dataset, idx, checkpoint_state, n_clusters, device):
    """
    YourMethod 在数据子集上重新推理：
    加载模型 → 只在指定样本上 forward → 返回硬标签
    """
    boot_dataset = Subset(dataset, idx)
    boot_loader = DataLoader(boot_dataset, batch_size=4, shuffle=False)

    model_boot = DeepClusteringModel(num_modalities=4, feature_dim=256, max_K=10).to(device)
    model_boot.load_state_dict(checkpoint_state, strict=False)
    model_boot.eval()

    z_list = []
    with torch.no_grad():
        for x, mask, modality_mask, pids in boot_loader:
            x = x.to(device)
            modality_mask = modality_mask.to(device)
            z, q, _, *_ = model_boot(x, modality_mask, epoch=999)
            z_list.append(z.cpu().numpy())

    z_boot_full = np.concatenate(z_list, axis=0)
    # 精化：用 DEC 风格在 bootstrap 特征上做最终聚类
    km_boot = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km_boot.fit(z_boot_full)
    centers_boot = torch.tensor(km_boot.cluster_centers_, dtype=torch.float32, device=device, requires_grad=True)
    z_t = torch.tensor(z_boot_full, dtype=torch.float32, device=device)
    opt = torch.optim.Adam([centers_boot], lr=0.01)
    for _ in range(30):
        opt.zero_grad()
        d2 = torch.sum((z_t.unsqueeze(1) - centers_boot.unsqueeze(0)) ** 2, dim=2)
        qb = (1.0 + d2) ** (-1.0)
        qb = qb / torch.sum(qb, dim=1, keepdim=True)
        with torch.no_grad():
            pb = qb ** 2 / torch.sum(qb, dim=0, keepdim=True)
            pb = pb / torch.sum(pb, dim=1, keepdim=True)
        lo = torch.mean(torch.sum(pb * torch.log(pb / (qb + 1e-10)), dim=1))
        lo.backward()
        opt.step()
    with torch.no_grad():
        d2 = torch.sum((z_t.unsqueeze(1) - centers_boot.unsqueeze(0)) ** 2, dim=2)
        qb = (1.0 + d2) ** (-1.0)
        qb = qb / torch.sum(qb, dim=1, keepdim=True)
        labels_boot = qb.argmax(dim=1).cpu().numpy()
    return labels_boot


# ============================================================
# 主函数
# ============================================================

def run_comparison(
    checkpoint_path,
    data_dir="Graduation/data",
    clinical_path=None,
    simclr_checkpoint=None,
    output_dir=None,
    n_clusters=None,
    n_bootstrap=5,
    bootstrap_ratio=0.8,
    k_neighbors=10,
    random_seed=42
):
    np.random.seed(random_seed)

    # --- 输出目录 ---
    exp_name = os.path.basename(os.path.dirname(checkpoint_path))
    if output_dir is None:
        output_dir = os.path.join("Graduation", "compare_results", exp_name, "full_comparison")
    os.makedirs(output_dir, exist_ok=True)

    # --- 加载数据 & 模型 ---
    print("=" * 60)
    print("加载数据 & 模型...")
    dataset = GBMDataset(data_dir)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    model = DeepClusteringModel(num_modalities=4, feature_dim=256, max_K=10).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    model_epoch = checkpoint.get("epoch", "?")
    checkpoint_state = checkpoint["model_state"]
    print(f"[OK] 模型加载完成: epoch {model_epoch}")

    # --- 提取特征 ---
    print("提取特征...")
    z_list, q_list, pid_list = [], [], []

    with torch.no_grad():
        for x, mask, modality_mask, pids in loader:
            x = x.to(DEVICE)
            modality_mask = modality_mask.to(DEVICE)
            z, q, clean_pi, *_ = model(x, modality_mask, epoch=999)
            z_list.append(z.cpu().numpy())
            q_list.append(q.cpu().numpy())
            pid_list.extend(pids)

    z_all = np.concatenate(z_list, axis=0)
    q_all = np.concatenate(q_list, axis=0)
    your_labels = q_all.argmax(axis=1)
    clean_pi = clean_pi.cpu().numpy()
    print(f"[OK] 提取了 {len(z_all)} 个患者的特征")

    # --- SimCLR 特征提取（可选） ---
    z_simclr = None
    if simclr_checkpoint and os.path.exists(simclr_checkpoint):
        print("\n加载 SimCLR 预训练编码器...")
        simclr_encoder = MultiViewEncoder(num_modalities=4, feature_dim=256).to(DEVICE)
        simclr_ckpt = torch.load(simclr_checkpoint, map_location=DEVICE, weights_only=False)
        simclr_encoder.load_state_dict(simclr_ckpt["encoder_state"])
        simclr_encoder.eval()
        print(f"[OK] SimCLR 编码器加载完成: epoch {simclr_ckpt.get('epoch', '?')}")

        z_simclr_list = []
        with torch.no_grad():
            for x, mask, modality_mask, pids in loader:
                x = x.to(DEVICE)
                modality_mask = modality_mask.to(DEVICE)
                z_s = simclr_encoder(x, modality_mask)
                z_simclr_list.append(z_s.cpu().numpy())
        z_simclr = np.concatenate(z_simclr_list, axis=0)
        print(f"[OK] SimCLR 特征提取完成: {z_simclr.shape}")
    elif simclr_checkpoint:
        print(f"[WARNING] SimCLR checkpoint 不存在: {simclr_checkpoint}，跳过")

    # --- 确定聚类数 ---
    if n_clusters is None:
        n_clusters = len(np.unique(your_labels))
    print(f"使用簇数 K={n_clusters}")

    # ============================================================
    # 1. 各方法硬标签
    # ============================================================
    print("\n" + "=" * 60)
    print("各方法聚类...")

    labels_your = your_labels.copy()
    labels_dec = run_dec(z_all, n_clusters, epochs=30)
    labels_idec = run_idec_var(z_all, n_clusters, epochs=30, var_weight=0.1)
    labels_kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(z_all)
    labels_gmm = GaussianMixture(n_components=n_clusters, random_state=42, covariance_type="full", n_init=5).fit_predict(z_all)

    all_labels = {
        "YourMethod": labels_your,
        "DEC": labels_dec,
        "IDEC-Var": labels_idec,
        "KMeans": labels_kmeans,
        "GMM": labels_gmm,
    }

    # SimCLR 特征上的 K-Means
    simclr_added = False
    if z_simclr is not None:
        labels_simclr = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(z_simclr)
        all_labels["SimCLR+KMeans"] = labels_simclr
        simclr_added = True
        print("  SimCLR+KMeans: 已添加")

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
        z_ref = z_simclr if (name == "SimCLR+KMeans" and z_simclr is not None) else z_all
        try:
            sil = silhouette_score(z_ref, labels)
            dbi = davies_bouldin_score(z_ref, labels)
            chi = calinski_harabasz_score(z_ref, labels)
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
        """计算簇大小分布的基尼系数，0=完全均匀，1=极度不均"""
        sorted_counts = np.sort(counts)
        n = len(sorted_counts)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * sorted_counts)) / (n * np.sum(sorted_counts)) - (n + 1) / n

    def cluster_entropy(counts):
        """计算簇分布的归一化熵，1=完全均匀，0=单簇"""
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        return entropy / np.log(len(counts))  # 归一化

    def feature_variance_per_dim(z):
        """特征空间每个维度的方差"""
        return np.var(z, axis=0)  # [feature_dim]

    overall_feature_var = np.mean(feature_variance_per_dim(z_all))
    print(f"  整体特征空间方差: {overall_feature_var:.4f}")

    # 收集所有方法的簇分布数据用于对比柱状图
    cluster_labels_for_plot = {}
    cluster_sizes_for_plot = {}

    for name, labels in all_labels.items():
        unique, counts = np.unique(labels, return_counts=True)
        n_clusters = len(unique)
        gini = gini_coefficient(counts)
        entropy_norm = cluster_entropy(counts)

        # 特征空间方差：使用该方法聚类所在的特征空间
        z_collapse = z_simclr if (name == "SimCLR+KMeans" and z_simclr is not None) else z_all

        within_cluster_vars = []
        for c in unique:
            mask = labels == c
            if mask.sum() > 1:
                z_c = z_collapse[mask]
                var_c = np.mean(feature_variance_per_dim(z_c))
                within_cluster_vars.append(var_c)
        mean_within_var = np.mean(within_cluster_vars) if within_cluster_vars else 0.0

        # 最大簇占比
        max_cluster_ratio = counts.max() / counts.sum()

        collapse_results[name] = {
            "n_clusters": int(n_clusters),
            "cluster_sizes": {int(k): int(v) for k, v in zip(unique, counts)},
            "gini_coefficient": float(gini),
            "normalized_entropy": float(entropy_norm),
            "max_cluster_ratio": float(max_cluster_ratio),
            "mean_within_cluster_variance": float(mean_within_var),
        }

        print(f"  {name}: K={n_clusters}, 最大簇占比={max_cluster_ratio:.3f}, "
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

    # ---- 特征空间方差箱线图 (所有方法共享同一个 z_all，所以只画一个) ----
    fig, ax = plt.subplots(figsize=(10, 5))
    per_dim_var = feature_variance_per_dim(z_all)
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
    n_samples = len(z_all)
    boot_indices = [np.random.choice(n_samples, int(n_samples * bootstrap_ratio), replace=False)
                    for _ in range(n_bootstrap)]

    # --- YourMethod：重新加载模型在 bootstrap 样本上 forward ---
    print("  YourMethod: 重新加载模型推理...")
    your_nmi_list, your_ari_list = [], []
    for boot_i, idx in enumerate(boot_indices):
        labels_boot_your = run_yourmethod_on_subset(dataset, idx, checkpoint_state, n_clusters, DEVICE)
        labels_ref_your = your_labels[idx]
        labels_boot_aligned = align_labels(labels_ref_your, labels_boot_your)
        your_nmi_list.append(normalized_mutual_info_score(labels_ref_your, labels_boot_aligned))
        your_ari_list.append(adjusted_rand_score(labels_ref_your, labels_boot_aligned))
        print(f"    Bootstrap {boot_i+1}/{n_bootstrap} 完成")
    stability_results["YourMethod"] = {
        "nmi_mean": float(np.mean(your_nmi_list)), "nmi_std": float(np.std(your_nmi_list)),
        "ari_mean": float(np.mean(your_ari_list)), "ari_std": float(np.std(your_ari_list)),
    }
    print(f"  YourMethod: NMI={np.mean(your_nmi_list):.4f}±{np.std(your_nmi_list):.4f}, "
          f"ARI={np.mean(your_ari_list):.4f}±{np.std(your_ari_list):.4f}")

    # --- DEC / IDEC-Var / KMeans / GMM / SimCLR+KMeans：纯特征空间重聚 ---
    other_methods = [name for name in all_labels.keys() if name != "YourMethod"]
    for name in other_methods:
        # SimCLR+KMeans 使用 SimCLR 特征空间
        z_ref = z_simclr if (name == "SimCLR+KMeans" and z_simclr is not None) else z_all

        nmi_list, ari_list = [], []
        for idx in boot_indices:
            z_boot = z_ref[idx]
            if name == "DEC":
                labels_boot = run_dec(z_boot, n_clusters, epochs=30)
            elif name == "IDEC-Var":
                labels_boot = run_idec_var(z_boot, n_clusters, epochs=30, var_weight=0.1)
            elif name == "KMeans" or name == "SimCLR+KMeans":
                labels_boot = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(z_boot)
            else:  # GMM
                labels_boot = GaussianMixture(n_components=n_clusters, random_state=42, covariance_type="full", n_init=5).fit_predict(z_boot)
            labels_ref = all_labels[name][idx]
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
    # 4. kNN 一致性
    # ============================================================
    print("\n" + "=" * 60)
    print(f"kNN 一致性 (k={k_neighbors})...")

    knn_results = {}
    for name, labels in all_labels.items():
        # SimCLR+KMeans 的聚类在 SimCLR 特征空间
        z_knn = z_simclr if (name == "SimCLR+KMeans" and z_simclr is not None) else z_all
        try:
            consistency = compute_knn_consistency(z_knn, labels, k=k_neighbors)
            knn_results[name] = {"kNN_consistency": float(consistency)}
            print(f"  {name}: kNN一致率={consistency:.4f}")
        except Exception as e:
            print(f"  {name}: kNN计算失败 ({e})")
            knn_results[name] = {"error": str(e)}

    # ============================================================
    # 5. π vs q̄ 分布图 (仅 YourMethod)
    # ============================================================
    print("\n" + "=" * 60)
    print("π vs q̄ 分布对比...")

    q_mean = q_all.mean(axis=0)
    pi_aligned, q_aligned = align_pi_q(clean_pi, q_mean)
    K_eff = np.sum(pi_aligned > 0.01)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(pi_aligned))
    width = 0.35
    ax.bar(x - width/2, pi_aligned, width, label="π (Learned)", color="steelblue", alpha=0.8)
    ax.bar(x + width/2, q_aligned, width, label="q̄ (Average soft assignment)", color="coral", alpha=0.8)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Probability")
    ax.set_title(f"Your Method: π vs q̄ (K_eff={K_eff})")
    ax.legend()
    ax.set_xticks(x)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pi_vs_q_bar.png"), dpi=150)
    plt.close()
    print(f"[OK] π vs q̄ 图已保存: {output_dir}/pi_vs_q_bar.png")

    # ============================================================
    # 6. UMAP 可视化 (所有方法对比)
    # ============================================================
    print("\n" + "=" * 60)
    print("UMAP 可视化...")

    # UMAP: YourMethod特征空间
    try:
        from umap import UMAP
        reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        z_2d = reducer.fit_transform(z_all)
        has_umap = True
    except Exception as e:
        print(f"[WARNING] UMAP 失败，降级到 PCA: {e}")
        z_2d = PCA(n_components=2).fit_transform(z_all)
        has_umap = False

    # UMAP: SimCLR特征空间（如有）
    z_simclr_2d = None
    if z_simclr is not None:
        try:
            reducer2 = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
            z_simclr_2d = reducer2.fit_transform(z_simclr)
        except Exception as e:
            print(f"[WARNING] SimCLR UMAP 失败，降级到 PCA: {e}")
            z_simclr_2d = PCA(n_components=2).fit_transform(z_simclr)

    n_methods = len(all_labels)
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))
    if n_methods == 1:
        axes = [axes]

    for ax, (name, labels) in zip(axes, all_labels.items()):
        # SimCLR+KMeans 使用自己的特征空间
        if name == "SimCLR+KMeans" and z_simclr_2d is not None:
            coords = z_simclr_2d
        else:
            coords = z_2d
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=labels, s=5, cmap="tab10")
        ax.set_title(name)
        ax.set_xlabel("UMAP1" if has_umap else "PC1")
        ax.set_ylabel("UMAP2" if has_umap else "PC2")
        plt.colorbar(sc, ax=ax, label="Cluster")
    plt.suptitle(f"Clustering Comparison (K={n_clusters})" + (f" @ epoch {model_epoch}" if has_umap else ""), fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "umap_comparison.png"), dpi=150)
    plt.close()
    print(f"[OK] UMAP对比图已保存: {output_dir}/umap_comparison.png")

    # ============================================================
    # 7. KM 生存分析 (所有方法)
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
            # 剔除 time 或 event 缺失的患者（74例无 OS_Months）
            merged = merged.dropna(subset=["time", "event"])

            if len(merged) < 2:
                print(f"  {name}: 临床数据匹配少于2人，跳过")
                continue
            clusters = sorted(merged["cluster"].unique())
            if len(clusters) < 2:
                print(f"  {name}: 有效亚型少于2个，跳过")
                continue

            # --- Kaplan-Meier 曲线 ---
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

            # --- Log-rank 检验 ---
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
    # 8. 保存完整报告
    # ============================================================
    report = {
        "experiment": exp_name,
        "model_epoch": model_epoch,
        "n_patients": len(z_all),
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
    print(f"{'方法':<12} {'Silhouette':>10} {'DBI':>8} {'CHI':>8} {'NMI':>8} {'ARI':>8} {'kNN':>8}")
    print("-" * 60)
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
        print(f"{name:<12} {fmt(sil):>10} {fmt(dbi):>8} {fmt_chi(chi):>8} {fmt(nmi):>8} {fmt(ari):>8} {fmt(knn):>8}")

    return report


# ============================================================
# 命令行入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="聚类方法全方位对比")
    parser.add_argument("--checkpoint", type=str,
                        default="Graduation/experiments/20260509_223937/checkpoint.pth",
                        help="YourMethod 模型权重路径")
    parser.add_argument("--simclr_checkpoint", type=str, default=None,
                        help="SimCLR 预训练编码器路径（可选，添加 SimCLR+KMeans baseline）")
    parser.add_argument("--data_dir", type=str, default="Graduation/data",
                        help="数据目录")
    parser.add_argument("--clinical", type=str, default=None,
                        help="临床信息CSV（含 patient_id, time, event 列）")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录")
    parser.add_argument("--K", type=int, default=None,
                        help="强制指定聚类数（默认从模型自动推断）")
    parser.add_argument("--n_bootstrap", type=int, default=5,
                        help="Bootstrap 轮数")
    parser.add_argument("--k_knn", type=int, default=10,
                        help="kNN 一致性的 k 值")

    args = parser.parse_args()

    run_comparison(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        clinical_path=args.clinical,
        simclr_checkpoint=args.simclr_checkpoint,
        output_dir=args.output_dir,
        n_clusters=args.K,
        n_bootstrap=args.n_bootstrap,
        k_neighbors=args.k_knn,
    )