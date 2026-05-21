# compare/metrics.py
# 聚类方法对比： vs K-Means vs GMM
# 只需要已有的特征z_all和标签，即可对比


# 使用方法：

# 1. 先修改第129行的 checkpoint 路径为实际路径
# checkpoint_path = "experiments/你实际的文件夹/checkpoint.pth"

# 2. 运行：
# $env:PYTHONPATH="."
# python -m compare.metrics


import numpy as np
import json
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def compute_internal_metrics(z_all, labels):
    """
    计算三个内部评价指标
    """
    metrics = {}
    try:
        metrics["silhouette"] = float(silhouette_score(z_all, labels))
    except:
        metrics["silhouette"] = None

    try:
        metrics["davies_bouldin"] = float(davies_bouldin_score(z_all, labels))
    except:
        metrics["davies_bouldin"] = None

    try:
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(z_all, labels))
    except:
        metrics["calinski_harabasz"] = None

    return metrics


def kmeans_clustering(z_all, n_clusters):
    """
    K-Means聚类
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(z_all)
    return labels


def gmm_clustering(z_all, n_clusters):
    """
    GMM聚类（高斯混合模型）
    """
    gmm = GaussianMixture(n_components=n_clusters, random_state=42, n_init=10)
    labels = gmm.fit_predict(z_all)
    return labels


def compare_methods(z_all, your_labels, n_clusters):
    """
    对比多种聚类方法

    参数:
        z_all:       [N, feature_dim] 你的模型提取的特征
        your_labels: [N] 你的模型预测的硬标签
        n_clusters: 聚类数
    """
    results = {}

    # 1. DeepClustering
    results["YourMethod"] = {
        "n_clusters": len(np.unique(your_labels)),
        "distribution": {int(k): int(v) for k, v in zip(*np.unique(your_labels, return_counts=True))},
        "metrics": compute_internal_metrics(z_all, your_labels)
    }
    print(f"你的方法: {results['YourMethod']['n_clusters']}簇, Silhouette={results['YourMethod']['metrics']['silhouette']:.4f}")

    # 2. K-Means
    km_labels = kmeans_clustering(z_all, n_clusters)
    results["KMeans"] = {
        "n_clusters": len(np.unique(km_labels)),
        "distribution": {int(k): int(v) for k, v in zip(*np.unique(km_labels, return_counts=True))},
        "metrics": compute_internal_metrics(z_all, km_labels)
    }
    print(f"K-Means:   {results['KMeans']['n_clusters']}簇, Silhouette={results['KMeans']['metrics']['silhouette']:.4f}")

    # 3. GMM
    gmm_labels = gmm_clustering(z_all, n_clusters)
    results["GMM"] = {
        "n_clusters": len(np.unique(gmm_labels)),
        "distribution": {int(k): int(v) for k, v in zip(*np.unique(gmm_labels, return_counts=True))},
        "metrics": compute_internal_metrics(z_all, gmm_labels)
    }
    print(f"GMM:       {results['GMM']['n_clusters']}簇, Silhouette={results['GMM']['metrics']['silhouette']:.4f}")

    # 保存结果到 compare_results/时间戳/ 目录
    from datetime import datetime
    exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("compare_results", exp_name)
    os.makedirs(save_dir, exist_ok=True)

    report_path = os.path.join(save_dir, "comparison.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] 对比结果已保存: {report_path}")

    return results


if __name__ == "__main__":
    import os
    import torch
    from torch.utils.data import DataLoader
    from src.dataset import GBMDataset
    from src.model import DeepClusteringModel


    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型和数据
    print("加载模型和数据...")
    dataset = GBMDataset("data")
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    model = DeepClusteringModel(num_modalities=4, feature_dim=256, max_K=10).to(DEVICE)

    # 从最新checkpoint加载
    checkpoint_path = "experiments/20260504_092551/checkpoint.pth"  # 这里在使用前需要修改
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # 提取特征
    print("提取特征...")
    z_list, q_list, label_list = [], [], []

    with torch.no_grad():
        for x, mask, modality_mask, pids in loader:
            x = x.to(DEVICE)
            modality_mask = modality_mask.to(DEVICE)
            z, q, _, *_ = model(x, modality_mask, epoch=999)
            z_list.append(z.cpu().numpy())
            q_list.append(q.cpu().numpy())
            label_list.extend(pids)

    z_all = np.concatenate(z_list, axis=0)
    q_all = np.concatenate(q_list, axis=0)
    your_labels = q_all.argmax(axis=1)

    # 对比（自动确定最优簇数，这里用你的模型发现的簇数）
    n_clusters = len(np.unique(your_labels))
    print(f"\n使用簇数: {n_clusters}")

    results = compare_methods(z_all, your_labels, n_clusters)

    # 打印总结
    print("\n===== 总结 =====")
    print(f"{'Method':<15} {'Silhouette':>12} {'DBI':>12} {'CHI':>12}")
    print("-" * 55)
    for name, res in results.items():
        m = res["metrics"]
        sil = f"{m['silhouette']:.4f}" if m.get('silhouette') else "N/A"
        dbi = f"{m['davies_bouldin']:.4f}" if m.get('davies_bouldin') else "N/A"
        chi = f"{m['calinski_harabasz']:.1f}" if m.get('calinski_harabasz') else "N/A"
        print(f"{name:<15} {sil:>12} {dbi:>12} {chi:>12}")