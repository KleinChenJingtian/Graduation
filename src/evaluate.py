"""
evaluate.py
聚类结果输出 & 生存分析评估

功能：
1. 加载训练好的模型，对所有患者输出硬分配亚型标签（Hard Label）
2. 基于亚型分组，计算 Kaplan-Meier 生存曲线
3. 对数秩检验（Log-rank test）评估亚型间生存差异显著性
4. 亚型间临床指标对比（年龄、肿瘤大小等）
"""

# ============================================================
# 运行方式（在项目根目录 D:\codeC\VsCodeP 下执行）
# ============================================================
#
# ---- 第一步：生成临床数据（只需执行一次，或临床数据更新后重新执行）----
#   python match_clinical.py
#   输出：matched_clinical.csv（完整临床数据，313 行 × 23 列）
#         clinical_eval.csv    （评估专用，313 行 × 7 列）
#
# ---- 第二步：设置环境变量 ----
#   $env:PYTHONPATH="."
#
# ---- 第三步：运行评估 ----
#   python -m Graduation.src.evaluate `
#       --data_dir Graduation/data `
#       --checkpoint Graduation/experiments/20260509_223937/checkpoint.pth `
#       --clinical Graduation/clinical_eval.csv
#
#   参数说明：
#     --data_dir     MRI 数据目录（含 313 个患者子目录）
#     --checkpoint   训练好的模型权重路径
#     --clinical     临床数据 CSV（由 match_clinical.py 生成）
#     --output_dir   可选，评估结果输出目录（默认自动推导）
#
# ---- 评估输出（保存在 evaluation_results/<实验名>/ 下）----
#   patient_cluster_labels.csv  每个患者的亚型标签
#   kaplan_meier.png            Kaplan-Meier 生存曲线
#   cox_univariate.csv          单变量 Cox 回归（cluster）
#   cox_multivariate.csv        多变量 Cox 回归（cluster + age + kps + cht）
#   clinical_comparison.json    亚型间临床指标对比（Kruskal-Wallis / 卡方检验）
#   evaluation_report.json      完整评估报告

import os
import json
import warnings
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from scipy.stats import mannwhitneyu, kruskal
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from Graduation.src.dataset import GBMDataset
from Graduation.src.model import DeepClusteringModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 1. 模型推理：获取所有患者的硬分配亚型
# ============================================================

def predict_clusters(model, loader, device):
    """
    遍历整个数据集，收集每个患者的：
    - 嵌入向量 z
    - 软分配 q
    - 硬分配 label（argmax）
    - clean_pi（无噪声的簇权重）

    返回:
        patient_ids: [N] 患者ID列表
        z_all:       [N, feature_dim]
        q_all:       [N, K]
        hard_labels: [N] int
        clean_pi:    [K] 簇权重（无噪声）
    """
    model.eval()
    z_list, q_list, pid_list = [], [], []
    last_clean_pi = None

    with torch.no_grad():
        for x, mask, modality_mask, pids in loader:
            x = x.to(device)
            modality_mask = modality_mask.to(device)
            z, q, clean_pi, *_ = model(x, modality_mask, epoch=999)

            z_list.append(z.cpu().numpy())
            q_list.append(q.cpu().numpy())
            pid_list.extend(pids)
            last_clean_pi = clean_pi.cpu()

    z_all = np.concatenate(z_list, axis=0)
    q_all = np.concatenate(q_list, axis=0)
    hard_labels = q_all.argmax(axis=1)

    return pid_list, z_all, q_all, hard_labels, last_clean_pi.numpy()


# ============================================================
# 2. 亚型分析 & 临床信息加载
# ============================================================

def load_clinical_data(clinical_path):
    """
    加载临床信息文件
    期望 CSV 格式，列：patient_id, time, event, age, kps, cht, rt
      time:  生存时间（月）
      event: 删失标记（1=死亡，0=删失/存活）
      age:   年龄（岁）
      kps:   KPS 分组（1=KPS≤70, 2=KPS≥80）
      cht:   化疗（1=是, 2=否）
      rt:    放疗（1=是, 2=否）
    由 match_clinical.py 从 Full_Clinical_Data.xlsx 自动生成。
    """
    if not os.path.exists(clinical_path):
        warnings.warn(f"临床数据文件不存在: {clinical_path}")
        return None

    df = pd.read_csv(clinical_path)
    df["patient_id"] = df["patient_id"].astype(str)
    return df


def compute_cluster_statistics(hard_labels, clinical_df, patient_ids):
    """
    按亚型分组统计临床指标：
    - 各亚型患者数、中位生存期、平均年龄、KPS分布、治疗比例
    """
    stats = {}
    for label in np.unique(hard_labels):
        idx = np.where(hard_labels == label)[0]
        pids = [patient_ids[i] for i in idx]

        if clinical_df is not None:
            sub_df = clinical_df[clinical_df["patient_id"].isin(pids)]
            stat_entry = {
                "n_patients": len(pids),
                "patient_ids": pids,
            }
            if "time" in sub_df.columns:
                stat_entry["median_survival_months"] = round(float(sub_df["time"].median()), 1) if sub_df["time"].notna().any() else None
            if "age" in sub_df.columns:
                stat_entry["mean_age"] = round(float(sub_df["age"].mean()), 1) if sub_df["age"].notna().any() else None
            if "kps" in sub_df.columns:
                kps_valid = sub_df["kps"].dropna()
                stat_entry["kps_high_pct"] = round(float((kps_valid == 2).mean()) * 100, 1) if len(kps_valid) > 0 else None
            if "cht" in sub_df.columns:
                cht_valid = sub_df["cht"].dropna()
                stat_entry["cht_yes_pct"] = round(float((cht_valid == 1).mean()) * 100, 1) if len(cht_valid) > 0 else None
            stats[int(label)] = stat_entry
        else:
            stats[int(label)] = {
                "n_patients": len(pids),
                "patient_ids": pids,
            }

    return stats


# ============================================================
# 3. Kaplan-Meier 生存分析
# ============================================================

def kaplan_meier_analysis(clinical_df, hard_labels, patient_ids, output_dir):
    """
    按亚型分组绘制 Kaplan-Meier 曲线
    进行 Log-rank 检验
    """
    if clinical_df is None or "time" not in clinical_df.columns:
        print("[INFO] 临床数据缺少 'time' 列，跳过生存分析")
        return {}

    # 构建分析用的 DataFrame
    label_df = pd.DataFrame({"patient_id": patient_ids, "cluster": hard_labels})
    merged = pd.merge(label_df, clinical_df, on="patient_id", how="inner")

    if len(merged) < 2:
        print("[WARNING] 匹配到临床数据的患者少于2人，跳过生存分析")
        return {}

    clusters = sorted(merged["cluster"].unique())
    if len(clusters) < 2:
        print("[INFO] 有效亚型少于2个，跳过组间比较")
        return {}

    # ========== Kaplan-Meier 曲线 ==========
    plt.figure(figsize=(10, 7))

    kmf_dict = {}
    for c in clusters:
        sub = merged[merged["cluster"] == c]
        kmf = KaplanMeierFitter()
        kmf.fit(sub["time"], sub["event"], label=f"Subtype {c} (n={len(sub)})")
        kmf.plot_survival_function(ax=plt.gca())
        kmf_dict[c] = kmf

    plt.title("Kaplan-Meier Survival Curves by Cluster")
    plt.xlabel("Time (months)")
    plt.ylabel("Survival Probability")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "kaplan_meier.png"), dpi=150)
    plt.close()
    print(f"[OK] Kaplan-Meier 曲线已保存: {output_dir}/kaplan_meier.png")

    # ========== Log-rank 检验 ==========
    results = {}
    if len(clusters) == 2:
        c1, c2 = clusters
        sub1 = merged[merged["cluster"] == c1]
        sub2 = merged[merged["cluster"] == c2]

        if len(sub1) > 0 and len(sub2) > 0:
            test = logrank_test(
                sub1["time"], sub2["time"],
                sub1["event"], sub2["event"]
            )
            results["logrank_statistic"] = float(test.test_statistic)
            results["logrank_p_value"] = float(test.p_value)
            print(f"[Log-rank] Subtype {c1} vs Subtype {c2}: p = {test.p_value:.4f}")
    else:
        # 多组：两两比较 + 整体
        from lifelines.statistics import multivariate_logrank_test
        mtest = multivariate_logrank_test(
            merged["time"], merged["cluster"], merged["event"]
        )
        results["multivariate_logrank_statistic"] = float(mtest.test_statistic)
        results["multivariate_logrank_p_value"] = float(mtest.p_value)
        print(f"[Log-rank Multivariate] p = {mtest.p_value:.4f}")

        # 两两比较
        pairwise = {}
        for i, c1 in enumerate(clusters):
            for c2 in clusters[i+1:]:
                sub1 = merged[merged["cluster"] == c1]
                sub2 = merged[merged["cluster"] == c2]
                if len(sub1) > 0 and len(sub2) > 0:
                    t = logrank_test(
                        sub1["time"], sub2["time"],
                        sub1["event"], sub2["event"]
                    )
                    pairwise[f"{c1}_vs_{c2}"] = {"statistic": float(t.test_statistic), "p": float(t.p_value)}
        results["pairwise_logrank"] = pairwise

    # ========== Cox 比例风险回归 (单变量 + 多变量) ==========
    cox_univariate = {}
    cox_multivariate = {}

    # --- 单变量 Cox：仅 cluster ---
    try:
        cox_df = merged[["time", "event", "cluster"]].copy()
        cox_df["cluster"] = cox_df["cluster"].astype(str)
        cph = CoxPHFitter()
        cph.fit(cox_df, duration_col="time", event_col="event")
        cox_univariate = {
            idx: {"coef": float(row["coef"]), "hr": float(row["exp(coef)"]), "p": float(row["p"])}
            for idx, row in cph.summary.iterrows()
        }
        cox_uni_df = pd.DataFrame({
            "coef": cph.summary["coef"].values,
            "exp(coef)": np.exp(cph.summary["coef"].values),
            "p": cph.summary["p"].values
        }, index=cph.summary.index)
        cox_uni_df.to_csv(os.path.join(output_dir, "cox_univariate.csv"))
        print(f"[OK] 单变量 Cox 回归 (cluster) 已保存")
    except Exception as e:
        print(f"[WARNING] 单变量 Cox 回归失败: {e}")

    # --- 多变量 Cox：cluster + age + kps + cht ---
    try:
        cov_cols = ["time", "event", "cluster", "age", "kps", "cht"]
        available = [c for c in cov_cols if c in merged.columns]
        cox_multi_df = merged[available].copy()
        cox_multi_df["cluster"] = cox_multi_df["cluster"].astype(str)
        # 删除含缺失值的行（kps/cht 可能有缺失）
        n_before = len(cox_multi_df)
        cox_multi_df = cox_multi_df.dropna()
        n_after = len(cox_multi_df)
        if n_after >= 20 and len(cox_multi_df["cluster"].unique()) >= 2:
            cph_multi = CoxPHFitter()
            cph_multi.fit(cox_multi_df, duration_col="time", event_col="event")
            cox_multivariate = {
                idx: {"coef": float(row["coef"]), "hr": float(row["exp(coef)"]), "p": float(row["p"])}
                for idx, row in cph_multi.summary.iterrows()
            }
            cox_multi_summary = pd.DataFrame({
                "coef": cph_multi.summary["coef"].values,
                "exp(coef)": np.exp(cph_multi.summary["coef"].values),
                "p": cph_multi.summary["p"].values
            }, index=cph_multi.summary.index)
            cox_multi_summary.to_csv(os.path.join(output_dir, "cox_multivariate.csv"))
            print(f"[OK] 多变量 Cox 回归 (cluster + age + kps + cht) 已保存 (n={n_after})")
        else:
            print(f"[INFO] 多变量 Cox 样本量不足 (n={n_after})，跳过")
    except Exception as e:
        print(f"[WARNING] 多变量 Cox 回归失败: {e}")

    results["cox_univariate"] = cox_univariate
    results["cox_multivariate"] = cox_multivariate

    return results


# ============================================================
# 4. 亚型间临床指标对比
# ============================================================

def compare_clinical_features(clinical_df, hard_labels, patient_ids, output_dir):
    """
    亚型间临床指标对比：
    - 连续型变量 (age, time)：Kruskal-Wallis + 两两 Mann-Whitney U
    - 分类型变量 (kps, cht, rt)：卡方检验
    """
    if clinical_df is None:
        return {}

    label_df = pd.DataFrame({"patient_id": patient_ids, "cluster": hard_labels})
    merged = pd.merge(label_df, clinical_df, on="patient_id", how="inner")

    numeric_cols = ["age", "time"]
    categorical_cols = ["kps", "cht", "rt"]
    results = {}

    # --- 连续型变量：Kruskal-Wallis + Mann-Whitney ---
    for col in numeric_cols:
        if col not in merged.columns:
            continue
        groups = [merged.loc[merged["cluster"] == c, col].dropna().values
                  for c in sorted(merged["cluster"].unique())]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2:
            continue

        try:
            stat, p_kw = kruskal(*groups)
            results[col] = {"kruskal_stat": float(stat), "kruskal_p": float(p_kw)}
            print(f"[Clinical] {col}: Kruskal-Wallis p = {p_kw:.4f}")

            from itertools import combinations
            pairwise = {}
            clusters = sorted(merged["cluster"].unique())
            for c1, c2 in combinations(clusters, 2):
                g1 = merged.loc[merged["cluster"] == c1, col].dropna().values
                g2 = merged.loc[merged["cluster"] == c2, col].dropna().values
                if len(g1) > 0 and len(g2) > 0:
                    _, p_mw = mannwhitneyu(g1, g2, alternative="two-sided")
                    pairwise[f"{c1}_vs_{c2}"] = float(p_mw)
                    print(f"  {c1} vs {c2}: p = {p_mw:.4f}")
            results[col]["pairwise_mannwhitney"] = pairwise
        except Exception as e:
            print(f"[WARNING] {col} 统计检验失败: {e}")

    # --- 分类型变量：卡方检验 ---
    from scipy.stats import chi2_contingency
    for col in categorical_cols:
        if col not in merged.columns:
            continue
        try:
            clusters = sorted(merged["cluster"].unique())
            # 构建列联表：行=亚型, 列=变量取值
            valid_data = merged[["cluster", col]].dropna()
            if len(valid_data) < 10:
                continue
            contingency = pd.crosstab(valid_data["cluster"], valid_data[col])
            if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
                chi2, p_chi, dof, _ = chi2_contingency(contingency)
                results[col] = {"chi2_stat": float(chi2), "chi2_p": float(p_chi), "dof": int(dof)}
                print(f"[Clinical] {col}: Chi-square p = {p_chi:.4f} (dof={dof})")
                # 各亚型分布
                for c in clusters:
                    sub = valid_data[valid_data["cluster"] == c][col]
                    if len(sub) > 0:
                        dist = sub.value_counts().to_dict()
                        print(f"  亚型 {c}: {dist}")
        except Exception as e:
            print(f"[WARNING] {col} 卡方检验失败: {e}")

    # 保存结果
    with open(os.path.join(output_dir, "clinical_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


# ============================================================
# 5. 主评估流程
# ============================================================

def evaluate(
    data_dir,
    checkpoint_path,
    clinical_path=None,
    output_dir=None,
    batch_size=4
):
    """
    完整评估流程

    参数:
        data_dir:         数据目录
        checkpoint_path:  模型权重路径
        clinical_path:    临床信息CSV路径（含 time, event 及可选的年龄、肿瘤体积等）
        output_dir:       评估结果输出目录
        batch_size:       batch大小
    """
    # --- 加载数据 ---
    dataset = GBMDataset(data_dir)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print(f"[INFO] 数据集: {len(dataset)} 患者")

    # --- 加载模型 ---
    model = DeepClusteringModel(
        num_modalities=4,
        feature_dim=256,
        max_K=10,
        beta=1.0,
        gamma=0.1
    ).to(DEVICE)

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"[INFO] 模型加载: epoch {checkpoint.get('epoch', '?')}")

    # --- 推理 ---
    patient_ids, z_all, q_all, hard_labels, pi = predict_clusters(model, loader, DEVICE)

    print(f"\n===== 聚类结果 =====")
    print(f"总患者数: {len(hard_labels)}")
    unique, counts = np.unique(hard_labels, return_counts=True)
    print(f"有效亚型数: {len(unique)}")
    for u, c in zip(unique, counts):
        print(f"  亚型 {u}: {c} 人  (π={pi[u]:.3f})")

    # --- 保存硬分配标签 ---
    labels_df = pd.DataFrame({
        "patient_id": patient_ids,
        "cluster_label": hard_labels,
        "cluster_prob": q_all.max(axis=1)
    })
    labels_csv_path = os.path.join(output_dir, "patient_cluster_labels.csv")
    labels_df.to_csv(labels_csv_path, index=False)
    print(f"[OK] 患者亚型标签已保存: {labels_csv_path}")

    # --- 加载临床数据 ---
    clinical_df = load_clinical_data(clinical_path) if clinical_path else None

    # --- 亚型统计 ---
    cluster_stats = compute_cluster_statistics(hard_labels, clinical_df, patient_ids)
    print(f"\n===== 亚型统计 =====")
    for k, v in cluster_stats.items():
        print(f"  亚型 {k}: {v}")

    # --- 内部评价指标（无标签聚类质量） ---
    print(f"\n===== 内部评价指标 =====")
    internal_metrics = {}
    try:
        silhouette = silhouette_score(z_all, hard_labels)
        dbi = davies_bouldin_score(z_all, hard_labels)
        chi = calinski_harabasz_score(z_all, hard_labels)
        internal_metrics = {
            "silhouette_score": float(silhouette),  # 越大越好（[-1, 1]）
            "davies_bouldin_index": float(dbi),     # 越小越好（[0, ∞)）
            "calinski_harabasz_index": float(chi)   # 越大越好（[0, ∞)）
        }
        print(f"  Silhouette Score: {silhouette:.4f} (越大越好)")
        print(f"  Davies-Bouldin Index: {dbi:.4f} (越小越好)")
        print(f"  Calinski-Harabasz Index: {chi:.4f} (越大越好)")
    except Exception as e:
        print(f"  [WARNING] 内部指标计算失败: {e}")
        internal_metrics = {"error": str(e)}

    # --- 生存分析 ---
    if clinical_df is not None:
        survival_results = kaplan_meier_analysis(clinical_df, hard_labels, patient_ids, output_dir)
        clinical_results = compare_clinical_features(clinical_df, hard_labels, patient_ids, output_dir)
    else:
        print("[INFO] 未提供临床数据，跳过生存分析和临床指标对比")
        survival_results = {}
        clinical_results = {}

    # --- 保存完整报告 ---
    report = {
        "n_patients": len(hard_labels),
        "n_clusters": int(len(unique)),
        "cluster_distribution": {int(u): int(c) for u, c in zip(unique, counts)},
        "pi_weights": {int(i): float(pi[i]) for i in range(len(pi))},
        "cluster_stats": cluster_stats,
        "survival_analysis": survival_results,
        "clinical_comparison": clinical_results,
        "internal_metrics": internal_metrics  # 【新增】无标签聚类质量评估
    }

    report_path = os.path.join(output_dir, "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[OK] 评估报告已保存: {report_path}")

    return report


# ============================================================
# 命令行入口
# ============================================================

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="聚类评估 & 生存分析")
    parser.add_argument("--data_dir", type=str, default="Graduation/data_test",
                        help="数据目录")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型权重路径")
    parser.add_argument("--clinical", type=str, default=None,
                        help="临床信息CSV（含 patient_id, time, event, age, tumor_volume 等列）")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="评估结果输出目录（默认根据checkpoint路径自动生成，如 evaluation_results/20260417_170547）")

    args = parser.parse_args()

    # 如果未指定 output_dir，从 checkpoint 路径自动推导
    if args.output_dir is None:
        # checkpoint 路径格式: experiments/20260417_170547/checkpoint.pth
        checkpoint_dir = os.path.dirname(args.checkpoint)  # experiments/20260417_170547
        exp_name = os.path.basename(checkpoint_dir)       # 20260417_170547
        # 直接用 evaluation_results（相对于当前工作目录）
        args.output_dir = os.path.join("Graduation", "evaluation_results", exp_name)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[INFO] 评估结果将保存至: {args.output_dir}")

    evaluate(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        clinical_path=args.clinical,
        output_dir=args.output_dir,
        batch_size=4
    )
