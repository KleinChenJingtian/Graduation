"""
将临床Excel与MRI患者匹配，并统计各变量缺失情况。
用法：python Graduation/match_clinical.py
"""
import pandas as pd
import os
import numpy as np

# ============================================================
# 1. 配置
# ============================================================
EXCEL_PATH = r"D:\codeC\VsCodeP\Graduation\Full_Clinical_Data.xlsx"
MRI_DATA_DIR = r"D:\codeC\VsCodeP\Graduation\data"
ID_COLUMN = "ID"   # Excel中作为患者ID的列名

# ============================================================
# 2. 读取Excel和MRI患者列表
# ============================================================
df = pd.read_excel(EXCEL_PATH)
df[ID_COLUMN] = df[ID_COLUMN].astype(str).str.strip()

mri_patients = set()
for d in os.listdir(MRI_DATA_DIR):
    patient_dir = os.path.join(MRI_DATA_DIR, d)
    if os.path.isdir(patient_dir):
        mask_path = os.path.join(patient_dir, "tumor_to_MNI.nii.gz")
        if os.path.exists(mask_path):
            mri_patients.add(d)

print(f"Excel总行数: {len(df)}")
print(f"MRI有效患者数: {len(mri_patients)}")

# ============================================================
# 3. 匹配
# ============================================================
df_matched = df[df[ID_COLUMN].isin(mri_patients)].copy()
df_unmatched = df[~df[ID_COLUMN].isin(mri_patients)].copy()

print(f"\n匹配成功（有MRI+有临床数据）: {len(df_matched)} 例")
print(f"仅在Excel中（有临床数据但无MRI）: {len(df_unmatched)} 例")

mri_unmatched = mri_patients - set(df[ID_COLUMN].values)
print(f"仅在MRI中（有MRI但无临床数据）: {len(mri_unmatched)} 例")
if mri_unmatched:
    print(f"  （前10个: {list(mri_unmatched)[:10]}）")

# ============================================================
# 4. 定义关心的变量并重命名为论文中使用的名称
# ============================================================
# Full_Clinical_Data.xlsx 实际列名 → 标准化列名

# RT, CHT, WHO2021, Tumor_Location 的列名包含中文，直接取 df.columns[idx]
# 其余英文列名可直接匹配
_CN_COLUMNS = {
    17: "RT",            # RT放疗
    18: "CHT",           # CHT化疗
    20: "WHO2021",       # 2021 WHO分类
    22: "Tumor_Location", # 肿瘤位置
}

VAR_MAP = {
    "Age":                                          "Age",
    "Death          Yes=1; No=2":                   "Death",
    "OS(Months)":                                   "OS_Months",
    "Pre KPS         (KPS>=80)=2;(KPS<=70)=1":      "KPS_Group",
    "Recurrence  Yes=1;   No=2":                    "Recurrence",
    "Ki67":                                         "Ki67",
    # 模态可用性标志（1=有, 0=无/缺失）
    "FLAIR":                                        "Has_FLAIR",
    "FLAIR-2MM":                                    "Has_FLAIR_2MM",
    "3DT1-1MM":                                     "Has_3DT1_1MM",
    "3DT1C-1MM":                                    "Has_3DT1C_1MM",
    "T1C":                                          "Has_T1C",
    "T1WI":                                         "Has_T1WI",
    "T2WI":                                         "Has_T2WI",
    "DWI":                                          "Has_DWI",
    "ADC":                                          "Has_ADC",
    "PWI":                                          "Has_PWI",
    "DTI":                                          "Has_DTI",
    "REST":                                         "Has_REST",
}

# 为中文列名创建反向映射（从原始列名到简写名）
_CN_REVERSE_MAP = {}
for idx, name in _CN_COLUMNS.items():
    orig_col = df.columns[idx]
    _CN_REVERSE_MAP[orig_col] = name
    VAR_MAP[orig_col] = name

# 打印列名映射（防止乱码）
print("\n列名映射:")
for orig, new in VAR_MAP.items():
    print(f"  {repr(orig):<55s} -> {new}")

# 只保留需要的列
keep_cols = [ID_COLUMN] + list(VAR_MAP.values())
# rename 之前先选出需要的列
df_out = df_matched[[ID_COLUMN] + list(VAR_MAP.keys())].copy()
df_out.rename(columns=VAR_MAP, inplace=True)

# 数值化
numeric_cols = ["Age", "Death", "OS_Months", "KPS_Group", "RT", "CHT", "Recurrence", "Ki67",
                "Has_FLAIR", "Has_FLAIR_2MM", "Has_3DT1_1MM", "Has_3DT1C_1MM",
                "Has_T1C", "Has_T1WI", "Has_T2WI", "Has_DWI", "Has_ADC",
                "Has_PWI", "Has_DTI", "Has_REST"]
for col in numeric_cols:
    if col in df_out.columns:
        df_out[col] = pd.to_numeric(df_out[col], errors="coerce")

# ============================================================
# 5. 缺失值统计
# ============================================================
print("\n" + "=" * 70)
print("匹配后各变量缺失统计（用于论文 表3-2 或 第5章开头）：")
print("=" * 70)
print(f"{'变量':<20} {'有效例数':<10} {'缺失例数':<10} {'有效占比':<10}")
print("-" * 50)

stats_rows = []
for col in df_out.columns:
    if col == ID_COLUMN:
        continue
    valid = df_out[col].notna().sum()
    missing = df_out[col].isna().sum()
    pct = valid / len(df_out) * 100
    print(f"{col:<20} {valid:<10} {missing:<10} {pct:<10.1f}%")
    stats_rows.append({"变量": col, "有效例数": valid, "缺失例数": missing, "有效占比": f"{pct:.1f}%"})

# ============================================================
# 6. 模态可用性统计（用于论文 表3-2）
# ============================================================
print("\n" + "=" * 70)
print("模态可用性统计（用于论文 表3-2）：")
print("=" * 70)
modality_cols = ["Has_FLAIR", "Has_FLAIR_2MM", "Has_3DT1_1MM", "Has_3DT1C_1MM",
                 "Has_T1C", "Has_T1WI", "Has_T2WI",
                 "Has_DWI", "Has_ADC", "Has_PWI", "Has_DTI", "Has_REST"]
print(f"{'模态':<20s} {'有数据':<8s} {'缺失/无效':<10s} {'可用率':<8s}")
print("-" * 50)
for col in modality_cols:
    if col in df_out.columns:
        n_has = int(df_out[col].sum()) if df_out[col].notna().any() else 0
        n_miss = len(df_out) - (df_out[col].notna().sum())
        # 区分"标记为1"和"标记为0或缺失"
        has_1 = int((df_out[col] == 1).sum())
        has_0 = int((df_out[col] == 0).sum())
        has_nan = int(df_out[col].isna().sum())
        pct = has_1 / len(df_out) * 100
        print(f"{col:<20s} 1={has_1:<5d} 0={has_0:<3d} NaN={has_nan:<3d} {pct:.0f}%")

# ============================================================
# 7. WHO 2021 分类分布
# ============================================================
print("\n" + "=" * 70)
print("WHO 2021 分类分布：")
print("=" * 70)
if "WHO2021" in df_out.columns:
    who_counts = df_out["WHO2021"].value_counts().sort_index()
    for k, v in who_counts.items():
        print(f"  类别 {k}: {v} 例 ({v/len(df_out)*100:.1f}%)")
    print(f"  (注: 1=星形细胞瘤IDH突变; 2=少突胶质细胞瘤IDH突变/1p19q共缺失; 3=GBM IDH-野生型)")

# ============================================================
# 8. 确定各分析可用的样本量
# ============================================================
n_total = len(df_out)
n_survival = df_out[["OS_Months", "Death"]].dropna().shape[0]
cox_cols = ["OS_Months", "Death", "Age", "KPS_Group", "CHT"]
n_cox = df_out[cox_cols].dropna().shape[0]

print(f"\n" + "=" * 70)
print("各分析阶段可用样本量：")
print(f"  聚类分析（全部313例MRI患者）:      {len(mri_patients)}")
print(f"  生存KM曲线（有OS_Months+Death）:    {n_survival}")
print(f"  Cox多变量回归（有完整协变量）:      {n_cox}")
print(f"  其中 Death=1(死亡) / Death=2(存活):  "
      f"{int((df_out['Death']==1).sum())} / {int((df_out['Death']==2).sum())}")

# ============================================================
# 7. 保存匹配后的数据
# ============================================================
OUTPUT_PATH = r"D:\codeC\VsCodeP\Graduation\matched_clinical.csv"
df_out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
print(f"\n匹配后的临床数据已保存至: {OUTPUT_PATH}")

# 同时保存缺失统计表
stats_df = pd.DataFrame(stats_rows)
stats_path = r"D:\codeC\VsCodeP\Graduation\clinical_missing_stats.csv"
stats_df.to_csv(stats_path, index=False, encoding="utf-8-sig")
print(f"缺失统计表已保存至: {stats_path}")

# ============================================================
# 9. 生成 evaluate.py / comparison.py 可直接使用的临床CSV
# ============================================================
print("\n" + "=" * 70)
print("生成评估用临床数据 (clinical_eval.csv)：")

df_eval = df_out.copy()

# 9.1 列名映射：matched_clinical → evaluate.py 期望的列名
#     ID        → patient_id
#     OS_Months → time
#     Death     → event (需重新编码：1=死亡→1, 2=存活→0)
#     Age       → age
#     KPS_Group → kps
#     CHT       → cht
#     RT        → rt
df_eval = df_eval.rename(columns={
    "ID":        "patient_id",
    "OS_Months": "time",
    "Age":       "age",
    "KPS_Group": "kps",
    "CHT":       "cht",
    "RT":        "rt",
})

# 9.2 Death 重新编码：原始 1=死亡, 2=存活 → 目标 1=死亡(事件), 0=存活(删失)
df_eval["event"] = df_eval["Death"].map({1.0: 1, 2.0: 0})
# 处理可能的 NaN
df_eval["event"] = df_eval["event"].fillna(0).astype(int)

# 9.3 只保留 evaluate.py / comparison.py 需要的列
eval_cols = ["patient_id", "time", "event", "age", "kps", "cht", "rt"]
df_eval = df_eval[eval_cols]

# 9.4 统计
n_eval = len(df_eval)
n_event = int(df_eval["event"].sum())
n_censor = n_eval - n_event
print(f"  总患者数:            {n_eval}")
print(f"  事件 (死亡=1):       {n_event}")
print(f"  删失 (存活=0):       {n_censor}")
print(f"  time 中位生存期:     {df_eval['time'].median():.1f} 月")
print(f"  age 有效/缺失:       {df_eval['age'].notna().sum()} / {df_eval['age'].isna().sum()}")
print(f"  kps 有效/缺失:       {df_eval['kps'].notna().sum()} / {df_eval['kps'].isna().sum()}")
print(f"  cht 有效/缺失:       {df_eval['cht'].notna().sum()} / {df_eval['cht'].isna().sum()}")
print(f"  rt  有效/缺失:       {df_eval['rt'].notna().sum()} / {df_eval['rt'].isna().sum()}")

EVAL_PATH = r"D:\codeC\VsCodeP\Graduation\clinical_eval.csv"
df_eval.to_csv(EVAL_PATH, index=False, encoding="utf-8-sig")
print(f"\n评估用临床数据已保存至: {EVAL_PATH}")
print("使用方式:")
print("  python -m Graduation.src.evaluate --checkpoint <path> --clinical Graduation/clinical_eval.csv")
print("  python -m Graduation.compare.comparison --checkpoint <path> --clinical Graduation/clinical_eval.csv")
