# GBM临床背景与影像亚型文献阅读笔记

> 收录与胶质母细胞瘤MRI影像分型、预后分层、无监督聚类直接相关的文献

---

## 1. Guo et al. — Integrating imaging and genomic data for the discovery of distinct glioblastoma subtypes: a joint learning approach

- **年份**：2024
- **来源**：Scientific Reports (Nature)
- **作者**：Jun Guo, Anahita Fathi Kazeroni, et al.
- **任务类型**：无监督聚类 + 生存预后分型
- **数据集**：571例 IDH野生型 GBM（回顾性多中心），预处理后462例有完整影像组学，355例有基因组学，246例有两模态完整数据
- **核心方法**：Anchor-based Partial Multi-modal Clustering (APMC)，融合12个MRI影像组学特征 + 13个驱动基因，L21范数特征选择，谱聚类
- **是否聚类**：✅ 是（谱聚类，通过Gap Statistic确定K=3）
- **是否使用MRI**：✅ 多参数MRI (T1, T1-Gd, T2, T2-FLAIR, DSC, DTI)
- **是否涉及生存分析**：✅ KM曲线 + Log-rank检验 + Cox回归
- **核心发现**：发现3个亚型（高/中/低风险），HR=1.64 (95%CI 1.17-2.31)，亚型为独立预后因子；多变量Cox证实亚型独立于年龄、MGMT、切除范围
- **本文可引用核心观点**：多模态数据联合聚类能发现具有显著生存差异的GBM亚型；影像+基因组学互补；APMC可处理模态缺失
- **适合引用章节**：1.1.1（GBM分型现状）、1.2.4（研究现状总结）、5.5（生存分析对比）
- **与本文的关联点**：验证了"影像特征聚类可发现预后相关亚型"这一核心假设，但该方法依赖基因组学数据且需预设K，本文完全基于MRI且自动确定K
- **备注**：WebFetch已确认细节，包括作者列表、精确HR、Cox回归结果

---

## 2. MRI-based radiomic clustering identifies a glioblastoma subtype enriched for neural stemness and proliferative programs

- **年份**：2024
- **来源**：Neuro-Oncology (期刊待确认)
- **任务类型**：无监督影像组学聚类 + 多组学验证
- **数据集**：IDH野生型GBM患者
- **核心方法**：无监督影像组学聚类 → 4个亚型，整合bulk RNA-seq、scRNA-seq、空间转录组学验证
- **是否聚类**：✅ 是
- **是否使用MRI**：✅
- **是否涉及生存分析**：✅ 显著生存差异
- **核心发现**：Group 4为高风险亚型（周边强化），富集细胞周期/增殖标记物和神经干细胞特征，VAX2为候选驱动基因
- **本文可引用核心观点**：影像聚类亚型可以通过多组学数据获得生物学验证；"影像表型 → 分子特征"的完整证据链
- **适合引用章节**：6.2.4（生物学解释性不足——未来需多组学验证）、6.3方向2（MRI+Genomics）
- **与本文的关联点**：展示了"影像亚型 → 分子特征关联"的理想路径，本文目前仅到生存验证；为未来工作提供蓝图
- **⚠️ 降级引用**：不放核心，不作为关键对照；用于"已有工作尝试将影像亚型与转录组联系"的铺垫

---

## 3. Radiomic MRI signature reveals three distinct subtypes of glioblastoma with different clinical and molecular characteristics

- **年份**：2018
- **来源**：Scientific Reports (Nature)
- **任务类型**：无监督聚类 + 生存分析 + 分子特征关联
- **数据集**：208例发现队列 + 53例验证队列，de novo GBM
- **核心方法**：267个影像组学特征（TU/NC/ED区域），无监督聚类发现亚型，关联IDH1、MGMT、EGFRvIII、转录组亚型
- **是否聚类**：✅ 是（无监督高维聚类）
- **是否使用MRI**：✅ 多参数MRI (T1, T1CE, T2, T2-FLAIR, DSC, DTI)
- **是否涉及生存分析**：✅ 显著（p<0.001）
- **核心发现**：3个影像亚型具有不同生存、分子特征（IDH1、MGMT、EGFRvIII）；提供超越WHO分类的风险分层
- **本文可引用核心观点**：影像组学聚类可发现具有临床和分子异质性的GBM亚型；影像亚型可指导精准治疗
- **适合引用章节**：1.1.1（GBM分型现状）、5.5（与本文方法对比）
- **与本文的关联点**：早期重要工作，证明纯MRI聚类可发现预后相关亚型；本文在此基础上引入深度学习和自动K确定

---

## 4. Foltyn-Dumitru et al. — Shape matters: unsupervised exploration of IDH-wildtype glioma imaging survival predictors

- **年份**：2024
- **来源**：European Radiology
- **任务类型**：无监督聚类 + 生存预测
- **数据集**：436例 IDH野生型胶质瘤 + UCSF验证队列(n=397)
- **核心方法**：PAM聚类，基于肿瘤形状影像组学特征 + 体积
- **是否聚类**：✅ 是（PAM）
- **是否使用MRI**：✅ MRI形状特征
- **是否涉及生存分析**：✅ KM曲线 + Log-rank (p=0.002)
- **核心发现**：两类——Cluster 1（高球形度/伸长率，中位OS 23.8个月）vs Cluster 2（大直径/表面积/体积，中位OS 11.4个月），外部验证确认
- **本文可引用核心观点**：影像形状特征的无监督聚类具有跨中心泛化能力；外部验证是关键
- **适合引用章节**：6.2.1（数据集层面局限性——单中心）、6.3方向1（多中心验证）
- **与本文的关联点**：证明了外部验证的可行性与必要性，本文目前仅在单中心数据上验证

---

## 5. Cho et al. — Multi-habitat radiomics unravels distinct phenotypic subtypes of glioblastoma with clinical and genomic significance

- **年份**：2020
- **来源**：Cancers
- **任务类型**：无监督聚类（多栖息地影像组学）
- **数据集**：GBM患者（多中心回顾性）
- **核心方法**：共识聚类，基于肿瘤多栖息地（增强区/水肿区/坏死区）影像组学特征
- **是否聚类**：✅ 是（共识聚类）
- **是否使用MRI**：✅ 多参数MRI
- **是否涉及生存分析**：✅ KM曲线
- **核心发现**：3个亚型——"异质增强型"、"环状增强坏死型"、"囊性型"，各有独特转录组特征和预后差异
- **本文可引用核心观点**：不同肿瘤子区域的影像特征反映不同的生物学行为，支持多模态/多区域分析的必要性
- **适合引用章节**：1.1.2（多模态MRI价值）、2.2.2（多模态融合挑战）
- **与本文的关联点**：支持本文"多模态融合"的设计动机——不同模态突出不同病理特征区域

---

## 6. Wroblewski et al. — Radiomic Consensus Clustering in Glioblastoma and Association with Gene Expression Profiles

- **年份**：2024
- **来源**：Cancers
- **任务类型**：无监督聚类 + 基因组关联
- **数据集**：114例 GBM（TCGA/CPTAC）
- **核心方法**：ConsensusClusterPlus（k-means + 欧氏距离），基于MRI影像组学特征
- **是否聚类**：✅ 是（共识聚类）
- **是否使用MRI**：✅ MRI影像组学
- **是否涉及生存分析**：✅ KM分析（但亚型间生存差异不显著）
- **核心发现**：发现3个簇，分别与免疫通路、DNA代谢通路等基因表达谱关联；但生存差异不显著
- **本文可引用核心观点**：纯影像组学聚类即使生存差异不显著，仍可捕捉与分子通路相关的影像表型差异
- **适合引用章节**：6.2.4（生物学解释性不足——聚类结果不一定直接对应生存差异）
- **与本文的关联点**：说明"聚类结果与生存相关"并非理所当然，本文的方法因深度特征学习而获得更显著的生存分层

---

## 7. Foltyn-Dumitru et al. — Cluster-based prognostication in glioblastoma: Unveiling heterogeneity based on diffusion and perfusion similarities

- **年份**：2024
- **来源**：Neuro-Oncology
- **任务类型**：无监督聚类 + 生存分层
- **数据集**：289例 GBM
- **核心方法**：PAM (Partitioning Around Medoids) 聚类，基于ADC、rCBV、rCBF灌注/扩散参数
- **是否聚类**：✅ 是（PAM）
- **是否使用MRI**：✅ 多参数生理MRI（DWI/DSC）
- **是否涉及生存分析**：✅ KM曲线 + Log-rank (p=0.012)
- **核心发现**：发现2个稳定亚型——低风险（高扩散低灌注，中位OS 26.6个月）vs 高风险（相反模式，中位OS 10.2个月）
- **本文可引用核心观点**：生理MRI参数聚类对GBM预后分层具有临床价值
- **适合引用章节**：1.1.3（选题意义）、5.5（生存分析对比）、6.1.3（方法优势分析）
- **与本文的关联点**：证明无监督聚类在GBM预后分层中可行，但仅用生理参数（2-3个特征）+ 传统聚类方法，本文用深度特征 + 自动K

---

## 8. Bonada et al. — Deep Learning for MRI Segmentation and Molecular Subtyping in Glioblastoma: Critical Aspects from an Emerging Field

- **年份**：2024
- **来源**：Biomedicines
- **任务类型**：综述
- **数据集**：N/A（综述）
- **核心方法**：系统综述DL在GBM MRI分割和分子分型中的应用
- **是否聚类**：N/A
- **是否使用MRI**：N/A（综述MRI相关方法）
- **是否涉及生存分析**：N/A
- **核心发现**：关键局限性包括：MRI异质性、信息序列缺乏、术后分割不准确、伦理问题
- **本文可引用核心观点**：MRI异质性和缺乏信息序列是多模态分析的核心挑战
- **适合引用章节**：1.1.2（多模态MRI挑战）、2.2.2（融合挑战）、6.2.2（方法局限性）
- **与本文的关联点**：全面概述了领域现状和挑战，为本文方法论设计提供背景支撑
- **⚠️ 降级引用**：综述最适合放introduction/background凑引用数，不需详细展开

---

## 9. Radiomics-based survival risk stratification of glioblastoma is associated with different genome alteration

- **年份**：2023
- **来源**：Computers in Biology and Medicine
- **任务类型**：影像组学 + 生存风险分层 + 基因组关联
- **数据集**：180例GBM（训练119 + 验证37 + 验证24）
- **核心方法**：开发radscore预测OS，C-index: 0.70/0.66/0.74；层次聚类发现2个表型簇
- **是否聚类**：✅ 是（层次聚类）
- **是否使用MRI**：✅ 多参数MRI (T1, T1-Gd, T2, T2-FLAIR)
- **是否涉及生存分析**：✅ C-index + Brier scores + Cox回归
- **核心发现**：影像组学特征和radscore是独立预后因子；2个聚类与不同生物学通路相关（VEGFA-VEGFR2, JAK-STAT, MAPK）
- **本文可引用核心观点**：影像组学风险分层可关联到基因组改变；为影像亚型提供分子解释
- **适合引用章节**：1.1.1（影像组学预后价值）、6.2.4（未来生物学验证方向）
- **与本文的关联点**：展示了"影像风险评分 → 基因组关联"的分析框架
- **⚠️ 降级引用**：适合放introduction/motivation，证明"影像表型 → 预后"的大框架

---

## 10. 基于影像组学的胶质瘤分子分型研究进展（中文文献）

- **年份**：2022-2024（多篇CNKI文献）
- **来源**：中华放射学杂志 / 中国医学影像技术 等
- **任务类型**：影像组学 + 分子分型
- **核心发现**：影像组学可以无创预测IDH突变、MGMT甲基化状态等分子标记物
- **本文可引用核心观点**：MRI影像组学为无创分子分型提供了可能
- **适合引用章节**：1.1.1（传统分型的局限性——引出MRI无创替代方案）
- **与本文的关联点**：为"从MRI中发现亚型"提供临床可行性的背景支撑
- **⚠️ 降级引用**：中文文献用于凑引用数，中文硕士/博士论文常用；不超过总引用5%

---

## 关键文献缺口与说明

- **直接研究缺口**：未找到同时满足"深度聚类 + 自动K确定 + 多模态MRI + GBM预后分层"四个条件的已发表论文，这为本文的创新性提供了支撑
- **Guo et al. (2024)** 是最接近本文的研究——但需要基因组学辅助且预设K，本文完全基于MRI且自动确定K
- **注意**：以上文献中，部分细节目录（如具体样本量、p值、统计方法）来自检索结果摘要，建议在引用前查阅原文确认