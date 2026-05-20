# 生存分析与临床验证文献阅读笔记

> 收录Kaplan-Meier生存分析、Log-rank检验、Cox回归在GBM影像亚型验证中的应用文献

---

## 1. Guo et al. — Integrating imaging and genomic data for the discovery of distinct glioblastoma subtypes (生存分析部分)

- **年份**：2024
- **来源**：Scientific Reports
- **任务类型**：聚类亚型的生存验证
- **数据集**：571例 IDH-wt GBM
- **核心方法**：Kaplan-Meier曲线 + Log-rank检验 + 多变量Cox回归
- **是否涉及生存分析**：✅ 核心方法
- **关键统计量**：HR=1.64 (95%CI 1.17-2.31)，Log-rank p<0.05；Cox多变量证实亚型为独立预后因子（与年龄、MGMT、切除范围并列）
- **本文可引用核心观点**：KM+Log-rank+Cox三重验证是聚类亚型临床验证的标准范式
- **适合引用章节**：5.5（生存分析实验设计依据）、2.6.2（临床验证指标）
- **与本文的关联点**：本文采用相同的三重验证范式（KM+Log-rank+Cox），验证标准与领域一致

---

## 2. Chung et al. — Cluster Analysis of DSC MRI, Dynamic Contrast-Enhanced MRI, and DWI Parameters Associated with Prognosis in Patients with Glioblastoma

- **年份**：2022
- **来源**：American Journal of Neuroradiology (AJNR)
- **任务类型**：生理MRI参数聚类 + 生存验证
- **数据集**：142例 GBM（全切除术后）
- **核心方法**：K-means聚类(nCBV, Ktrans, ADC) → 6簇；KM曲线 + Log-rank (P=0.037)；多变量Cox (HR=3.04, P=0.048)；C-index=0.699
- **是否涉及生存分析**：✅ 核心
- **核心发现**：簇4（低nCBV、低Ktrans、最低ADC）是独立不良预后因子；簇体积>9.70%预测1年PFS
- **本文可引用核心观点**：聚类衍生的簇体积分数可以作为连续预后生物标志物
- **适合引用章节**：5.5.1（KM曲线结果解读）、5.5.3（Cox回归分析）
- **与本文的关联点**：展示了"聚类 → 定量预后指标"的转化路径，本文可借鉴此思路讨论簇的临床转化价值

---

## 3. Sacli-Bilmez et al. — Identifying overall survival in 98 glioblastomas using VASARI features at 3T

- **年份**：2023
- **来源**：Clinical Imaging
- **任务类型**：影像特征 + 生存分析
- **数据集**：98例 GBM
- **核心方法**：VASARI特征 + KM + Log-rank + 多变量Cox + 监督ML分类器
- **是否涉及生存分析**：✅ 核心
- **核心发现**：肿瘤位置(p<0.001)、非增强成分比例(p=0.048)、坏死比例(p=0.02)与OS显著相关；ML分类12月生存准确率96.4%
- **本文可引用核心观点**：影像定性/定量特征可以通过生存分析验证其预后价值
- **适合引用章节**：1.1.2（多模态MRI的预后信息）、5.5（生存分析背景）
- **与本文的关联点**：说明了"哪些MRI特征与预后相关"，为本文跨模态融合的必要性提供支撑

---

## 4. Rathore et al. — Imaging phenotypes predict overall survival in glioma more accurately than basic demographic and cell mutation profiles

- **年份**：2023
- **来源**：Computer Methods and Programs in Biomedicine
- **任务类型**：影像表型 + 生存预测
- **数据集**：胶质瘤患者
- **核心方法**：11个MRI + 11个病理影像特征，Log-rank p<0.01筛选；多组学模型(MRI+病理+临床+基因) C-index=0.87
- **是否涉及生存分析**：✅ 核心
- **核心发现**：影像表型的预后预测能力超过基本人口学和突变信息
- **本文可引用核心观点**：影像表型（而非基因组学）可能是最强的预后预测因子——支持"纯影像聚类"的临床价值
- **适合引用章节**：1.1.3（选题意义——为什么纯MRI聚类有价值）、6.4（结尾升华——影像独立预后价值）
- **与本文的关联点**：为本文"纯MRI聚类"的临床定位提供了关键支撑——影像本身即具有独立预后信息

---

## 5. Foltyn-Dumitru et al. — Cluster-based prognostication in glioblastoma (生存分析部分)

- **年份**：2024
- **来源**：Neuro-Oncology
- **任务类型**：聚类 + 生存验证
- **核心方法**：KM曲线（中位OS: 26.6 vs 10.2个月），Log-rank p=0.012
- **核心发现**：仅用3个生理参数聚类，即可发现显著生存差异——说明"聚类+生存"范式有效
- **本文可引用核心观点**：简单的无监督聚类方法即可从MRI参数中发现预后相关的亚型
- **适合引用章节**：5.5（生存分析——作为baseline对比参照）

---

## 6. Machine learning-based prognostic subgrouping of glioblastoma: A multicenter study

- **年份**：2024
- **来源**：Neuro-Oncology
- **作者**：Multicenter collaborative (22 institutions, 3 continents)
- **任务类型**：机器学习预后分层（使用影像和临床数据）
- **数据集**：2838例患者，人口学多样，多中心
- **核心方法**：使用常规临床数据、MRI和分子测量，分为3个预后亚组（I, II, III）
- **是否涉及生存分析**：✅ 强烈（Cox比例风险模型，HR计算）
- **核心发现**：
  - HRs: I-II = 1.62 (95% CI: 1.43-1.84, p < .001), I-III = 3.48 (95% CI: 2.94-4.11, p < .001)
  - 影像特征分析揭示了多个对预后有独特贡献的肿瘤属性
  - 使用常规影像数据（非复杂影像协议）即可实现
- **本文可引用核心观点**：常规MRI数据可用于构建可泛化的预后分类系统；**多中心大样本**验证的重要性
- **适合引用章节**：1.1.3（选题意义——大规模验证价值）、5.5（与本文小样本对比）
- **与本文的关联点**：展示了多中心大样本验证的价值；本文目前仅在单中心验证，未来可扩展到多中心
- **备注**：⚠️ 需要查阅原文确认是否为**无监督聚类**还是监督/半监督方法

---

## 7. Radiomics-based survival risk stratification of glioblastoma is associated with different genome alteration

- **年份**：2023
- **来源**：Computers in Biology and Medicine
- **任务类型**：影像组学 + 生存风险分层 + 基因组关联
- **数据集**：180例GBM（训练119 + 验证37 + 验证24）
- **核心方法**：开发radscore预测OS，C-index: 0.70/0.66/0.74；层次聚类发现2个表型簇；Cox回归验证独立性
- **是否聚类**：✅ 是（层次聚类）
- **是否使用MRI**：✅ 多参数MRI (T1, T1-Gd, T2, T2-FLAIR)
- **是否涉及生存分析**：✅ C-index + Brier scores + Cox回归
- **核心发现**：
  - MRI-based radscore与OS显著相关（训练集C-index: 0.70）
  - 多变量分析揭示radscore是独立预后因子
  - 2个聚类与不同生物学通路相关（VEGFA-VEGFR2, JAK-STAT, MAPK）
- **本文可引用核心观点**：影像组学风险分层可关联到基因组改变；为影像亚型提供分子解释
- **适合引用章节**：1.1.1（影像组学预后价值）、6.2.4（未来生物学验证方向）
- **与本文的关联点**：展示了"影像风险评分 → 基因组关联"的分析框架；本文可借鉴此思路进行生物学验证

---

## 8. Kaizenet 2023 — Prognostic Model for GBM MRI Based on Imbalanced Classes, Classification, and Kaplan-Meier Survival Curve Learning Algorithms (BIOM-66)

- **年份**：2023
- **来源**：Neuro-Oncology (会议摘要)
- **任务类型**：监督ML + KM生存预测
- **数据集**：GBM MRI
- **核心方法**：SVM/决策树/随机森林/MLP + 肿瘤/水肿体积特征；MLP达96.97%准确率
- **是否涉及生存分析**：✅
- **核心发现**：总水肿体积是GBM生存的最强预测因子
- **本文可引用核心观点**：水肿区域特征（FLAIR模态突出）对生存预测至关重要
- **适合引用章节**：1.1.2（FLAIR模态的临床意义——显示水肿）、2.2.2（融合挑战——不同模态关注不同病理区域）
- **与本文的关联点**：解释了为什么单独使用T1c不够——水肿信息在FLAIR中，需要多模态融合

---

## 9. 生存分析方法论标准总结

### KM生存曲线
- **目的**：可视化不同组别的生存概率随时间的变化
- **关键要素**：
  - 横轴：时间（月）
  - 纵轴：生存概率
  - 每组标注：n=患者数
  - 图例标注：Log-rank p值
  - 删失标记（+）：存活或失访患者
- **本文应用**：图5-9（本文方法KM曲线）、图5-10（基线方法KM曲线对比）

### Log-rank检验
- **目的**：统计检验不同组别生存曲线是否有显著差异
- **关键要素**：
  - 零假设：各组生存曲线无差异
  - p<0.05为显著
  - 适用于整条曲线的全局比较
- **本文应用**：表5-9（各方法Log-rank p值汇总对比）

### Cox比例风险回归
- **目的**：量化各因素对生存的独立影响（多变量调整）
- **关键要素**：
  - Hazard Ratio (HR)：>1 高风险，<1 低风险
  - 95% 置信区间
  - p值
  - 可校正混杂因素（年龄、切除范围等）
- **本文应用**：表5-10（Cox回归系数表）、图5-11（HR森林图）

---

## 10. 生存分析在无监督聚类验证中的特殊考量

| 考量 | 说明 | 本文应对 |
|------|------|---------|
| **循环论证风险** | 聚类未使用生存信息，因此KM显著性是"独立验证"而非"自我实现" | ✅ 聚类完全无监督，未触碰生存标签 |
| **多重比较问题** | 多组KM比较时需校正p值 | 建议在论文中注明检验方式 |
| **删失处理** | 生存数据必然有删失（存活/失访） | ✅ KM方法自动处理右删失 |
| **样本量要求** | 每组至少需一定事件数（死亡）才能有统计效力 | 本文313例应足够 |
| **临床意义 vs 统计显著性** | p<0.05不等于临床有意义，需报告效应量（HR/中位OS差异） | ✅ 报告HR和中位OS |

---

## 文献归纳小结

- **验证范式**：KM+Log-rank+Cox是领域内公认的聚类亚型临床验证标准流程
- **关键文献**：
  - Guo et al. (2024) - 三重验证范式参照
  - **Machine learning-based prognostic subgrouping (2024)** - 多中心大样本验证（2838例）
  - **Radiomics-based survival risk stratification (2023)** - 影像组学+基因组关联
- **本文优势**：聚类完全不使用生存信息，因此KM显著性是真正的外部验证（而非循环论证）
- **重要发现**：
  - Rathore et al. (2023) 的"C-index 0.87"说明纯影像特征可具有强预后能力
  - **多中心验证的价值**——未来本文可扩展到多中心验证
  - **影像组学+基因组关联**——未来本文可进行生物学验证