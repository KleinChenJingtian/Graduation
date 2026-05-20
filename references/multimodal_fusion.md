# 多模态医学图像融合文献阅读笔记

> 收录多模态MRI融合、Cross-Attention Transformer、医学图像多视图学习相关文献

---

## 1. Lin et al. — CKD-TransBTS: Clinical Knowledge-Driven Hybrid Transformer With Modality-Correlated Cross-Attention for Brain Tumor Segmentation

- **年份**：2023
- **来源**：IEEE Transactions on Medical Imaging (IEEE TMI)
- **任务类型**：多模态MRI脑肿瘤分割
- **数据集**：BraTS 2021
- **核心方法**：双分支架构——CNN分支（局部特征）+ Swin Transformer分支（全局上下文）；Modality-Correlated Cross-Attention (MCCA)——按MRI物理原理分组（T1/T1ce vs T2/FLAIR），组间交叉注意力；Trans&CNN Feature Calibration (TCFC)
- **是否聚类**：❌ 否（分割任务）
- **是否使用MRI**：✅ 4模态MRI
- **是否涉及生存分析**：❌ 否
- **核心发现**：按MRI物理学原理组织模态进行交叉注意力是有效的融合策略；跨模态注意力优于简单拼接
- **本文可引用核心观点**：多模态MRI融合需要超越简单拼接/相加，Cross-Attention可以实现模态间的语义级交互
- **适合引用章节**：2.2.3（Cross-Attention融合策略）、4.2（多模态编码器设计）
- **与本文的关联点**：本文的Transformer Self-Attention实现了类似MCCA的模态间信息交互，但目标不同（聚类 vs 分割）

---

## 2. Shi et al. — M²FTrans: Modality-Masked Fusion Transformer for Incomplete Multi-Modality Brain Tumor Segmentation

- **年份**：2024（在线发表2023年10月）
- **来源**：IEEE Journal of Biomedical and Health Informatics (JBHI)
- **任务类型**：多模态脑肿瘤分割（处理模态缺失）
- **数据集**：BraTS 2018/2020/2021
- **核心方法**：可学习融合Token + 掩码自注意力（masked self-attention）实现模态缺失下的鲁棒融合；空间权重注意力 + 通道融合Transformer
- **是否聚类**：❌ 否（分割任务）
- **是否使用MRI**：✅ 4模态MRI
- **是否涉及生存分析**：❌ 否
- **核心发现**：通过masked fusion tokens可以处理任意模态缺失，在临床不完整数据场景下具有实用价值
- **本文可引用核心观点**：临床实际中模态缺失是常见问题，融合方法需具备鲁棒性
- **适合引用章节**：3.2（数据预处理——模态缺失零填充处理）、6.2.1（数据集局限性——模态缺失）
- **与本文的关联点**：本文数据集同样存在模态缺失问题（patient2394 FLAIR缺失），采用零填充策略处理

---

## 3. CvTFuse: An unsupervised medical image fusion method of gliomas T1-DWI mode

- **年份**：2026
- **来源**：Magnetic Resonance Imaging (Elsevier)
- **作者**：Q Huang, W Chen, J Zeng, et al.
- **任务类型**：**无监督**医学图像融合（T1-DWI）
- **数据集**：胶质瘤MRI（T1和DWI模态）
- **核心方法**：双分支网络（CNN + Vision Transformer），编码器提取局部+全局特征；Global Context Aggregation Module (GCAM)聚合多尺度特征；能量感知和梯度增强融合策略；5层卷积+2个跳跃连接的解码器
- **是否聚类**：❌ 否（图像融合任务）
- **是否使用MRI**：✅ T1WI + DWI
- **是否涉及生存分析**：❌ 否
- **核心发现**：定性结果显示清晰的纹理细节和尖锐边界；定量结果：平均梯度4.5975、信息熵4.9073、互信息2.5181、视觉显著性0.77；优于DenseFuse、RFN-Nest、MSDNet、IFCNN、CDDFuse、SwinFusion
- **本文可引用核心观点**：**无监督**融合方法可以很好地保留源图像的显著信息；CNN+ViT混合架构适合医学图像融合
- **适合引用章节**：2.2.3（无监督融合策略）、4.2（编码器设计——无监督学习的启示）
- **与本文的关联点**：本文同样是**无监督**学习（聚类），该文展示了无监督医学图像融合的可行性——支持本文"无监督方法在医学图像中有效"的论点
- **备注**：⭐️ 重要！这是**无监督**融合方法，与本文的无监督聚类有相通之处

---

## 4. MMAformer: Multiscale Modality-Aware Transformer for Medical Image Segmentation

- **年份**：2024
- **来源**：Applied Sciences (MDPI)
- **任务类型**：多模态脑肿瘤分割
- **数据集**：BraTS 2020、BraTS 2021
- **核心方法**：多阶段编码器，使用跨模态下采样（CMD）块在不同尺度学习和整合互补信息；Multimodal Gated Aggregation (MGA) 块结合双注意力机制和多喝门控聚类，有效融合不同MRI序列的空间、通道和模态特征
- **是否聚类**：❌ 否（分割任务）
- **是否使用MRI**：✅ 多模态MRI
- **是否涉及生存分析**：❌ 否
- **核心发现**：在BraTS 2020和2021上平均Dice分别为86.3%和91.53%，超越现有state-of-the-art方法
- **本文可引用核心观点**：多阶段编码器和跨模态下采样可以有效捕捉和整合多尺度互补信息
- **适合引用章节**：2.2.3（Cross-Attention设计变体）、4.2（多模态编码器设计考量）
- **与本文的关联点**：本文采用单层Transformer + MLP后融合的简化设计，适合聚类任务（不需要像素级精度）

---

## 5. Zeng et al. — DBTrans: A Dual-Branch Vision Transformer for Multi-Modal Brain Tumor Segmentation

- **年份**：2023
- **来源**：MICCAI 2023
- **任务类型**：多模态脑肿瘤分割
- **数据集**：BraTS 2021
- **核心方法**：双分支编码器/解码器——Shifted Window Self-attention（局部）+ Shuffle Window Cross-attention（全局跨模态）；通道注意力模态加权
- **是否聚类**：❌ 否（分割任务）
- **是否使用MRI**：✅ 4模态MRI
- **核心发现**：局部自注意力 + 全局交叉注意力的双分支设计能同时捕获细节和跨模态依赖
- **本文可引用核心观点**：窗口注意力机制可以有效降低3D Transformer的计算复杂度
- **适合引用章节**：4.2（多模态编码器——Transformer设计效率考量）
- **与本文的关联点**：本文使用1层标准TransformerEncoderLayer而非窗口注意力，因为输入已是压缩后的4×576 Token（模态级而非体素级）

---

## 6. Xing et al. — NestedFormer: Nested Modality-Aware Transformer for Brain Tumor Segmentation

- **年份**：2022
- **来源**：MICCAI 2022
- **任务类型**：多模态脑肿瘤分割
- **数据集**：BraTS
- **核心方法**：嵌套模态感知Transformer——多层次跨模态特征融合
- **是否聚类**：❌ 否（分割任务）
- **是否使用MRI**：✅ 多模态MRI
- **核心发现**：嵌套/层级式的跨模态融合比单次融合更有效
- **本文可引用核心观点**：多层次跨模态交互可以逐步精炼融合特征
- **适合引用章节**：2.2.3（Cross-Attention设计变体）
- **与本文的关联点**：本文采用单层Transformer + MLP后融合的简化设计，适合聚类任务（不需要像素级精度）

---

## 7. Xie et al. — MACTFusion: Lightweight Cross Transformer for Adaptive Multimodal Medical Image Fusion

- **年份**：2024
- **来源**：IEEE JBHI
- **任务类型**：多模态医学图像融合（无监督）
- **数据集**：多模态医学图像
- **核心方法**：轻量级交叉Transformer——跨多轴注意力（cross multi-axis attention），自适应融合权重
- **是否聚类**：❌ 否（图像融合）
- **是否使用MRI**：✅ 多模态医学图像（含MRI）
- **是否涉及生存分析**：❌ 否
- **核心发现**：轻量级交叉注意力可以在保持融合质量的同时显著降低计算量
- **本文可引用核心观点**：交叉注意力在多模态医学图像融合中具有效率和效果的双重优势
- **适合引用章节**：2.2.3（Cross-Attention融合策略）、4.2（编码器轻量设计）

---

## 8. Li et al. — TranSiam: Aggregating Multi-modal Visual Features with Locality for Medical Image Segmentation

- **年份**：2022 (arXiv) / 2024 (Expert Systems with Applications)
- **来源**：Expert Systems with Applications
- **任务类型**：多模态医学图像分割
- **数据集**：BraTS、Vestibular Schwannoma
- **核心方法**：双路径CNN + Transformer；Locality-Aware Aggregation (LAA) 块——使用局部交叉注意力进行多模态融合
- **是否聚类**：❌ 否（分割任务）
- **是否使用MRI**：✅ 多模态MRI
- **核心发现**：局部交叉注意力在保留空间细节方面优于全局交叉注意力
- **本文可引用核心观点**：交叉注意力的粒度需要匹配任务需求（局部→分割、全局→聚类）
- **适合引用章节**：4.2（为什么本文使用全局Self-Attention——任务是聚类而非分割）

---

## 9. Zeng et al. — Missing as Masking: Arbitrary Cross-modal Feature Reconstruction for Incomplete Multimodal Brain Tumor Segmentation (M³FeCon)

- **年份**：2024
- **来源**：MICCAI 2024
- **任务类型**：模态不完整情况下的脑肿瘤分割
- **数据集**：BraTS 2018
- **核心方法**：MAE风格——将缺失模态视为被mask，通过跨模态特征重建恢复缺失信息
- **是否聚类**：❌ 否（分割任务）
- **是否使用MRI**：✅
- **核心发现**："缺失即掩码"范式可以在不完整模态下保持分割性能
- **本文可引用核心观点**：模态缺失处理是多模态医学图像分析的实际挑战
- **适合引用章节**：3.2（缺失模态处理策略）、6.2.1（数据局限性）

---

## 10. Zheng et al. — AMTNet: Automated Multi-modal Transformer Network for 3D Medical Images Segmentation

- **年份**：2023
- **来源**：Physics in Medicine & Biology
- **任务类型**：3D多模态医学图像分割
- **数据集**：前列腺MRI + BraTS 2021
- **核心方法**：3D U型Transformer + Adaptive Interleaved Transformer Fusion (AITF)——自适应交错Transformer融合
- **是否聚类**：❌ 否（分割任务）
- **是否使用MRI**：✅ 3D MRI
- **是否涉及生存分析**：❌ 否
- **核心发现**：3D Transformer直接在体素空间操作，交错融合比串行/并行更有效
- **本文可引用核心观点**：3D医学图像融合需要考虑三维空间结构
- **适合引用章节**：2.1.1（3D卷积的操作原理，与2D的区别）

---

## 11. Integrative Cross-Modal Fusion of Preoperative MRI and Histopathological Signatures

- **年份**：2024
- **来源**：Bioengineering (MDPI)
- **任务类型**：多模态融合（MRI + 全幻灯片图像）用于生存预测
- **数据集**：多模态GBM数据（MRI + WSI）
- **核心方法**：对比学习（InfoNCE损失）对齐MRI和WSI的语义嵌入空间；检索式代理病理表示策略（解决推理时WSI缺失问题）；对称InfoNCE损失
- **是否聚类**：❌ 否（生存预测）
- **是否使用MRI**：✅ 多序列MRI
- **是否涉及生存分析**：✅ 是（总体生存预测）
- **核心发现**：跨模态语义关联可以提升生存预测性能；检索策略可以有效处理模态缺失
- **本文可引用核心观点**：对比学习可以实现不同模态间的语义对齐；检索式方法可以处理推理时的模态缺失
- **适合引用章节**：2.2.3（多模态融合策略）、5.5（生存分析——跨模态方法对比）
- **与本文的关联点**：本文仅使用MRI模态，该文展示了如何整合多模态（MRI+WSI）提升生存预测——未来本文可扩展到多模态

---

## 12. 多模态融合策略对比总结

| 融合策略 | 代表方法 | 优势 | 劣势 | 本文采用情况 |
|---------|---------|------|------|------------|
| 早期融合（像素拼接） | 传统CNN | 简单直接 | 模态间无交互、灰度主导问题 | ❌ |
| 晚期融合（独立编码+决策融合） | 多分支网络 | 模态独立 | 忽略模态互补性 | ❌ |
| Cross-Attention融合 | CKD-TransBTS, M²FTrans | 模态间语义交互 | 计算复杂度较高 | ✅ Self-Attention变体 |
| Token融合 | M²FTrans | 处理模态缺失 | 需额外设计 | ❌ |
| **无监督融合** | **CvTFuse (2026)** | **无需标注、保留细节** | **需要精心设计的损失函数** | **⚠️ 参考其无监督思路** |

---

## 文献归纳小结

- **领域特点**：多模态MRI融合研究高度集中在**分割任务**（BraTS基准），聚类任务中的融合研究很少
- **融合趋势**：从拼接→注意力→交叉注意力→可学习Token融合，交互粒度越来越细
- **本文定位**：将Cross-Attention从分割迁移到聚类，利用Transformer Self-Attention实现模态级（而非体素级）语义交互，这是一个有价值的跨任务迁移
- **关键区别**：本文的融合发生在压缩后的特征Token空间（4×576），而非原始体素空间，这使得计算可行且更适合聚类目标