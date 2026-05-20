# Transformer与自注意力机制文献阅读笔记

> 收录Transformer、Self-Attention、Cross-Attention及其在多模态医学图像融合中的应用文献

---

## 1. Vaswani et al. — Attention Is All You Need

- **年份**：2017
- **来源**：NeurIPS
- **作者**：Vaswani A, Shazeer N, Parmar N, Uszkoreit J, Jones L, Gomez AN, Kaiser Ł, Polosukhin I
- **任务类型**：序列建模基础架构（机器翻译）
- **数据集**：WMT 2014 English-German, WMT 2014 English-French
- **核心方法**：完全基于注意力的序列到序列架构——Scaled Dot-Product Attention, Multi-Head Attention, Positional Encoding, Feed-Forward Network
- **是否聚类**：❌ 否
- **是否使用MRI**：❌ 否（NLP基础方法）
- **是否涉及生存分析**：❌ 否
- **核心发现**：纯注意力机制可以完全替代RNN/CNN进行序列建模；Multi-Head Attention允许模型同时关注不同表示子空间的信息；训练速度显著优于RNN（并行化）
- **本文可引用核心观点**：自注意力的内容依赖聚合机制可迁移至多模态特征交互场景，允许不同模态根据语义相似性动态检索和整合信息
- **适合引用章节**：2.2.1（自注意力机制原理）、3.3（模态融合模块设计）
- **与本文的关联点**：本文将Scaled Dot-Product Attention从NLP序列建模迁移至MRI模态特征交互——4个模态Token通过Self-Attention建立全连接信息交换

---

## 2. Lin et al. — CKD-TransBTS (已在 multimodal_fusion.md 中详细记录)

- **年份**：2023
- **来源**：IEEE Trans Med Imaging
- **在本文中的应用**：[20] 用于引用Cross-Attention在医学图像中的融合范式——Modality-Correlated Cross-Attention按MRI物理原理组织模态间交互

---

## 文献归纳小结

- **核心脉络**：Self-Attention提供动态内容依赖聚合(Vaswani 2017)，可迁移至多模态MRI特征交互
- **本文定位**：将Self-Attention应用于压缩后的模态Token序列（4×576），实现模态级而非体素级的语义交互
- **关键区别**：原始Transformer面向NLP序列（词Token），本文Token为经3D CNN编码的多模态MRI特征向量
