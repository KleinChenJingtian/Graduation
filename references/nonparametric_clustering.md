# 非参数贝叶斯与自动簇数确定文献阅读笔记

> 收录Dirichlet Process、Gumbel-Sigmoid Stick-Breaking、自动K确定相关文献

---

## 1. Ronen et al. — DeepDPM: Deep Clustering with an Unknown Number of Clusters

- **年份**：2022
- **来源**：CVPR
- **任务类型**：深度聚类（自动确定K）
- **数据集**：MNIST、Fashion-MNIST、CIFAR-10/100、ImageNet-50/100
- **核心方法**：基于Dirichlet Process Mixture Model的深度聚类框架——使用Split/Merge Monte Carlo采样动态调整簇数，结合深度特征学习
- **是否聚类**：✅ 是
- **是否使用MRI**：❌ 否（自然图像）
- **是否涉及生存分析**：❌ 否
- **核心发现**：在未知K的真实场景中，DeepDPM能准确恢复真实簇数，且聚类质量与已知最优K的方法相当
- **本文可引用核心观点**：Dirichlet Process为深度聚类中的自动K确定提供了严格的非参数贝叶斯框架
- **适合引用章节**：2.5.1（DP混合模型原理）、4.3（Gumbel-DP自动K设计）
- **与本文的关联点**：DeepDPM使用MCMC采样在特征空间调整K，本文使用Gumbel-Sigmoid可微松弛实现端到端训练——两种不同的DP实现路径。本文方法的优势是端到端梯度优化
- **备注**：CVPR是top会议，此方法为本文方法的重要对比baseline

---

## 2. Nalisnick & Smyth — Stick-Breaking Variational Autoencoders

- **年份**：2017
- **来源**：ICLR
- **任务类型**：深度生成模型 + 非参数贝叶斯
- **数据集**：MNIST、SVHN
- **核心方法**：将Stick-Breaking过程嵌入VAE的潜在变量先验，实现无限混合模型的变分推断；Beta分布→Stick-Break权重
- **是否聚类**：✅ 是（通过潜在变量聚类）
- **是否使用MRI**：❌ 否
- **核心发现**：Stick-Breaking可以在VAE框架内实现可微的无限混合模型
- **本文可引用核心观点**：Stick-Breaking将DP的随机过程转化为可微操作，为深度网络中的非参数建模奠定基础
- **适合引用章节**：2.5.1（DP基础）、2.5.2（Gumbel-Sigmoid的演化脉络）
- **与本文的关联点**：本文用Gumbel-Sigmoid替代Beta分布实现Stick-Break的逐维可微门控

---

## 3. Jang et al. — Categorical Reparameterization with Gumbel-Softmax

- **年份**：2017
- **来源**：ICLR
- **任务类型**：离散变量可微松弛
- **核心方法**：Gumbel-Softmax分布——通过温度参数τ控制softmax的锐度，实现从离散类别分布中可微采样
- **是否聚类**：❌ 否（采样技术）
- **核心公式**：y_i = softmax((log π_i + g_i)/τ)，g_i ~ Gumbel(0,1)
- **本文可引用核心观点**：Gumbel-Softmax为离散随机变量的梯度优化提供了通用解决方案
- **适合引用章节**：2.5.2（Gumbel-Sigmoid的数学基础）
- **与本文的关联点**：本文将Gumbel-Softmax的softmax替换为sigmoid（逐元素而非归一化），实现Stick-Break的逐维可微门控

---

## 4. Maddison et al. — The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables

- **年份**：2017
- **来源**：ICLR
- **任务类型**：离散变量可微松弛（与Gumbel-Softmax同期独立提出）
- **核心方法**：Concrete分布——Gumbel-Softmax的等价形式
- **本文可引用核心观点**：温度τ控制离散松弛的偏差-方差权衡（低τ→低偏差高方差，高τ→高偏差低方差）
- **适合引用章节**：2.5.3（温度退火策略的理论依据）
- **与本文的关联点**：本文的τ退火策略（2.0→0.5）正是基于Concrete分布的这一性质

---

## 5. Potapczynski et al. — Invertible Gaussian Reparameterization: Revisiting the Gumbel-Softmax

- **年份**：2020
- **来源**：NeurIPS
- **作者**：Andres Potapczynski, Gabriel Loaiza-Ganem, John P. Cunningham
- **任务类型**：Gumbel-Softmax的改进（IGR——Invertible Gaussian Reparameterization）
- **核心方法**：提出模块化且更灵活的重参数化分布族，通过可逆变換将高斯噪声转换为one-hot近似；包含改进的softmax++和Stick-Breaking过程；可扩展到可数无限支持（非参数模型）
- **是否聚类**：❌ 否（重参数化技术）
- **核心发现**：IGR相比Gumbel-Softmax有理论优势（闭式KL散度）；Stick-Breaking过程允许将重参数化技巧用于可数无限支持的分布
- **本文可引用核心观点**：Gumbel-Softmax可以通过Stick-Breaking扩展到非参数模型；改进的softmax++提供更好的梯度特性
- **适合引用章节**：2.5.2（Gumbel-Sigmoid的改进方向）、2.5.3（温度参数设计）
- **与本文的关联点**：本文使用简单的sigmoid门控而非完整IGR，未来可考虑引入IGR以提升性能
- **备注**：NeurIPS 2020，顶级会议；此方法为Gumbel-Softmax的重要改进

---

## 6. Miao et al. — Discovering Discrete Latent Topics with Neural Variational Inference

- **年份**：2017
- **来源**：ICML
- **作者**：Yishu Miao, Edward Grefenstette, Phil Blunsom
- **任务类型**：神经主题模型 + Stick-Breaking（GSB和RSB）
- **核心方法**：Gaussian Stick-Breaking (GSB) 和 Recurrent Stick-Breaking (RSB) 用于神经主题模型；RSB使用RNN动态生成无限个breaks
- **是否聚类**：✅ 是（主题聚类/文档聚类）
- **核心发现**：Stick-Breaking构造可以嵌入神经网络实现无限主题发现；RNN可以建模无限长度的序列
- **本文可引用核心观点**：Stick-Breaking可以通过RNN实现无限维扩展；神经主题模型可以无界发现主题
- **适合引用章节**：2.5.1（Stick-Breaking的神经网络实现）、2.5.2（Gumbel-Sigmoid的上下文）
- **与本文的关联点**：本文使用简单的sigmoid门控实现Stick-Breaking，该文使用RNN实现更复杂的无限维扩展

---

## 7. Yang et al. — Gene-SGAN: A Multi-View Weakly-Supervised Deep Clustering for Disease Subtyping

- **年份**：2024
- **来源**：Nature Communications
- **任务类型**：多视图弱监督深度聚类 + 疾病亚型发现
- **数据集**：多中心脑疾病数据（含影像和基因组学）
- **核心方法**：多视图SGAN——将表型特征（影像）和基因型特征（基因组学）联合聚类，弱监督来自部分已知标签，可发现新亚型
- **是否聚类**：✅ 是（弱监督深度聚类）
- **是否使用MRI**：✅ 影像特征（含MRI衍生）
- **是否涉及生存分析**：✅ 相关（疾病进展分析）
- **核心发现**：弱监督信号可以引导聚类方向，同时保留发现新亚型的能力
- **本文可引用核心观点**：疾病亚型发现需要平衡无监督探索与临床先验引入
- **适合引用章节**：6.3方向3（半监督/弱监督聚类）
- **与本文的关联点**：本文为纯无监督方法，Gene-SGAN展示了弱监督信号的引入价值——未来可加入IDH/MGMT等已知标签

---

## 8. Sethuraman — A Constructive Definition of Dirichlet Priors (Stick-Breaking原始论文)

- **年份**：1994
- **来源**：Statistica Sinica
- **任务类型**：理论——DP的Stick-Breaking构造性定义
- **核心方法**：β_k ~ Beta(1, α)，π_k = β_k · Π_{j=1}^{k-1} (1-β_j)
- **是否聚类**：✅ 理论（DP混合模型基础）
- **核心公式**：Stick-Breaking过程：一根长度为1的棍子，每次折断剩余部分的比例β_k
- **本文可引用核心观点**：Stick-Breaking提供了DP的构造性定义，是截断实现的数学基础
- **适合引用章节**：2.5.1（DP Stick-Breaking过程示意）
- **与本文的关联点**：本文将Beta替换为Sigmoid(Gumbel)，实现可微Stick-Breaking

---

## 9. Rebaudo et al. — Variational Sensitivity Analysis for Stick-Breaking Priors

- **年份**：2023
- **来源**：Bayesian Analysis, 18(1), pp. 287–366
- **任务类型**：贝叶斯方法——Stick-Breaking先验灵敏度分析
- **核心方法**：变分贝叶斯方法评估DP先验参数（浓度参数α、Stick-Breaking分布选择）对后验推断的敏感度
- **是否聚类**：✅ 相关
- **核心发现**：截断水平（即本文的max_K）的选择对推断结果有实质影响
- **本文可引用核心观点**：DP截断水平（max_K）是需要谨慎选择的超参数
- **适合引用章节**：5.7.1（max_K上界敏感性分析——正是对这一问题的回应）
- **与本文的关联点**：为本文5.7.1的max_K敏感性实验提供了理论依据

---

## 10. Chakrabarti et al. — Graphical Dirichlet Process

- **年份**：2023
- **来源**：arXiv:2302.09111
- **任务类型**：非参数贝叶斯聚类（分组数据）
- **核心方法**：将DP扩展为图形化DP——使用有向无环图结构建模组间依赖关系，同时使用Stick-Breaking表示
- **是否聚类**：✅ 是
- **核心发现**：DP的Stick-Breaking表示可以扩展到结构化依赖场景
- **本文可引用核心观点**：DP的Stick-Breaking表示具有灵活的扩展性
- **适合引用章节**：2.5.1（DP的不同表示形式）
- **与本文的关联点**：本文的截断Stick-Break（max_K截断）是DP的有限维近似，Trade-off在灵活性和计算可行性之间

---

## 截断DP vs 全DP vs Gumbel-DP的本文化

| 方案 | 原理 | 可微性 | 本文使用 |
|------|------|--------|---------|
| 全DP（CRP表示） | 中国餐馆过程 | ❌ 离散采样 | 否 |
| 截断DP（Stick-Break） | π_K = v_K·Π(1-v_j) | ❌ Beta采样 | 否 |
| **Gumbel-Sigmoid Stick-Break** | v_k = σ((a_k+g_k)/τ) | ✅ 完全可微 | **是** |
| DeepDPM (Split/Merge) | MCMC采样 | ❌ 需额外操作 | 否 |

---

## 文献归纳小结

- **核心脉络**：DP理论(1994) → Stick-Breaking VAE(2017) → Gumbel-Softmax(2017) → DeepDPM(2022) → **本文Gumbel-DP(2025)**
- **本文贡献**：将Gumbel-Sigmoid Stick-Break首次应用于医学图像深度聚类，结合τ退火和π权重调度实现自动K
- **与DeepDPM的区别**：DeepDPM用MCMC在特征空间操作，本文用梯度下降端到端学习——更适合大规模3D医学数据