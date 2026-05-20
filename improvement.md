# 非参数多视图分布对比聚类 — 改进方案

> 本文档用于后续讨论。每次讨论完成后，请在此文件中标注决策结论。

---

## 一、模型当前状态

### 已完成的修改
- Entropy Loss 符号修复（`-gamma * L_entropy`，驱动均匀分配）
- τ 退火改进（初始2.0，第15轮开始，衰减率0.97，最小0.5）
- α_inter sigmoid 平滑预热
- Logits 温度可学习化
- 新增 UniformDistributionLoss 正则化
- evaluate.py 完整重写（Kaplan-Meier + Log-rank + Cox回归 + 临床指标对比）

### 当前核心问题
- 只有一个簇占主导，不符合GBM多亚型预期

---

## 二、改进方向（按性价比排序）

### 🔴 高优先级

#### 1. 完整Wasserstein距离实现
**问题：** 代码中 `inter_loss` 只用了对角协方差近似，Trace项只做了Var之和，与PPT公式有落差。

**PPT原公式：**
```
W₂²(k,j) = ‖μ_k − μ_j‖² + Tr(Σ_k + Σ_j − 2(Σ_k^{1/2} Σ_j Σ_k^{1/2})^{1/2})
```

**当前实现（简化）：**
```
L_inter = -Σ_i≠j (‖μ_i − μ_j‖² + Σ_d Var(z_d))
```

**改进方案：**
```python
# 完整实现需要：
# 1. ClusterDistribution 改为输出 FULL 协方差矩阵（非对角）
#    - mu: [K, D]
#    - L_log_diag: 可学习对角元素（保持数值稳定）
#    - L_off_diag: 可学习协方差非对角元素（需要对称约束）
#    协方差矩阵需要做 SVD 或特征分解来计算 √Σ_k^{1/2} Σ_j Σ_k^{1/2}

# 2. 水я陆地球 mover's distance）的矩阵级实现
#    from scipy.linalg import sqrtm
#    cov_k_sqrt = sqrtm(cov_k @ cov_j)  # 矩阵平方根
#    trace_term = np.trace(cov_k + cov_j - 2 * cov_k_sqrt)
```

**是否实现：** □ 待讨论

---

#### 2. 生存损失（Cox Proportional Hazards Loss）
**前提：** 需要有患者生存时间数据（已在 evaluate.py 的临床CSV中支持）

**原理：** 如果聚类是有意义的，不同亚型患者的生存曲线应显著分离。Cox Loss 可以将这一信号反向传播到特征学习。

**改进方案：**
```python
class CoxSurvivalLoss(nn.Module):
    """
    Cox PH Loss for survival prediction
    h_i(t) = h_0(t) * exp(z_i @ β)
    Loss = - Σ_{observed} (z_i @ β - log Σ_{j∈R_i} exp(z_j @ β))
    """
    def forward(self, z, time, event):
        # 只在 event=1（死亡/进展）的患者上计算
        risk_score = z @ self.beta  # [N]
        # 偏序对：event=1 的患者应比 time 更小的 censored 患者有更高风险
        ...
```

**是否实现：** □ 待讨论（需要确认数据中有 time/event 列）

---

#### 3. 模态缺失掩码机制
**问题：** patient0151 的 FLAIR_2MM 分辨率不匹配，当前硬跳过

**改进方案：**
```python
# 在 MultiViewEncoder.forward 中：
# 1. 输入 x: [B, M, H, W, D]，M=4 固定
# 2. 为每个患者维护一个 mask: [B, M]，标记哪些模态有效
# 3. 无效模态的 score → -∞，attention weight → 0
# 4. 融合时只对有效模态加权平均

# 融合权重计算时：
valid_mask = torch.where(modality_exists, 1.0, -1e8)  # [B, M]
alpha = torch.softmax(S + valid_mask.unsqueeze(-1), dim=1)  # 无效模态 → 0
z_fused = (alpha * Z).sum(dim=1)
```

**是否实现：** □ 待讨论

---

### 🟡 中优先级

#### 4. MedicalNet 预训练 Encoder
**原理：** 3D医学影像预训练权重（来自大量CT/MRI数据），可大幅提升特征质量

**改进方案：**
```python
# 用 MedicalNet 预训练的 ResNet50 替换现有 Conv3d encoder
# github.com/Tencent/MedicalNet
# 加载预训练权重：
# model = MedicalNet3D(num_classes=1, pretrain_path="pretrain.pth")
# 冻结前面的层，只微调最后几层
```

**是否实现：** □ 待讨论

---

#### 5. SimCLR 对比损失
**原理：** 同患者多模态特征应接近（正样本），跨患者应远离（负样本）

**改进方案：**
```python
class SimCLR loss(nn.Module):
    def forward(self, z_list):
        # z_list: [M, B, D] — M个模态的特征
        # 正样本对：同一患者的不同模态
        # 负样本对：不同患者
        ...
        return loss
```

**是否实现：** □ 待讨论

---

#### 6. 改进 g(z) 评分函数
**当前：** 两层MLP (fused_dim → 32 → 1)

**可改进为：**
- 带有 Squeeze-Excitation (SE) Block 的评分网络
- Cross-Attention：让各模态特征互相"看"对方再打分

**是否实现：** □ 待讨论

---

### 🟢 低优先级（工作量大，效果不确定）

#### 7. Transformer / Swin-UNETR 替换 CNN Encoder
#### 8. Indian Buffet Process (IBP) 替代 Stick-Break
#### 9. Dirichlet Process Mixture Model (DPMM) 理论框架

---

## 三、验证方案增强

### 当前已支持
- [x] Silhouette Score、Davies-Bouldin Index
- [x] GBM 分类交叉验证（无标签时使用 z+q 特征）
- [x] Kaplan-Meier 生存曲线
- [x] Log-rank 检验
- [x] Cox 回归
- [x] 临床指标亚型间差异检验（Kruskal-Wallis + Mann-Whitney）

### 建议补充
- [ ] TCGA/RECOUNT 外部数据集泛化验证
- [ ] 影像组学（Radiomics）特征与亚型对比
- [ ] 与现有方法（DEC、DEPICT、DCC）的对比实验

---

## 四、理论深度补充（答辩用）

建议在论文中增加以下讨论：

1. **Wasserstein距离 vs 欧氏距离**：为什么分布对比用Wasserstein更合适？
   - 分布不必重叠也能计算距离
   - 考虑了协方差结构
   - 有几何直觉（质量运输）

2. **Gumbel-Sigmoid vs 真实Beta过程**：近似误差分析

3. **与经典聚类方法的联系与区别**：DEC、DEPICT、DCC、M3L

4. **收敛性讨论**：非参数K的收敛条件

---

## 五、数据准备

### 临床数据CSV格式（用于 evaluate.py）
```csv
patient_id,time,event,age,tumor_volume
patient0012,18.5,1,58,4521.3
patient0019,24.0,0,62,3890.1
...
```
- `time`: 生存时间（月）
- `event`: 1=死亡/进展，0=删失
- `age`: 年龄（可选）
- `tumor_volume`: 肿瘤体积（可选）

**请确认：** 数据集中是否有保存此类临床信息？文件路径是？

---

## 六、2MM问题（FLAIR分辨率）

**问题：** patient0151 的 `FLAIR_2MM_to_MNI.nii.gz` 素间距为2mm，其他模态为1mm

**方案选择：**
- [ ] A. 重采样：用 SimpleITK/nibabel 将 2MM 重采样到 1mm MNI 空间
- [ ] B. 排除：论文中注明该患者因分辨率不匹配被排除
- [ ] C. 插值：保留但用 trilinear 插值对齐（精度有限，不推荐）

---

## 七、决策记录

| 编号 | 改进项 | 决策 | 备注 |
|------|--------|------|------|
| 1 | 完整Wasserstein距离 | □ 实现 □ 跳过 |  |
| 2 | 生存损失Cox Loss | □ 实现 □ 跳过 | 需确认数据 |
| 3 | 模态缺失掩码 | □ 实现 □ 跳过 |  |
| 4 | MedicalNet预训练 | □ 实现 □ 跳过 |  |
| 5 | SimCLR对比损失 | □ 实现 □ 跳过 |  |
| 6 | g(z)评分网络改进 | □ 实现 □ 跳过 |  |
| 7 | 2MM问题处理 | □ A □ B |  |
| 8 | 临床数据CSV | 有/无，路径:______ |  |

---

## 下一步讨论议题

1. **（必选）** 确认哪些高优先级改进要实现
2. **（必选）** 2MM问题和临床数据情况
3. **（可选）** 是否接入 MedicalNet 预训练权重
4. **（重要）** Wasserstein距离的矩阵级实现细节
