import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 多视图 Encoder（轻量级 Transformer Cross-Attention 版）
# ============================================================

class MultiViewEncoder(nn.Module):
    """
    流程:
    1. 4个模态独立CNN + 多尺度Pool → 4个288d特征向量（Tokens）
    2. 4个Token组成序列 [Batch, 4, 288]，送入TransformerEncoderLayer（Self-Attention）
    3. Attention输出 [Batch, 4, 288]，展平拼接 → 1152d
    4. MLP降维 → 128d（聚类特征z）
    """
    def __init__(self, num_modalities=4, feature_dim=128):
        super().__init__()
        self.num_modalities = num_modalities

        # ---------- 每个模态独立 encoder（bottleneck结构：1→16→32→64） ----------
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(1, 16, 3, padding=1), nn.ReLU(),   # 1→16
                nn.MaxPool3d(2),                              # 降分辨率

                nn.Conv3d(16, 32, 3, padding=1), nn.ReLU(),   # 16→32
                nn.MaxPool3d(2),                              # 降分辨率

                nn.Conv3d(32, 64, 3, padding=1), nn.ReLU(),   # 32→64
            ) for _ in range(num_modalities)
        ])

        # 多尺度池化：全局(1×1×1) + 局部(2×2×2)
        self.pool_global = nn.AdaptiveAvgPool3d(1)   # [B,64,1,1,1]
        self.pool_local  = nn.AdaptiveAvgPool3d(2)   # [B,64,2,2,2]

        # 每个模态的特征维度：64(全局) + 64*8(局部) = 576
        self.token_dim = 64 * (1 + 8)

        # ---------- Transformer EncoderLayer（轻量级） ----------
        # 单层self-attention：4个模态token互相交流
        # d_model=288（等于token_dim）
        # nhead=8：每个head 36维，轻量且表达力足够
        # dim_feedforward=576：两倍于d_model，够用但不臃肿
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.token_dim,   # 576
            nhead=8,                  # 8头，每头72维
            dim_feedforward=self.token_dim * 2,  # 1152
            dropout=0.0,               # 无dropout（小数据集防欠拟合）
            batch_first=True          # [Batch, Seq, Feat] 格式
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # ---------- 降维MLP ----------
        # 4个token展平拼接：4 × 576 = 2304
        # 映射到聚类特征维度 feature_dim（默认256）
        fused_dim = self.token_dim * num_modalities  # 2304

        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, feature_dim * 2),  # 先扩展
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim)  # 再压缩
        )

    def forward(self, x, modality_mask=None):
        """
        参数:
            x:             [B, M, H, W, D] 影像张量
            modality_mask: [B, M] 可选，各模态是否有效（1=有效，0=无效）
                           无效模态特征置零，再送入Transformer
        返回:
            z: [B, feature_dim] 融合后的聚类特征
        """
        B = x.shape[0]

        # ---------- 1. 特征提取：每个模态独立CNN + 多尺度Pool ----------
        tokens = []
        for m in range(self.num_modalities):
            feat = self.encoders[m](x[:, m:m+1])  # [B,64,*,*,*]

            z_global = self.pool_global(feat).flatten(1)  # [B,64]
            z_local  = self.pool_local(feat).flatten(1)   # [B,256]
            z_m = torch.cat([z_global, z_local], dim=1)   # [B,320]

            tokens.append(z_m)

        # tokens: list of 4 × [B, 288] → stack → [B, 4, 288]
        tokens = torch.stack(tokens, dim=1)

        # ---------- 2. 模态掩码：无效模态置零 ----------
        if modality_mask is not None:
            # modality_mask: [B, M] → [B, M, 1]，直接与 [B, M, 576] 广播相乘
            tokens = tokens * modality_mask.unsqueeze(-1)  # 无效模态×0=0

        # ---------- 3. Transformer Self-Attention ----------
        # tokens: [B, 4, 576] → transformer → [B, 4, 576]
        tokens = self.transformer(tokens)

        # ---------- 4. 展平拼接 + MLP降维 + L2归一化 ----------
        # [B, 4, 576] → flatten(1) → [B, 2304]
        fused = tokens.flatten(1)
        z = self.mlp(fused)  # [B, 2304] → [B, 256]
        # 不再 L2 归一化：L2 将 per-dim std 锁死在 ~1/√256≈0.06，
        # 与 VICReg var target (std≥1.0) 冲突，导致特征方差崩溃至 0.002。
        # 防塌缩职责完全交给 VICRegLoss（var_weight=1.0, cov_weight=0.05）。

        return z


# ============================================================
# 非参数自动 K（Gumbel-Sigmoid DP）
# ============================================================

class NonParamK(nn.Module):
    def __init__(self, max_K=10, tau=2.0, tau_min=0.5, tau_anneal_start=15, tau_anneal_factor=0.97):
        super().__init__()
        self.max_K = max_K
        self.tau_min = tau_min
        self.tau_anneal_start = tau_anneal_start
        self.tau_anneal_factor = tau_anneal_factor

        # 【修复】：注册为buffer（能被checkpoint保存），用@property伪装成float（避免Tensor运算问题）
        self.register_buffer('_tau_buffer', torch.tensor([tau], dtype=torch.float32))

        # 【改进】：用数学公式反推 a_k，使得初始的 pi 是均匀分布 (1/K)
        import math
        init_a = []
        for k in range(max_K):
            vk = 1.0 / (max_K - k)
            ak = math.log(vk / (1.0 - vk + 1e-8))  # 逆 sigmoid
            init_a.append(ak)

        self.a_k = nn.Parameter(torch.tensor(init_a, dtype=torch.float32))
        
        # 一开始全部设置为 0，意味着每个簇的初始活跃程度相同。
        # 在训练过程中，模型会通过反向传播来调整这些 a_k 的值，使得模型能够自动学习出最合适的簇数量。
        # 通过调整 a_k 的值，模型可以学习到哪些簇更重要，哪些簇可以被忽略。当 a_k 很大时，对应的簇就更有可能被激活（v 接近 1）
        # 当 a_k 很小时，对应的簇就更有可能被抑制（v 接近 0）。这样，模型可以在训练过程中逐渐确定最适合数据的簇数量，而不需要事先指定 K 的具体值.

    @property
    def tau(self):
        return self._tau_buffer.item()

    def forward(self):
        g = -torch.log(-torch.log(torch.rand_like(self.a_k) + 1e-8) + 1e-8)
         # _like(self.a_k) 便捷写法，为了生成一个与 a_k 形状相同的随机张量，里面的值是从均匀分布 U(0, 1) 中采样的。
        # 加上 1e-8 是为了防止 u 等于 0 导致 log(0) 变成无穷大（数值稳定性）
        # logits = (self.a_k + g) / self.tau
        # v = torch.sigmoid(logits) # 使用 Sigmoid 函数将结果压缩到 (0, 1) 之间,代表了每个簇被激活的概率
        # 如果 a_k 很大，v 就接近 1；如果 a_k 很小（负数），v 就接近 0。通过调整 a_k 的值，模型可以学习到哪些簇更重要，哪些簇可以被忽略。
        logits = (self.a_k + g) / self.tau  # self.tau是float，无Tensor问题
        v = torch.sigmoid(logits)

        remaining_stick = torch.cumprod(
            torch.cat([torch.ones(1, device=v.device), 1 - v[:-1]]), dim=0
        )
        pi = v * remaining_stick
        return pi

    def anneal_tau(self, epoch=None):
        # 【改进】：从 epoch 15 开始退火，使用更温和的衰减率
        # 每轮都调用一次，但只在 epoch >= tau_anneal_start 时才衰减
        if epoch is not None and epoch < self.tau_anneal_start:
            return
        new_tau = max(self.tau * self.tau_anneal_factor, self.tau_min)
        self._tau_buffer.fill_(new_tau)  # 原地更新buffer


# ============================================================
# 簇分布模块
# ============================================================

class ClusterDistribution(nn.Module):
    def __init__(self, feature_dim, K):
        super().__init__()
        self.K = K
        self.feature_dim = feature_dim
        # self.mu = nn.Parameter(torch.randn(K, feature_dim)) # 可学习的参数，形状为 [K, feature_dim]，每一行 mu[k] 代表了第 k 个簇的中心在特征空间中的位置。初始值是随机生成的，随着训练的进行，模型会通过反向传播来调整这些 mu 的值，使得它们能够更好地代表数据中的簇结构。
        
        # 1、改！！！
        # 【修复】：mu初始化改为0.1
        # 之前0.01太小导致所有簇中心挤在原点，距离无区分度
        # 1.0太大导致dist2期望128，e^-128下溢或L_inter爆炸
        # 0.1：期望dist2≈12.8，e^-12.8≈2.7e-6，安全值
        self.mu = nn.Parameter(torch.randn(K, feature_dim) * 0.1)
        
        self.log_sigma = nn.Parameter(torch.zeros(K, feature_dim)) # 每一行 log_sigma[k] 代表了第 k 个簇在特征空间中的扩散程度（或者说簇的“宽度”）。初始值是 0，意味着初始时每个簇的扩散程度是 1（因为 sigma = exp(log_sigma)）。
        # 为什么要用 log_sigma？ 因为数学上标准差必须是正数。存储对数（log）值可以确保无论模型怎么计算，通过 torch.exp 还原回来后永远是正数

    def compute_mu(self, features, q):
        return (q.T @ features) / (q.sum(0).unsqueeze(1) + 1e-8) # q.T 的形状是 [K, Batch]，features 的形状是 [Batch, feature_dim]，所以 q.T @ features 的结果是 [K, feature_dim]，表示每个簇的加权特征和。
        # 除以 q.sum(0).unsqueeze(1) 是为了计算每个簇的加权平均特征，也就是新的簇中心 mu_batch。
        # sum(0) 表示沿着第 0 维进行求和，得到一个形状为 [K] 的向量，表示每个簇的总权重。
        # unsqueeze 是增加一个维度，把形状从 [K] 变成 [K, 1]，这样在除法运算中就可以正确地广播到每个特征维度上。

    def intra_loss(self, features, q, mu):
        loss = 0.0
        for k in range(self.K): 
            diff = features - mu[k]  # 计算每个病人特征与第 k 类中心点的距离
            loss += (q[:, k:k+1] * diff.pow(2)).sum() / (q[:, k].sum() + 1e-8)
            # q[:, k:k+1] * ...：这是一个软掩码，它会根据每个病人属于第 k 类的概率（q[:, k]）来加权计算损失。对于那些更可能属于第 k 类的病人（q[:, k] 较大），它们与簇中心 mu[k] 的距离会对损失贡献更大；对于那些不太可能属于第 k 类的病人（q[:, k] 较小），它们的距离对损失的贡献就较小。  
        return loss / self.K

    # def inter_loss(self):
    #     sigma = torch.exp(self.log_sigma)
    #     loss = 0.0
    #     for i in range(self.K):
    #         for j in range(i+1, self.K):
    #             loss += (self.mu[i] - self.mu[j]).pow(2).sum()
    #             loss += (sigma[i] - sigma[j]).pow(2).sum()
    #             # 为了计算速度和数值稳定性，我们假设协方差矩阵 Σ 是对角阵（即特征之间相互独立）。当 Σ 是对角阵时，复杂的 Tr(...) 部分可以简化为此处
    #     return -loss
    
    
    # 2、改！！！
    def inter_loss(self):
    # 【关键修复】：用 torch.clamp 限制 log_sigma 的范围，防止 exp() 指数爆炸
    # 限制在 [-5, 2] 之间，意味着 sigma 永远在 [0.006, 7.38] 之间，绝对安全！
        safe_log_sigma = torch.clamp(self.log_sigma, min=-5.0, max=2.0)
        sigma = torch.exp(safe_log_sigma)
        
        loss = 0.0
        count = 0
        for i in range(self.K):
            for j in range(i+1, self.K):
                loss += (self.mu[i] - self.mu[j]).pow(2).sum()
                loss += (sigma[i] - sigma[j]).pow(2).sum()
                count += 1
                
        return -loss / count


    

class EntropyLoss(nn.Module):
    def forward(self, q):
        """
        q: [B, K] soft assignment
        """
        q_bar = q.mean(dim=0)  # [K]
        return -torch.sum(q_bar * torch.log(q_bar + 1e-8))


class UniformDistributionLoss(nn.Module):
    """
    【新增】：均匀分布正则化器
    惩罚 q 的批次均值偏离均匀分布 1/K
    L_unif = KL(u || q_bar) = sum(u * log(u / q_bar)) = -sum(u * log(q_bar)) + log(K)
    这直接鼓励每个簇被选中的概率接近相等，防止单个簇主导
    """
    def forward(self, q):
        q_bar = q.mean(dim=0)           # [K]
        K = q_bar.numel()
        uniform = torch.full_like(q_bar, 1.0 / K)
        # KL(uniform || q_bar) = sum(uniform * log(uniform / q_bar))
        kl = uniform * (torch.log(uniform + 1e-8) - torch.log(q_bar + 1e-8))
        return kl.sum()


# ============================================================
# 反塌缩 + 表示稳定
# ============================================================

class AntiCollapseLoss(nn.Module):
    def forward(self, q, pi):
        """
        L_ac = Σ_i q_i * log(q_i / π_i)
        如果模型极度确信某个患者不属于某个簇，q 可能会输出绝对的 0.0（在 float32 精度下）。
        此时 0.0 * log(0.0) 在数学上极限是 0，但在代码里 log(0.0) 是 -inf，0.0 * -inf 会直接变成 nan
        用对数减法代替除法：log(q/pi) = log(q) - log(pi)
        +1e-8防止log(0)导致-inf，避免0*(-inf)=nan
        """
        pi = pi.unsqueeze(0)  # [1, K]
        # q * log(q/pi) = q * (log(q) - log(pi))
        loss = q * (torch.log(q + 1e-8) - torch.log(pi + 1e-8))
        return loss.sum(1).mean()


class FeatureVarianceLoss(nn.Module):
    def __init__(self, eps=1.0):
        super().__init__()
        self.eps = eps # 这是一个阈值。表示希望每个维度的特征至少要有多大的波动（标准差），如果某个维度的标准差小于 eps，那么这个损失就会变成正数，鼓励模型增加这个维度的特征多样性；如果标准差已经大于 eps，那么这个损失就会接近于 0，不会对模型产生额外的压力。通过调整 eps 的值，可以控制模型在特征空间中保持多样性的程度。      

    def forward(self, z):
        # 【修复】：unbiased=False 让分母为 N 而非 N-1
        # 当 N=1 时，方差=0，加1e-6后sqrt=安全值，避免除以0导致nan
        std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-6) # 计算这一批病人（Batch）在每个特征维度上的标准差。var(dim=0) 计算的是沿着 Batch 维度的方差，得到一个形状为 [feature_dim] 的向量，表示每个特征维度的方差。加上1e-6 是为了防止方差为 0 导致的数值不稳定问题。
        return torch.mean(F.relu(self.eps - std))
        # 如果当前特征的波动 std 已经大于 eps（比如 1.2 > 1.0），说明特征足够丰富，eps - std 为负，relu 结果为 0，不产生损失。
        # 惩罚场景：如果这一批病人的某个特征维度全是一模一样的数（比如全是 0.5），那么 std 就会接近 0。此时 eps - std 接近 1.0，relu 会产生一个很大的损失。
        # 效果：它逼迫 MultiViewEncoder 提取出的特征向量 z 必须具有区分度，不能偷懒输出同样的特征。


class VICRegLoss(nn.Module):
    """
    驯服版 VICReg：保护特征不塌缩，但不破坏聚类流形
    - var_weight：方差约束，每维 std > 1.0（好人）
    - cov_weight：协方差约束，权重降到 0.05（不再阻碍聚类）
    """
    def __init__(self, feature_dim, var_weight=1.0, cov_weight=0.05):
        super().__init__()
        self.feature_dim = feature_dim
        self.var_weight = var_weight
        self.cov_weight = cov_weight

    def forward(self, z):
        B = z.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=z.device)

        # 1. 方差约束：每维 std > 1.0
        std_z = torch.sqrt(z.var(dim=0) + 1e-4)
        loss_var = torch.mean(F.relu(1.0 - std_z))

        # 2. 协方差约束：off-diagonal 接近 0（刺头，权重降为 0.05）
        z_centered = z - z.mean(dim=0)
        corr_matrix = (z_centered.T @ z_centered) / (B - 1)
        off_diag = corr_matrix.fill_diagonal_(0)
        loss_cov = (off_diag.pow(2)).sum() / self.feature_dim

        return self.var_weight * loss_var + self.cov_weight * loss_cov


# ============================================================
# 总模型
# ============================================================

class DeepClusteringModel(nn.Module):
    def __init__(self, num_modalities=4, feature_dim=128, max_K=10, beta=1.0, gamma=0.05):
        super().__init__()
        self.encoder = MultiViewEncoder(num_modalities, feature_dim)
        self.nonparamK = NonParamK(max_K, tau=2.0, tau_min=0.5, tau_anneal_start=15, tau_anneal_factor=0.97)
        self.cluster = ClusterDistribution(feature_dim, max_K)
        self.entropy_loss = EntropyLoss()
        self.uniform_loss = UniformDistributionLoss()
        self.anticollapse = AntiCollapseLoss()
        self.varloss = FeatureVarianceLoss()
        self.vicreg = VICRegLoss(feature_dim)
        self.beta = beta
        self.gamma = gamma

        # 【改进】：alpha_inter 目标值，调度通过 get_alpha_inter 控制
        self.alpha_inter_target = 1.0
        self.alpha_inter = 0.01   # 初始 inter 权重

        # 【改进】：logits 温度作为可学习参数
        # 初始值-4.0：sigmoid(-4)=0.018, temp=0.018*20+1≈1.36
        # 低温让q更尖锐，给π更强的梯度刺激，π能更快追赶真实分布
        self.logits_temp = nn.Parameter(torch.tensor([-4.0]))

    def get_alpha_inter(self, epoch, warmup_epochs=10):
        # 【改进】：使用 sigmoid 平滑预热 alpha_inter，从 epoch 10 开始预热
        if epoch < warmup_epochs:
            t = (epoch - warmup_epochs / 2) / (warmup_epochs / 4)
            alpha = 0.01 + (self.alpha_inter_target - 0.01) * torch.sigmoid(torch.tensor(t, dtype=torch.float32)).item()
        else:
            alpha = self.alpha_inter_target
        self.alpha_inter = alpha
        return alpha

    def forward(self, x, modality_mask=None, epoch=0):
        """
        参数:
            x:             [B, M, H, W, D] 影像张量
            modality_mask: [B, M] 可选，各模态是否有效
            epoch:         当前epoch，用于两阶段L_ac策略
        """
        z = self.encoder(x, modality_mask)         # [B, feature_dim]
        pi = self.nonparamK()                      # 带Gumbel噪声的簇权重

        # ---------- 计算可学习版本的 clean_pi ----------
        # 【修复】：去掉no_grad，让clean_pi参与反向传播，a_k才能被更新
        # Phase 1（epoch<15）：clean_pi作为固定先验，约束q不要瞎分
        # Phase 2（epoch>=15）：clean_pi跟随q更新，让π反映真实分配
        clean_v = torch.sigmoid(self.nonparamK.a_k / self.nonparamK.tau)
        remaining_stick = torch.cat([
            torch.ones(1, device=clean_v.device),
            1 - clean_v[:-1]
        ]).cumprod(0)
        clean_pi = clean_v * remaining_stick  # [K]，参与梯度计算

        # soft assignment with π_k（z 先centering，防均值漂移让q变尖锐）
        mu = self.cluster.mu   # [K, feature_dim]
        z_centered = z - z.mean(dim=0, keepdim=True)
        dist2 = ((z_centered.unsqueeze(1) - mu.unsqueeze(0)) ** 2).sum(2)

        # 1. 平滑先验（0.001，避免数值爆炸）
        clean_pi = clean_pi + 0.001
        clean_pi = clean_pi / clean_pi.sum()

        # 2. 尖锐的分配（固定temp=1.0，让Q更尖锐）
        temp = 1.0  # 固定全程
        # pi_weight 渐进退火：前20轮温和，20-40轮接管，40轮后完全DP
        if epoch < 20:
            pi_weight = 0.1
        elif epoch < 40:
            pi_weight = 0.5
        else:
            pi_weight = 1.0

        # 放大距离差异
        logits = (-5.0 * dist2 + pi_weight * torch.log(clean_pi + 1e-8)) / temp
        q = F.softmax(logits, dim=1)

        # losses
        L_intra = self.cluster.intra_loss(z, q, mu)
        L_inter = self.cluster.inter_loss()

        # 3. 两阶段EM逻辑（与pi_weight退火同步）
        if epoch < 20:
            L_ac = self.anticollapse(q, clean_pi.detach())   # 固定π，训练q
        else:
            L_ac = self.anticollapse(q.detach(), clean_pi)   # 固定q，训练π

        L_var = self.varloss(z)
        L_vicreg = self.vicreg(z)
        L_entropy = self.entropy_loss(q)

        # 4. 课程式熵约束
        if epoch < 30:
            entropy_weight = 0.01
        else:
            entropy_weight = 0.005

        # 5. 终极 Loss
        # 【P1】L_intra 1.0→0.1：降低cluster pressure，保护feature geometry
        # 【P2】VICReg 1.0→5.0：适度加强防塌缩，不暴力破坏聚类结构
        # 【P3】epoch<20 warmup：仅 encoder+VICReg，聚类head在表示形成后再介入
        if epoch < 20:
            L_total = 5.0 * L_vicreg
        else:
            L_total = 0.1 * L_intra + 0.1 * L_inter + self.beta * L_ac - entropy_weight * L_entropy + 5.0 * L_vicreg

        return z, q, clean_pi, L_total, L_intra, L_inter, L_ac, L_var, L_vicreg
