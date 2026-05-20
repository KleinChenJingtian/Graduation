import torch
from torch.utils.data import DataLoader

from src.dataset import GBMDataset
from src.model import DeepClusteringModel

DEVICE = torch.device("cuda")

# test

dataset = GBMDataset("Graduation/data_test")
loader = DataLoader(dataset, batch_size=2, shuffle=True) # 这个语句的作用是创建一个数据加载器（DataLoader）对象，用于从给定的数据集（dataset）中加载数据。具体来说：
# batch_size=2: 这指定了每个批次（batch）中包含的数据样本数量。在这个例子中，每个批次将包含2个样本。
# shuffle=True: 这表示在每个训练周期（epoch）开始时，数据加载器会随机打乱数据的顺序。这有助于提高模型的泛化能力，避免模型过拟合到数据的特定顺序。

model = DeepClusteringModel(
    num_modalities=4,
    feature_dim=128,
    max_K=10,
    beta=1.0,
    gamma=0.1
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
# model.parameters()：把模型里所有的权重（卷积层、全连接层、折棍过程的参数等）都交给优化器管理。
# Adam：目前最先进、最稳健的优化算法。它能自动调节每个参数的学习速度（自适应学习率），非常适合 3D 医疗影像这种复杂的任务。

# for epoch in range(5):  # 整个数据集被“阅读”5次
#     model.train()
#     total_loss = 0.0  # 每个 Epoch 开始前清空累计损失

#     for x, mask, pid in loader:
#         x = x.to(DEVICE)

#         z, q, pi, loss = model(x)  # 【前向传播】：计算特征、概率分布和损失

#         # 【反向传播和优化】：根据损失调整模型参数
#         optimizer.zero_grad()   # 在每次反向传播之前，先把之前的梯度清零。因为 PyTorch 默认会累积梯度，如果不清零，梯度就会叠加，导致参数更新不正确。
#         loss.backward()    # 反向传播。它会计算损失函数相对于模型参数的梯度，并将这些梯度存储在每个参数的 .grad 属性中。也就是根据 Loss 的大小，利用链式法则算出模型里每一个参数“该往哪边改、改多少”
#         optimizer.step()   # 实际更新。优化器根据算出的梯度和学习率，真正动手修改模型里的权重。

#         total_loss += loss.item()  # 累加 Loss 用于观察趋势

#     model.nonparamK.anneal_tau()   # 【关键】：退火操作 随着训练进行，让 Gumbel-Softmax 的温度系数 τ 逐渐变小。
#     # 前期：τ 大，模型随机性强，到处探索。
#     # 后期：τ 小，模型变得果断，强制把权重集中在少数几个亚型上。

#     avg_loss = total_loss / len(loader)
#     print(f"Epoch {epoch}: avg_Loss = {avg_loss:.4f}")


for epoch in range(5):
    model.train()
    total_loss = 0
    total_dist = 0
    total_dp = 0
    total_ent = 0
    total_var = 0
    total_unif = 0

    for x, mask, pid in loader:
        x = x.to(DEVICE)

        z, q, pi, loss, L_dist, L_DP, L_entropy, L_var, L_uniform = model(x)
        

        optimizer.zero_grad()
        loss.backward()

        # 新增！！！
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        optimizer.step()

        total_loss += loss.item()
        total_dist += L_dist.item()
        total_dp += L_DP.item()
        total_ent += L_entropy.item()
        total_var += L_var.item()
        total_unif += L_uniform.item()

    print(f"""
Epoch {epoch}
L_total={total_loss/len(loader)}
L_dist ={total_dist/len(loader)}
L_DP   ={total_dp/len(loader)}
L_ent  ={total_ent/len(loader)}
L_var  ={total_var/len(loader)}
L_unif ={total_unif/len(loader)}
""")

