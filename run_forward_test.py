# run_forward_test.py

import torch

from src.dataset import GBMDataset
from src.model import DeepClusteringModel

# ============================================================
# 基本配置
# ============================================================

DATA_DIR = "Graduation/data_test"     # 小数据集，forward 测试
NUM_MODALITIES = 4
FEATURE_DIM = 128
MAX_K = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 加载数据
# ============================================================

dataset = GBMDataset(DATA_DIR)

print(f"Total samples in data_test: {len(dataset)}")

x, mask, pid = dataset[0]
print("Patient ID:", pid)
print("Raw image shape:", x.shape)   # [4, H, W, D]

# batch 维度
x = x.unsqueeze(0).to(DEVICE)        # [1, 4, H, W, D]

x = torch.cat([x, x], dim=0) # 把同一个病人复制成两个，凑成 Batch=2

# ============================================================
# 构建模型
# ============================================================

model = DeepClusteringModel(
    num_modalities=NUM_MODALITIES,
    feature_dim=FEATURE_DIM,
    max_K=MAX_K,
    beta=1.0,
    gamma=0.1
).to(DEVICE)

model.eval()
# 继承自 nn.Module 的模型内置的一个非常关键的方法。它并不改变模型的结构，而是改变模型的工作模式。
# 当调用 model.eval() 时，模型会进入评估模式，这会影响到某些特定层的行为，比如 BatchNorm 和 Dropout。
# 在训练模式下，这些层会根据当前批次的数据进行统计和随机丢弃，而在评估模式下，它们会使用在训练过程中学到的统计数据，并且不进行随机丢弃。
# 这对于确保模型在测试或推理阶段的稳定性和一致性非常重要。

# ============================================================
# Forward 测试
# ============================================================

with torch.no_grad():
    z, q, pi, loss, L_dist, L_DP, L_entropy, L_var, L_uniform = model(x)
    # torch.no_grad() 是一个上下文管理器，表示在这个代码块中不需要计算梯度。它的作用是告诉 PyTorch 不要为这个代码块中的操作构建计算图，也就是说，不会记录这些操作以供反向传播使用。
    # 这对于评估模型或者进行推理时非常有用，因为在这些场景下通常不需要计算梯度，这样可以节省内存和计算资源，提高效率。

print("\n===== Forward Test Result =====")
print("Embedding z shape:", z.shape)    # [1, 128]
print("q shape:", q.shape)              # [1, K]
print("pi shape:", pi.shape)            # [K]

print("pi:", pi.detach().cpu().numpy())
# .detach(): 从当前计算图中分离出 pi，使其成为一个新的张量，不再与原来的计算图相关联。这意味着对 pi 的任何操作都不会影响到原来的计算图，也不会被记录用于反向传播。
# .cpu(): 显存（GPU）里的数据无法直接转成 NumPy 格式，将 pi 从 GPU 内存移动到 CPU 内存，这样就可以使用 NumPy 来处理它了。
# .numpy(): 将 pi 从 PyTorch 的张量格式转换成 NumPy 的数组
print("q :", q.detach().cpu().numpy())
print("Loss:", loss.item())
print("L_uniform:", L_uniform.item()) # .item() 是 PyTorch 张量的一个方法，用于获取张量中的单个数值。
# 当 loss 是一个标量张量（即只有一个元素）时，调用 loss.item() 会返回这个元素的值作为一个 Python 数字（比如 float）。
# 这对于打印损失值或者将其用于其他计算非常方便，因为它将 PyTorch 张量转换成了一个普通的 Python 数据类型。