# from src.dataset import GBMDataset, load_patient

# # 测试 patient0151（已知有FLAIR_2MM）
# dataset = GBMDataset("data")
# print(f"数据集患者数: {len(dataset)}")

# # 直接加载patient0151，检查是否触发重采样
# import os
# for p in dataset.patients:
#     if "patient0151" in p:
#         images, mask, mod_mask = load_patient(p)
#         print(f"patient0151 modality_mask: {mod_mask}")
#         print(f"images shape: {images.shape}")
#         break




# 测试 check_modality_files 函数
# from src.dataset import check_modality_files
# valid, missing, needs = check_modality_files('D:/codeC/VsCodeP/data/patient0151')
# print('valid:', valid)
# print('missing:', missing)
# print('needs_resample:', needs)



# 测试 load_patient 函数 (load_nifti 或 resample_nifti)
# from src.dataset import load_patient
# import os

# p = 'D:/codeC/VsCodeP/data/patient0151'
# print('Files in dir:', os.listdir(p))

# images, mask, mod_mask = load_patient(p)
# print('modality_mask:', mod_mask)



# 测试一下是否会显示实际的 voxel_size 和 shape，确认2MM文件是否真的是2mm间距
# import nibabel as nib
# import numpy as np

# p = 'D:/codeC/VsCodeP/data/patient0151/FLAIR_2MM_to_MNI.nii.gz'
# img = nib.load(p)
# affine = img.affine
# voxel_size = np.abs(np.diag(affine[:3, :3]))
# print('voxel_size:', voxel_size)
# print('is_2mm:', np.any(np.abs(voxel_size - 2.0) < 0.5))
# print('shape:', img.get_fdata().shape)



# import torch
# from src.model import DeepClusteringModel

# model = DeepClusteringModel(num_modalities=4, feature_dim=128, max_K=10)
# x = torch.randn(2, 4, 182, 218, 182)

# # 直接测试encoder
# encoder = model.encoder
# print('encoder.num_modalities:', encoder.num_modalities)

# # 只走encoder前几行
# tokens = []
# for m in range(encoder.num_modalities):
#     feat = encoder.encoders[m](x[:, m:m+1])
#     z_global = encoder.pool_global(feat).flatten(1)
#     z_local = encoder.pool_local(feat).flatten(1)
#     z_m = torch.cat([z_global, z_local], dim=1)
#     print(f'm={m}, z_m shape: {z_m.shape}')
#     tokens.append(z_m)

# tokens_stacked = torch.stack(tokens, dim=1)
# print('tokens stacked shape:', tokens_stacked.shape)



# import torch
# from src.model import MultiViewEncoder

# encoder = MultiViewEncoder(num_modalities=4, feature_dim=128)
# x = torch.randn(2, 4, 182, 218, 182)
# modality_mask = torch.ones(2, 4)

# # 测试encoder（不完整，只到tokens）
# tokens = []
# for m in range(encoder.num_modalities):
#     feat = encoder.encoders[m](x[:, m:m+1])
#     z_global = encoder.pool_global(feat).flatten(1)
#     z_local = encoder.pool_local(feat).flatten(1)
#     z_m = torch.cat([z_global, z_local], dim=1)
#     tokens.append(z_m)

# tokens_stacked = torch.stack(tokens, dim=1)
# print('Before transformer:', tokens_stacked.shape)

# # 测试完整encoder forward
# z = encoder(x, modality_mask)
# print('After encoder:', z.shape)

import torch
from src.model import DeepClusteringModel

model = DeepClusteringModel(num_modalities=4, feature_dim=128, max_K=10)
print('模型参数量:', sum(p.numel() for p in model.parameters()))

x = torch.randn(2, 4, 182, 218, 182)
modality_mask = torch.ones(2, 4)

z, q, pi, loss, L_dist, L_DP, L_entropy, L_var, L_uniform = model(x, modality_mask)
print(f'z shape: {z.shape}')
print(f'q shape: {q.shape}')
print(f'loss: {loss.item():.4f}')
