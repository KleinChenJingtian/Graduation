# src/data_loader.py

import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset

# ============================================================
# 固定使用的模态（顺序非常重要！）
# ============================================================

MODALITIES = [
    "FLAIR_to_MNI.nii.gz",
    "T1C_to_MNI.nii.gz",
    "T1WI_to_MNI.nii.gz",
    "T2WI_to_MNI.nii.gz",
]

MASK_NAME = "tumor_to_MNI.nii.gz"


# ============================================================
# 基础工具函数
# ============================================================


def load_nifti(path):
    """读取 nii.gz 并返回 (data, affine, voxel_size)"""
    try:
        img = nib.load(path)
        data = img.get_fdata(dtype=np.float32)
        affine = img.affine  # 这个矩阵包含了空间变换信息，可以用来计算体素间距（voxel size）和进行重采样等操作
        # voxel_size: (x, y, z) 物理间距，单位mm
        voxel_size = np.abs(np.diag(affine[:3, :3]))
        return data, affine, voxel_size
    except Exception as e:
        raise RuntimeError(f"Failed to load NIfTI file: {path}\n{e}")


def resample_nifti(data, affine, target_voxel_size=(1.0, 1.0, 1.0), target_shape=None):
    """
    将 NIfTI 数据重采样到目标体素大小（默认 1mm 各向同性）

    参数:
        data:          原始 3D 数据 (H, W, D)
        affine:        NIfTI 的 affine 矩阵（包含空间变换信息）
        target_voxel_size: 目标体素间距 (mm)，默认 (1, 1, 1)
        target_shape:  可选，指定目标 shape，优先级高于 voxel_size

    返回:
        resampled_data: 重采样后的 3D 数据
        new_affine:    更新后的 affine（与 target_shape 对应）
    """
    from scipy.ndimage import zoom

    # 计算当前 voxel size
    current_voxel_size = np.abs(np.diag(affine[:3, :3]))

    if target_shape is None:
        # 计算各维度需要缩放的比例
        zoom_factors = current_voxel_size / np.array(target_voxel_size)
    else:
        # 根据目标 shape 计算缩放因子
        zoom_factors = np.array(data.shape) / np.array(target_shape)

    # 执行 trilinear 插值重采样 trilinear是一种常用的插值方法，适用于连续数据（如医学影像）。它通过在三维空间中对八个最近邻点进行线性插值来计算新位置的值。相比于最近邻插值，trilinear 插值能够产生更平滑、更自然的结果，减少锯齿状伪影，非常适合处理 MRI 这种连续的医学图像数据。
    resampled_data = zoom(data, zoom_factors, order=1)

    # 更新 affine 矩阵以反映新的形状和体素大小
    new_affine = affine.copy()
    # 对角线：体素大小 * shape 归一化因子
    for i in range(3):
        new_affine[i, i] = np.sign(affine[i, i]) * target_voxel_size[i]
        new_affine[i, 3] = affine[i, 3]  # 保持原点不变

    return resampled_data, new_affine


def check_modality_files(patient_dir):
    """
    检查患者目录下所有模态文件是否存在，并识别需要重采样的模态

    返回:
        valid_modalities: dict {模态名: 文件路径}，已排除不存在的模态
        missing_modalities: list [模态名]，缺失的模态
        needs_resample: dict {模态名: 文件路径}，需要重采样的模态（如 FLAIR_2MM）
    """
    valid_modalities = {}
    missing_modalities = []
    needs_resample = {}

    for name in MODALITIES:
        path = os.path.join(patient_dir, name)
        if os.path.exists(path):
            valid_modalities[name] = path
        else:
            # 检查是否有 2MM 变体（如 FLAIR_2MM_to_MNI.nii.gz）
            alt_name = name.replace("_to_MNI.nii.gz", "_2MM_to_MNI.nii.gz")
            alt_path = os.path.join(patient_dir, alt_name)
            if os.path.exists(alt_path):
                needs_resample[alt_name] = alt_path
            else:
                missing_modalities.append(name)

    return valid_modalities, missing_modalities, needs_resample



def normalize_intensity(img, mask, method="zscore"):
    """仅在 tumor ROI 内做强度归一化"""
    voxels = img[mask > 0]

    if voxels.size == 0:
        return img

    if method == "zscore":
        mean = voxels.mean()
        std = voxels.std() if voxels.std() > 0 else 1.0  # 计算肿瘤区域内的标准差
        img = (img - mean) / std
    else:
        raise ValueError("Only zscore normalization is supported")

    return img


# ============================================================
# 单患者加载（支持模态缺失掩码）
# ============================================================

def load_patient(patient_dir, target_voxel_size=(1.0, 1.0, 1.0)):
    """
    加载单个患者的影像和mask，支持2MM模态重采样

    参数:
        patient_dir:       患者目录路径
        target_voxel_size: 目标体素间距（mm），默认1mm各向同性

    返回:
        images:        torch.Tensor [4, H, W, D] — 4个模态的3D影像
        mask:          torch.Tensor [H, W, D]    — 肿瘤mask
        modality_mask: torch.Tensor [4]          — 各模态是否有效（1=有效，0=无效）
    """
    try:
        # ---------- load mask ----------
        mask_path = os.path.join(patient_dir, MASK_NAME)
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Missing mask: {mask_path}")

        mask_data, _, _ = load_nifti(mask_path)
        mask = mask_data > 0  # > 0 是为了把掩码变成布尔值（True/False），确定哪些地方是肿瘤
        mask = mask.astype(np.uint8)  # 把 True/False 转成数字 1 和 0。1 代表肿瘤，0 代表背景

        # ---------- load modalities ----------
        images = []
        modality_mask = []  # 标记每个模态是否有效

        for name in MODALITIES:
            img_path = os.path.join(patient_dir, name)

            # 情况1：标准路径存在，直接加载
            if os.path.exists(img_path):
                img, _, _ = load_nifti(img_path)
                modality_mask.append(1)

            else:
                # 情况2：检查是否有2MM变体（如FLAIR_2MM_to_MNI.nii.gz）
                alt_name = name.replace("_to_MNI.nii.gz", "_2MM_to_MNI.nii.gz")
                alt_path = os.path.join(patient_dir, alt_name)

                if os.path.exists(alt_path):
                    img, affine, voxel_size = load_nifti(alt_path)

                    # 检查是否真的需要重采样（间距明显偏离1mm）
                    is_2mm = np.any(np.abs(voxel_size - 2.0) < 0.5)

                    if is_2mm:
                        print(f"[INFO] Reampling {alt_name} (voxel_size={voxel_size}) → 1mm")
                        img, _ = resample_nifti(img, affine, target_voxel_size)

                    modality_mask.append(1)
                else:
                    # 情况3：模态完全缺失，用零填充
                    print(f"[WARNING] Modality missing, zero-filling: {name}")
                    img = np.zeros_like(mask_data, dtype=np.float32)
                    modality_mask.append(0)

            img = normalize_intensity(img, mask)
            img = img * mask  # 只保留肿瘤区域
            images.append(img)

        images = np.stack(images, axis=0)   # 把 4 个 3D 矩阵叠在一起，变成一个 4D 矩阵 [4, H, W, D]

        return (
            torch.from_numpy(images),                        # [4, H, W, D]
            torch.from_numpy(mask.astype(np.float32)),       # [H, W, D]
            torch.tensor(modality_mask, dtype=torch.float32) # [4]
        )

    except Exception as e:
        # 统一抛出，交给 Dataset.__getitem__ 处理
        raise RuntimeError(f"Corrupted patient data: {patient_dir}\n{e}")



# ============================================================
# PyTorch Dataset
# ============================================================

# 改动：因为数据中有样本缺少某些模态，所以在 Dataset 中先扫描一遍，找出哪些患者是完整的（4 个模态 + 1 个掩码都存在），只把这些患者加入到数据集中。这样就保证了后续训练时不会因为缺少模态而报错。
# patient2394缺少FLAIR_to_MNI.nii.gz，有的是FLAIR_2MM_to_MNI.nii.gz
# 现在模型的隐含假设是：所有模态已在 MNI 空间，同一体素网格（shape 一致）​，可逐 voxel 对齐
# FLAIR_2MM_to_MNI 的问题是素间距是 2 mm，即便 shape 可能“看起来”相近，也并不保证 voxel-to-voxel 对齐
# 如果之后想解决这个问题的话
# 正确流程（等之后补 待完成！！！）
# 显式重采样
# 把 FLAIR_2MM_to_MNI 重采样到 1 mm MNI
# 使用 ANTs, SimpleITK 或 nibabel + scipy

# 质量控制
# 检查插值伪影
# 确认 shape 与其他模态完全一致

# 在论文中明确说明
# 哪些数据是重采样的
# 插值方法（linear / BSpline）
class GBMDataset(Dataset):
    def __init__(self, root_dir, allow_2mm_variants=True):
        """
        参数:
            root_dir:          数据根目录
            allow_2mm_variants: 是否允许包含2MM变体的患者（如FLAIR_2MM_to_MNI.nii.gz）
        """
        self.root_dir = root_dir
        self.patients = []

        for d in os.listdir(root_dir):
        #  # os.listdir(root_dir) 会返回 root_dir 目录下的所有文件和文件夹的名字列表。比如如果 root_dir 是 "data"，返回 ["patient0012", "patient0013", "notes.txt"] 这样的列表。
            patient_dir = os.path.join(root_dir, d)
            if not os.path.isdir(patient_dir):
                continue

            # 检查 mask 是否存在（必须）
            if not os.path.exists(os.path.join(patient_dir, MASK_NAME)):
                continue

            # 检查模态：标准路径 OR 2MM 变体
            valid_modalities, missing, needs_resample = check_modality_files(patient_dir)

            if allow_2mm_variants:
                # 允许：只要mask存在+至少有一个模态（标准或2MM变体）即可
                if len(valid_modalities) + len(needs_resample) > 0:
                    self.patients.append(patient_dir)
            else:
                # 严格：必须所有标准模态都存在
                if len(missing) == 0:
                    self.patients.append(patient_dir)

        print(f"Using {len(self.patients)} patients")


    def __len__(self):   # 定义获取长度的函数，返回患者的数量，也就是数据集的大小
        return len(self.patients)

    
    def __getitem__(self, idx): # idx 是索引
        patient_dir = self.patients[idx]
        try:
            images, mask, modality_mask = load_patient(patient_dir)
            patient_id = os.path.basename(patient_dir)
            #  # 把路径末尾的文件夹名字切出来。比如路径是 "/data/patient0012"，切出来的 patient_id 就是 patient0012
            return images, mask, modality_mask, patient_id
        except RuntimeError as e:
            print(e)
            new_idx = np.random.randint(0, len(self.patients))
            return self.__getitem__(new_idx) # 无法正常加载。如果直接抛出异常，整个训练过程就会中断。通过随机换一个样本，我们可以跳过这个有问题的样本，继续训练其他正常的样本。虽然这种做法可能会导致某些数据被忽略，但在实际训练中，偶尔丢失一个样本通常不会对整体性能产生太大影响，尤其是当数据集较大时。
            new_idx = np.random.randint(0, len(self.patients))
            return self.__getitem__(new_idx)



# ============================================================
# 简单测试
# ============================================================

if __name__ == "__main__":
    dataset = GBMDataset("data_test")   # Python 中，相对路径是相对于终端当前所在位置计算的。
    # 在项目根目录D:\codeC\VsCodeP>下运行这个脚本，那么 "data_test" 就是正确的路径。
    # 如果进入到src目录下运行，将上面修改为../data_test 就可以了，其中 .. 代表上一层目录 
    # 如果要保持原来的路径"data_test"不变，可以在终端中直接运行 python src/dataset.py，这样相对路径就不会出问题了。
    x, mask, pid = dataset[0]
    print(pid)
    print("Images shape:", x.shape)  # [4, H, W, D]
    print("Mask shape:", mask.shape)
