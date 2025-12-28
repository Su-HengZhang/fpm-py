"""
傅里叶叠层显微术(Fourier Ptychography, FPM)工具函数

本模块为FPM实现提供辅助函数，
主要用于设备选择和配置。
"""

import numpy as np
import torch  # PyTorch张量操作库


def get_default_device() -> torch.device:
    """
    自动选择最佳可用计算设备。

    此函数按以下顺序检查可用的硬件加速器：
    1. CUDA（NVIDIA GPU）- 大多数深度学习任务的最佳性能
    2. MPS（Apple Metal Performance Shaders）- 适用于Apple Silicon设备
    3. CPU - 当没有硬件加速可用时的回退选项

    返回:
        torch.device: 最佳可用计算设备
    """

    # 检查CUDA（NVIDIA GPU）是否可用
    if torch.cuda.is_available():
        return torch.device("cuda")

    # 检查MPS（Apple Metal）是否可用且已正确配置
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")

    # 如果没有硬件加速可用，则回退到CPU
    return torch.device("cpu")


def image_center_region(image, roi_rows=None, roi_cols=None):
    """
    以图像中心为基准提取指定大小的区域，自动处理边界情况

    功能说明:
    - 当ROI小于图像尺寸时：从图像中心裁剪出指定大小的区域
    - 当ROI大于图像尺寸时：保持图像中心不变，在边界外补零至指定大小
    - 始终保证原图像中心与提取区域中心完全对齐

    参数:
        image: 输入的2D numpy数组
        roi_rows: 提取区域的行数（默认为image的行数）
        roi_cols: 提取区域的列数（默认为image的列数）

    返回:
        center_region: 提取的中心区域，尺寸为 (roi_rows, roi_cols)

    示例:
        >>> img = np.arange(25).reshape(5, 5)
        >>> small = extract_center_region(img, 3, 3)  # 裁剪
        >>> large = extract_center_region(img, 7, 7)  # 补零
    """
    h, w = image.shape

    # 使用默认值
    roi_rows = h if roi_rows is None else roi_rows
    roi_cols = w if roi_cols is None else roi_cols

    # 计算图像中心
    h_center = h // 2
    w_center = w // 2

    # 计算ROI中心
    roi_r_center = roi_rows // 2
    roi_c_center = roi_cols // 2

    # 计算ROI在原图中的范围（以对齐中心为基准）
    r_min = h_center - roi_r_center
    r_max = r_min + roi_rows
    c_min = w_center - roi_c_center
    c_max = c_min + roi_cols

    # 计算需要补零的量和有效区域
    pad_top = max(0, -r_min)
    pad_bottom = max(0, r_max - h)
    pad_left = max(0, -c_min)
    pad_right = max(0, c_max - w)

    # 计算在原图中实际要提取的区域
    img_r_start = max(0, r_min)
    img_r_end = min(h, r_max)
    img_c_start = max(0, c_min)
    img_c_end = min(w, c_max)

    # 如果不需要补零，直接切片返回
    if pad_top == 0 and pad_bottom == 0 and pad_left == 0 and pad_right == 0:
        return image[img_r_start:img_r_end, img_c_start:img_c_end]

    # 需要补零的情况：创建结果数组并填充
    result = np.zeros((roi_rows, roi_cols), dtype=image.dtype)

    # 计算在结果数组中的填充位置
    res_r_start = pad_top
    res_r_end = roi_rows - pad_bottom
    res_c_start = pad_left
    res_c_end = roi_cols - pad_right

    # 将图像数据复制到结果数组
    result[res_r_start:res_r_end, res_c_start:res_c_end] = image[
        img_r_start:img_r_end, img_c_start:img_c_end
    ]

    return result


def ideal_lowpass_filter(shape, cutoff_frequency):
    """
    创建理想低通滤波器

    参数:
        shape: tuple, 滤波器的形状 (rows, cols)
        cutoff_frequency: float, 截止频率 (从中心点的距离)

    返回:
        filter: ndarray, 理想低通滤波器矩阵
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2  # 中心点坐标

    # 创建坐标网格
    x = np.arange(cols) - ccol
    y = np.arange(rows) - crow
    X, Y = np.meshgrid(x, y)

    # 计算每个点到中心的距离
    D = np.sqrt(X**2 + Y**2)

    # 创建理想低通滤波器：距离小于截止频率的为1，否则为0
    filter_mask = (D <= cutoff_frequency).astype(float)

    return filter_mask
