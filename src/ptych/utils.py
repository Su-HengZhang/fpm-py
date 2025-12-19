"""
傅里叶叠层显微术(Fourier Ptychography, FPM)工具函数

本模块为FPM实现提供辅助函数，
主要用于设备选择和配置。
"""

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
