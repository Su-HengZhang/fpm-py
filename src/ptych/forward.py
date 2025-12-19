"""
傅里叶叠层显微术(Fourier Ptychography, FPM)前向模型

本模块实现了傅里叶叠层显微术的前向成像模型。

数学模型:
前向模型将物体O(x,y)、瞳孔函数P(u,v)和一组k-向量{k_i}映射到一组强度图像{I_i}:

对于每个k-向量(kx, ky):
1. 在频域中倾斜物体: O'(u,v) = O(u - kx, v - ky)
   (通过空间域与相位斜坡相乘实现)
2. 应用瞳孔滤波器: F(u,v) = O'(u,v) * P(u,v)
3. 逆傅里叶变换得到复图像场: f(x,y) = IFFT[F(u,v)]
4. 计算强度: I(x,y) = |f(x,y)|²

输入: O, P, {k_i} → 输出: {I_i}
"""

from typing import Callable, cast
from functools import partial
import torch
import torch.nn.functional as F

# --------------------------
# 傅里叶变换工具
# --------------------------
# 配置酉傅里叶变换（norm="ortho"确保能量守恒）
fft2 = cast(Callable[..., torch.Tensor], partial(torch.fft.fft2, norm="ortho"))  # 2D 快速傅里叶变换
ifft2 = cast(Callable[..., torch.Tensor], partial(torch.fft.ifft2, norm="ortho"))  # 2D 逆傅里叶变换
fftshift = cast(Callable[..., torch.Tensor], torch.fft.fftshift)  # 将零频率移到中心
ifftshift = cast(Callable[..., torch.Tensor], torch.fft.ifftshift)  # fftshift的逆操作


def forward_model(
    object_tensor: torch.Tensor,
    pupil_tensor: torch.Tensor,
    kx: torch.Tensor,
    ky: torch.Tensor,
    downsample_factor: int = 1
) -> torch.Tensor:
    """
    傅里叶叠层显微术的前向模型
    
    通过为每个照明角度（由k-向量表示）生成强度图像来模拟成像过程。

    参数:
        object_tensor (torch.Tensor): 物体的复数表示 [H, W]
        pupil_tensor (torch.Tensor): 瞳孔函数（光圈）[H, W]
        kx (torch.Tensor): x方向的波矢偏移 [B]
        ky (torch.Tensor): y方向的波矢偏移 [B]
        downsample_factor (int): 减小输出图像尺寸的因子

    返回:
        torch.Tensor: 预测的强度图像 [B, H, W]（如果指定了下采样因子则进行下采样）
    """

    # 获取物体的尺寸和数据类型
    H, W = object_tensor.shape
    dtype = object_tensor.dtype

    # --------------------------
    # 1. 创建坐标网格
    # --------------------------
    # 生成高度和宽度的1D坐标数组
    y_coords = torch.arange(H, dtype=torch.float32)
    x_coords = torch.arange(W, dtype=torch.float32)
    
    # 从1D坐标创建2D网格（矩阵式索引）
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')

    # --------------------------
    # 2. 生成相位斜坡
    # --------------------------
    # 重塑k-向量以进行广播 [B] → [B, 1, 1]
    # 除以图像大小以转换为空间频率单位
    kx_normalized = kx.view(-1, 1, 1) / W
    ky_normalized = ky.view(-1, 1, 1) / H

    # 计算每个像素和k-向量的相位: 2π * (kx*x + ky*y) / N
    phase = 2 * torch.pi * (kx_normalized * x_grid[None, :, :] + ky_normalized * y_grid[None, :, :])
    
    # 创建相位斜坡: exp(i * phase)
    # 这在空间域中实现了频率偏移
    phase_ramps = torch.exp(1j * phase.to(dtype))  # 形状: [B, H, W]

    # --------------------------
    # 3. 将相位斜坡应用于物体
    # --------------------------
    # 将物体扩展为 [B, H, W] 并与相位斜坡相乘
    # 空间域乘法 = 频域偏移
    # 这模拟了从不同角度照明物体
    tilted_objects = object_tensor[None, :, :] * phase_ramps  # 形状: [B, H, W]

    # --------------------------
    # 4. 傅里叶变换到频域
    # --------------------------
    # 对每个倾斜的物体应用2D FFT
    # fftshift将零频率移到频谱中心
    objects_fourier = fftshift(fft2(tilted_objects), dim=(-2, -1))  # 形状: [B, H, W]

    # --------------------------
    # 5. 应用瞳孔滤波器
    # --------------------------
    # 将每个傅里叶频谱与瞳孔函数相乘
    # 这模拟了显微镜的有限光圈
    filtered_fourier = pupil_tensor[None, :, :] * objects_fourier  # 形状: [B, H, W]

    # --------------------------
    # 6. 逆傅里叶变换回到空间域
    # --------------------------
    # 将滤波后的傅里叶频谱转换回复图像场
    complex_image_fields = ifft2(filtered_fourier)  # 形状: [B, H, W]

    # --------------------------
    # 7. 计算强度图像
    # --------------------------
    # 强度是复场的幅度平方
    predicted_intensities = torch.abs(complex_image_fields)**2  # 形状: [B, H, W]

    # --------------------------
    # 8. 可选的下采样
    # --------------------------
    # 如果指定了下采样因子，则减小图像尺寸（FPM中常用以匹配相机分辨率）
    if downsample_factor > 1:
        predicted_intensities = F.avg_pool2d(
            predicted_intensities, 
            kernel_size=downsample_factor, 
            stride=downsample_factor
        )

    return predicted_intensities
