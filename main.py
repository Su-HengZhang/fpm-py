#!/usr/bin/env python3
"""
傅里叶叠层显微术(Fourier Ptychographic Microscopy, FPM)演示程序主入口。

本程序展示了完整的FPM工作流程：
1. 加载样本图像
2. 模拟不同照明角度下的前向成像过程
3. 求解逆问题以重建原始物体
4. 可视化并比较结果

傅里叶叠层显微术是一种计算成像技术，通过组合在不同照明角度下拍摄的多个低分辨率图像，
重建出样本的高分辨率图像。
"""

# 导入FPM核心函数
from ptych import forward_model, solve_inverse
# 导入可视化工具
from ptych.analysis import plot_comparison, plot_curves
# 导入设备选择工具
from ptych.utils import get_default_device
# 导入PyTorch用于张量操作
import torch
# 导入图像读取功能
from torchvision.io import read_image, ImageReadMode
# 导入用于生成k-向量网格的工具
from itertools import product

# 自动选择最佳可用设备 (CUDA > MPS > CPU)
pytorch_device = get_default_device()
# 设置所有后续张量操作的默认设备
torch.set_default_device(pytorch_device)

# 打印所选设备以进行验证
print("运行设备: ", pytorch_device)

# --------------------------
# 1. 加载并准备样本
# --------------------------
# 加载灰度图像并将其归一化到[0, 1]范围
amplitude = read_image('data/bars.png', mode=ImageReadMode.GRAY).squeeze(0).float() / 255.0
# 生成与振幅成正比的相位（FPM演示中常用）
phase = torch.pi * amplitude
# 创建物体的复数表示：amplitude * exp(i * phase)
image_complex = (amplitude * torch.exp(1j * phase)).to(pytorch_device)

# 获取图像尺寸并打印进行验证
height, width = image_complex.shape
print(f"图像尺寸: {height}x{width}")

# ------------------------
# 2. 创建瞳孔函数
# ------------------------
# 定义圆形瞳孔的半径（模拟显微镜光圈）
radius = 50
# 为图像创建坐标网格
y_coords, x_coords = torch.meshgrid(
    torch.arange(height, dtype=torch.float32),
    torch.arange(width, dtype=torch.float32),
    indexing='ij'  # 使用'ij'索引方式以匹配矩阵坐标
)
# 计算图像的中心
center_y, center_x = height / 2, width / 2
# 计算每个像素到中心的距离
distance = torch.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
# 创建二值圆形瞳孔（半径内为1，外为0）
pupil = (distance <= radius).float()

# --------------------------------
# 3. 生成k-向量（照明角度）
# --------------------------------
# 创建一个从-50到50，步长为10的k-向量网格
# 这将创建11x11=121个不同的照明角度
k_vectors: list[tuple[int, int]] = [(k[0], k[1]) for k in product(range(-50, 51, 10), repeat=2)]
# 找到零频率（轴上）k-向量的索引
zero_idx = k_vectors.index((0, 0))
# 打印k-向量的总数
print(f"k-向量总数: {len(k_vectors)}")

# ------------------------------
# 4. 模拟前向成像
# ------------------------------
# 提取所有k-向量的x和y分量
kx_all = torch.tensor([k[0] for k in k_vectors]).float()
ky_all = torch.tensor([k[1] for k in k_vectors]).float()
# 使用前向模型在所有k-向量下生成捕获图像
# downsample_factor=2将图像尺寸减半（FPM中常用）
captures = forward_model(image_complex, pupil, kx_all, ky_all, downsample_factor=2)  # 输出形状: [B, H, W]

# --------------------------------
# 5. 求解逆问题（重建）
# --------------------------------
# 定义重建输出的大小（高于输入以实现超分辨率）
output_size = 1024
# 用均匀振幅(0.5)初始化物体猜测
object = 0.5 * torch.ones(output_size, output_size, dtype=torch.complex64)
# 用均匀透射率初始化瞳孔猜测
pupil = 0.5 * torch.ones(output_size, output_size, dtype=torch.complex64)

# 求解逆问题以重建物体
# 返回: (重建的物体, 重建的瞳孔, 训练指标)
pred_O, _, metrics = solve_inverse(captures, object, pupil, kx_all, ky_all)
# 归一化重建振幅以进行可视化
pred_O_amplitude = torch.abs(pred_O) / torch.max(torch.abs(pred_O))

# ------------------------------
# 6. 可视化结果
# ------------------------------
# 绘制原始图像、捕获图像（零k-向量）和重建图像的比较
plot_comparison([amplitude.cpu(), captures[zero_idx].cpu(), pred_O_amplitude.cpu()], 
                ['原始图像', '捕获图像（轴上）', '重建物体'], 
                'tmp/adamw.png')

# 可选: 绘制物体和瞳孔振幅的比较
# plot_comparison([pred_O_amplitude.cpu(), pred_P_amplitude.cpu()], 
#                ['物体振幅', '瞳孔振幅'])

# 绘制训练指标（损失曲线）
plot_curves(metrics)
