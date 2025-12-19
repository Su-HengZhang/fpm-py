"""
傅里叶叠层显微术(Fourier Ptychography, FPM)分析与可视化工具

本模块提供了用于可视化和分析傅里叶叠层显微术实验结果的函数，包括：
- 并排比较多张图像（原始、捕获、重建）
- 绘制训练指标曲线（如损失曲线）
- 将可视化结果保存到文件
"""

import os  # For directory creation utilities
from matplotlib import pyplot as plt  # Plotting library
import torch  # For tensor operations



def plot_comparison(images: list[torch.Tensor], labels: list[str], save_path: str | None = None):
    """
    并排绘制多张图像以进行视觉比较。
    
    此函数创建具有相应标签的水平图像布局，
    非常适合比较原始物体、捕获图像和重建结果。

    参数:
        images (list[torch.Tensor]): 要显示的图像列表（每张应为2D张量）
        labels (list[str]): 每张图像对应的标签列表
        save_path (str | None): 可选的保存图像路径
    """
    
    # 确保图像数量和标签数量相同
    assert len(images) == len(labels), "图像数量和标签数量必须匹配"
    
    # 计算图像数量
    n = len(images)

    # 创建具有适当大小的图形（每张图像5英寸宽）
    _ = plt.figure(figsize=(5 * n, 6))

    # 在各自的子图中绘制每张图像
    for i, (im, label) in enumerate(zip(images, labels)):
        # 创建子图（1行，n列，第i+1个位置）
        _ = plt.subplot(1, n, i + 1)
        
        # 以灰度显示图像
        _ = plt.imshow(im, cmap='gray')
        
        # 添加标签作为标题
        _ = plt.title(label)
        
        # 隐藏坐标轴以获得更清晰的可视化效果
        _ = plt.axis('off')

    # 调整子图之间的间距以提高可读性
    _ = plt.tight_layout()

    # 如果提供了保存路径，则保存图形
    if save_path:
        # 创建目录结构（如果不存在）
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 以高分辨率（300 dpi）和紧密边界框保存
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # 显示图像
    _ = plt.show()


def plot_curves(metric_dict: dict[str, list[float]], save_path: str | None = None):
    """
    绘制训练指标曲线（如损失、学习率）随迭代次数的变化。
    
    此函数创建线图，显示指标随时间的变化，
    用于监控逆问题求解器的训练进度。

    参数:
        metric_dict (dict[str, list[float]]): 指标名称到其各迭代值的字典
        save_path (str | None): 可选的保存图像路径
    """
    
    # 确保至少有一个指标要绘制
    assert len(metric_dict) > 0, "未提供任何指标"

    # 创建标准大小的图形
    _ = plt.figure(figsize=(10, 6))

    # 将每个指标绘制为单独的线
    for key, values in metric_dict.items():
        # 绘制值，使用指标名称作为标签
        _ = plt.plot(values, label=key)

    # 添加坐标轴标签
    _ = plt.xlabel('迭代次数 (Epoch)')
    _ = plt.ylabel('数值 (Value)')
    
    # 添加图例以区分不同的指标
    _ = plt.legend()
    
    # 添加低透明度网格以提高可读性
    _ = plt.grid(True, alpha=0.3)

    # 调整布局以防止内容被裁剪
    _ = plt.tight_layout()

    # 如果提供了保存路径，则保存图形
    if save_path:
        # 创建目录结构（如果不存在）
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 以高分辨率（300 dpi）和紧密边界框保存
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # 显示图像
    _ = plt.show()
