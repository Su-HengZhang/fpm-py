"""
傅里叶叠层显微术(Fourier Ptychography, FPM)逆问题求解器

本模块实现了傅里叶叠层显微术中逆问题的优化算法，该算法从不同照明角度捕获的多个低分辨率强度图像中重建高分辨率物体。

逆问题使用基于梯度的优化方法求解，步骤如下：
1. 初始化对物体和/或瞳孔函数的猜测
2. 使用前向模型预测这些猜测会产生什么样的图像
3. 使用损失函数比较预测结果与实际捕获的图像
4. 使用反向传播更新猜测值以最小化损失
5. 重复上述步骤直到收敛
"""

# 导入前向模型以模拟图像形成
from ptych.forward import forward_model
import torch
from tqdm import tqdm  # 进度条工具


def solve_inverse(
    captures: torch.Tensor,  # 输入捕获图像 [B, h, w] - 浮点强度图像
    object: torch.Tensor,    # 物体的初始猜测 [H, W] - 复数场
    pupil: torch.Tensor,     # 瞳孔的初始猜测 [H, W] - 复数场
    kx_batch: torch.Tensor,  # k-向量的x分量 [B] - 浮点数
    ky_batch: torch.Tensor,  # k-向量的y分量 [B] - 浮点数
    learn_object: bool = True,     # 是否在优化过程中更新物体
    learn_pupil: bool = True,      # 是否在优化过程中更新瞳孔
    learn_k_vectors: bool = False, # 是否在优化过程中更新k-向量
) -> tuple[torch.Tensor, torch.Tensor, dict[str, list[float]]]:
    """
    使用基于梯度的优化方法求解傅里叶叠层显微术的逆问题。
    
    从多个低分辨率捕获图像中重建高分辨率物体（可选地包括瞳孔函数）。

    参数:
        captures: 来自不同照明角度的低分辨率强度图像
        object: 复数物体场的初始猜测
        pupil: 复数瞳孔函数的初始猜测
        kx_batch: 每个照明角度的波矢x分量
        ky_batch: 每个照明角度的波矢y分量
        learn_object: 是否在优化过程中更新物体
        learn_pupil: 是否在优化过程中更新瞳孔
        learn_k_vectors: 是否在优化过程中更新k-向量

    返回:
        tuple[torch.Tensor, torch.Tensor, dict[str, list[float]]]:
            - 重建的复数物体
            - 重建的复数瞳孔
            - 训练指标（损失和学习率）
    """

    # 确保至少有一个参数在被优化
    assert learn_object or learn_pupil or learn_k_vectors, \
        "learn_object、learn_pupil或learn_k_vectors中至少有一个必须为True"

    # --------------------------
    # 配置和设置
    # --------------------------
    epochs = 100  # 优化迭代次数

    # 计算下采样因子 (output_size / capture_size)
    output_size = object.shape[0]
    capture_size = captures[0].shape[0]
    downsample_factor = output_size // capture_size
    
    # 打印配置以进行验证
    print("训练循环开始")
    print(f"捕获图像大小: {capture_size}")
    print(f"输出大小: {output_size}")
    print(f"下采样因子: {downsample_factor}")

    # --------------------------
    # 准备可学习参数
    # --------------------------
    # 用于保存所有将被优化的张量的列表
    learned_tensors: list[torch.Tensor] = []
    
    # 如果需要，将物体配置为可学习
    if learn_object:
        # 创建一个与计算图分离的克隆，并设置requires_grad=True
        object = object.clone().detach().requires_grad_(True)
        learned_tensors.append(object)
    
    # 如果需要，将瞳孔配置为可学习
    if learn_pupil:
        pupil = pupil.clone().detach().requires_grad_(True)
        learned_tensors.append(pupil)
    
    # 如果需要，将k-向量配置为可学习
    if learn_k_vectors:
        kx_batch = kx_batch.clone().detach().requires_grad_(True)
        ky_batch = ky_batch.clone().detach().requires_grad_(True)
        learned_tensors.append(kx_batch)
        learned_tensors.append(ky_batch)

    # --------------------------
    # 优化器设置
    # --------------------------
    # 使用指定的学习率初始化AdamW优化器
    # AdamW是Adam的一个变体，添加了权重衰减正则化
    optimizer = torch.optim.AdamW(learned_tensors, lr=0.1)

    # 添加学习率调度器以逐渐降低学习率
    # CosineAnnealingLR按照余弦曲线降低学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,  # 一个完整余弦周期的总轮数
        eta_min=0.01   # 最小学习率
    )

    # --------------------------
    # 训练指标
    # --------------------------
    # 用于存储训练进度的字典
    metrics: dict[str, list[float]] = {
        'loss': [],   # 每轮的损失值
        'lr': []      # 每轮的学习率
    }

    # --------------------------
    # 训练循环
    # --------------------------
    # 使用tqdm显示进度条
    for _ in tqdm(range(epochs), desc="求解中"):
        # --------------------------
        # 1. 前向传播：预测图像
        # --------------------------
        # 使用当前参数猜测来预测我们会捕获什么样的图像
        predicted_intensities = forward_model(
            object, 
            pupil, 
            kx_batch, 
            ky_batch, 
            downsample_factor=downsample_factor
        )

        # --------------------------
        # 2. 计算损失：比较预测与实际捕获的图像
        # --------------------------
        # 使用预测强度和实际强度之间的L1损失（平均绝对误差）
        # L1损失比L2损失对异常值更不敏感
        total_loss = torch.nn.functional.l1_loss(predicted_intensities, captures)

        # --------------------------
        # 3. 反向传播：计算梯度
        # --------------------------
        # 清除前一次迭代的梯度
        optimizer.zero_grad()
        
        # 通过反向传播计算梯度
        # 这计算了每个可学习参数对损失的影响
        total_loss.backward()
        
        # 使用优化器更新参数
        # 这将梯度应用于参数，使其向最小化损失的方向更新
        optimizer.step()
        
        # 根据调度器更新学习率
        scheduler.step()

        # --------------------------
        # 4. 记录指标
        # --------------------------
        # 存储损失值（转换为Python浮点数）
        metrics['loss'].append(total_loss.item())
        # 存储当前学习率
        metrics['lr'].append(scheduler.get_last_lr()[0])

    # --------------------------
    # 返回结果
    # --------------------------
    # 在返回前将张量从计算图中分离
    # 这可以防止以后意外的梯度计算
    return object.detach(), pupil.detach(), metrics
