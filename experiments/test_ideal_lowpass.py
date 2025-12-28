import numpy as np
import matplotlib.pyplot as plt


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


def apply_lowpass_filter(image, cutoff_frequency):
    """
    对图像应用理想低通滤波器

    参数:
        image: ndarray, 输入图像
        cutoff_frequency: float, 截止频率

    返回:
        filtered_image: ndarray, 滤波后的图像
    """
    # 傅里叶变换
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)  # 将零频率移到中心

    # 创建滤波器
    filter_mask = ideal_lowpass_filter(image.shape, cutoff_frequency)

    # 应用滤波器
    f_filtered = f_shift * filter_mask

    # 逆傅里叶变换
    f_ishift = np.fft.ifftshift(f_filtered)
    filtered_image = np.fft.ifft2(f_ishift)
    filtered_image = np.abs(filtered_image)

    return filtered_image


# 示例使用
if __name__ == "__main__":
    # 创建测试图像：带噪声的方波
    size = 256
    image = np.zeros((size, size))
    image[64:192, 64:192] = 255

    # 添加高斯噪声
    noise = np.random.normal(0, 30, image.shape)
    noisy_image = image + noise

    # 应用不同截止频率的低通滤波器
    cutoff_freqs = [20, 40, 60]

    # 可视化
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))

    # 显示原始图像和噪声图像
    axes[0, 0].imshow(image, cmap="gray")
    axes[0, 0].set_title("原始图像")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(noisy_image, cmap="gray")
    axes[0, 1].set_title("噪声图像")
    axes[0, 1].axis("off")

    # 应用滤波器
    for i, cutoff in enumerate(cutoff_freqs):
        filtered = apply_lowpass_filter(noisy_image, cutoff)
        axes[0, i + 1].imshow(filtered, cmap="gray")
        axes[0, i + 1].set_title(f"滤波后 (截止={cutoff})")
        axes[0, i + 1].axis("off")

        # 显示滤波器
        filter_mask = ideal_lowpass_filter((size, size), cutoff)
        axes[1, i + 1].imshow(filter_mask, cmap="gray")
        axes[1, i + 1].set_title(f"滤波器 (截止={cutoff})")
        axes[1, i + 1].axis("off")

    # 显示频谱
    f_transform = np.fft.fft2(noisy_image)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)

    axes[1, 0].imshow(magnitude_spectrum, cmap="gray")
    axes[1, 0].set_title("频谱图")
    axes[1, 0].axis("off")

    plt.tight_layout()
    plt.show()

    print("理想低通滤波器演示完成！")
    print(f"截止频率越小，滤波效果越强，图像越模糊")
    print(f"截止频率越大，保留的高频信息越多，图像越清晰")
