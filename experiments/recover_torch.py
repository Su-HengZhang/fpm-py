import torch
import torch.fft as fft
import numpy as np

from tqdm import tqdm
from matplotlib import pyplot as plt

from ptych.utils import led_array_dir_cos
from ptych.utils import spiral_indices_from_center
from ptych.utils import ideal_lowpass_filter

# 设置设备 (GPU如果可用)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

################################################################################
# 初始化重建像
################################################################################
recover_rows, recover_cols = 1024, 1024
# 重建像的采样间隔
recover_image_interval = 4e-3  # 以mm为长度单位
recover_image_interval = recover_image_interval / 16  #
# 理想像的频谱间隔
ideal_spectrum_interval = 1 / (recover_image_interval * recover_rows)

################################################################################
# LED阵列照明的参数
################################################################################
# LED作为准单色光照明, 中心波长
wavelength = 0.633e-3  # 以mm为单位
# LED 矩阵的行列数目(奇数行，奇数列)
led_rows, led_cols = 9, 9
# LED阵列两灯之间的间隔
led_d = 3.133  # 以mm为单位
# LED灯板到样本的距离
distance_led2sample = 62.66  # 以mm为单位
distance_led2sample = distance_led2sample / led_d  # 以led_d为单位
# LED的方向余弦矩阵
led_dir_cos = led_array_dir_cos(led_rows, led_cols, distance_led2sample)
# LED照明的空间频率矩阵
led_frequency = led_dir_cos / wavelength
led_frequency = led_frequency / ideal_spectrum_interval  # 以理想像的频谱间隔为单位

# 转换为torch tensor
led_frequency = torch.from_numpy(led_frequency).float().to(device)

################################################################################
# 实际物镜的参数
################################################################################
# 实际物镜的数值孔径
numerical_aperture = 0.1
# 实际物镜有限孔径对应的截止频率
cutoff_freq = numerical_aperture / wavelength
cutoff_freq = cutoff_freq / ideal_spectrum_interval  # 以理想像的频谱间隔为单位

################################################################################
# 图像重建
################################################################################
# 如果数据是numpy格式，需要转换
intensity_image_data = np.load(f"data/intensity_image_list_{led_rows}x{led_cols}.npz")
intensity_image_list = (
    torch.from_numpy(intensity_image_data["intensity_image_list"]).float().to(device)
)

spiral_indices = spiral_indices_from_center((led_rows, led_cols))

# 将中心LED照明（正入射）低分辨率图像作为重建像的初始值
mc, nc = spiral_indices[0]
center_intensity_image = intensity_image_list[mc, nc, :, :]

recover_image = torch.sqrt(center_intensity_image)
recover_image_fourier = fft.fftshift(fft.fft2(fft.ifftshift(recover_image)))

epoch = 10
for _ in tqdm(range(epoch), desc="Epoch"):
    recover_image_fourier_new = recover_image_fourier.clone()
    # 遍历所有LED照明
    for m, n in tqdm(spiral_indices, desc="LED"):
        # 生成理想低通滤波器
        # 注意: 如果ideal_lowpass_filter返回numpy数组，需要转换为torch
        lowpass_filter = ideal_lowpass_filter(
            (recover_rows, recover_cols),
            cutoff_freq,
            (led_frequency[m, n, 0].item(), led_frequency[m, n, 1].item()),
        )
        # 如果lowpass_filter是numpy数组，转换为torch tensor
        if not isinstance(lowpass_filter, torch.Tensor):
            lowpass_filter = torch.from_numpy(lowpass_filter).float().to(device)

        # 低通滤波
        low_image_fourier = recover_image_fourier_new * lowpass_filter
        low_image = fft.fftshift(fft.ifft2(fft.ifftshift(low_image_fourier)))

        # 替换振幅
        intensity_image = intensity_image_list[m, n, :, :]
        low_image_new = torch.sqrt(intensity_image) * torch.exp(
            1j * torch.angle(low_image)
        )

        # 更新傅里叶域
        low_image_fourier_new = fft.fftshift(fft.fft2(fft.ifftshift(low_image_new)))
        recover_image_fourier_new = (
            recover_image_fourier_new * (1 - lowpass_filter)
            + low_image_fourier_new * lowpass_filter
        )

    recover_image_fourier = recover_image_fourier_new.clone()

# 计算重建图像
recover_image = fft.fftshift(fft.ifft2(fft.ifftshift(recover_image_fourier)))

# 转换到CPU用于显示
recover_image_cpu = recover_image.cpu()

# 显示重建图像的振幅与相位
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(torch.abs(recover_image_cpu).numpy(), cmap="gray")
plt.title("Recovered Image Amplitude")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(torch.angle(recover_image_cpu).numpy(), cmap="gray")
plt.title("Recovered Image Phase")
plt.axis("off")

plt.show()
