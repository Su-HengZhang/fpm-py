from scipy.io import loadmat
import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
from utils.utils import image_center_region
from utils.utils import ideal_lowpass_filter
from utils.utils import led_array_dir_cos

################################################################################
# 理想像的参数
################################################################################
# 加载usaf4um.mat, 作为理想像
mat_data = loadmat("data/usaf4um.mat")
f = mat_data["usaf4um"]
# 理想图像ideal_image是物体经理想成像系统(无像差, 孔径无限大)成像到相机CMoS
# 传感器上, 放大率为1,
# 像由CMOS相机采集到的, 相机的像素大小为4um, 相机的采集区域为1024x1024
ideal_rows, ideal_cols = 1024, 1024
ideal_image = image_center_region(f, 1024, 1024)
# 理想像的采样间隔
ideal_image_interval = 4e-3  # 以mm为长度单位
ideal_image_interval = ideal_image_interval / 16  # 将采样间隔缩小16倍，组号加4
# 理想像的频谱间隔
ideal_spectrum_interval = 1 / (ideal_image_interval * ideal_rows)

################################################################################
# 实际物镜的参数
################################################################################
# 实际物镜的数值孔径
numerical_aperture = 0.1
# 准单色光照明光波长
wavelength = 0.633e-3  # 以mm为单位
# 实际物镜有限孔径对应的截止频率
cutoff_freq = numerical_aperture / wavelength
cutoff_freq = cutoff_freq / ideal_spectrum_interval  # 以理想像的频谱间隔为单位
print(f"实际物镜有限孔径对应的截止频率为: {cutoff_freq}")

################################################################################
# LED阵列照明的参数
################################################################################
# LED 矩阵的行列数目(奇数行，奇数列)
led_rows, led_cols = 5, 5
# LED阵列两灯之间的间隔
led_d = 3.133  # 以mm为单位
# LED灯板到样本的距离
distance_led2sample = 62.0  # 以mm为单位
distance_led2sample = distance_led2sample / led_d  # 以led_d为单位
# LED的方向余弦矩阵
led_dir_cos = led_array_dir_cos(led_rows, led_cols, distance_led2sample)
# LED照明的空间频率矩阵
led_frequency = led_dir_cos / wavelength
led_frequency = led_frequency / ideal_spectrum_interval  # 以理想像的频谱间隔为单位
print(f"LED照明最大频率间隔为: {led_frequency[led_rows//2+1, led_cols//2, 0]}")

# 生成理想低通滤波器
lowpass_filter = ideal_lowpass_filter(
    (ideal_rows, ideal_cols),
    cutoff_freq,
    (
        led_frequency[led_rows // 2 + 1, led_cols // 2, 0],
        led_frequency[led_rows // 2 + 1, led_cols // 2, 1],
    ),
)

# 低通滤波
ideal_image_fourier = fft.fftshift(fft.fft2(fft.ifftshift(ideal_image)))
real_image_fourier = ideal_image_fourier * lowpass_filter
# 逆傅里叶变换, 得到实际像
real_image = fft.fftshift(fft.ifft2(fft.ifftshift(real_image_fourier)))


# 显示
plt.figure(figsize=(6, 6))

# 显示ideal image
plt.subplot(2, 2, 1)
plt.imshow(ideal_image, cmap="gray")
plt.axis("off")
plt.title("Ideal Image (Group No.+4)")

# 显示spatial spectrum
plt.subplot(2, 2, 2)
plt.imshow(np.log(np.abs(ideal_image_fourier) + 1), cmap="gray")
plt.axis("off")
plt.title("Spatial Spectrum")

# 显示滤波后的频谱
plt.subplot(2, 2, 3)
plt.imshow(
    np.log(np.abs(ideal_image_fourier) + 1) * (lowpass_filter + 1),
    cmap="gray",
)
plt.axis("off")
plt.title("Filtered Spectrum")

# 显示real image
plt.subplot(2, 2, 4)
plt.imshow(np.abs(real_image), cmap="gray")
plt.axis("off")
plt.title("Real Image")

plt.show()
