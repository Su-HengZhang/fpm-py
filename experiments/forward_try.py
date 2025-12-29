from scipy.io import loadmat
import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
from ptych.utils import image_center_region
from ptych.utils import ideal_lowpass_filter

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


# 实际物镜的数值孔径
numerical_aperture = 0.008
# 准单色光照明光波长
wavelength = 0.633e-3  # 以mm为单位
# 实际物镜有限孔径对应的截止频率
rho_cutoff = numerical_aperture / wavelength

# 理想像的频谱间隔
ideal_spectrum_interval = 1 / (ideal_image_interval * ideal_rows)

# 理想低通滤波器的截止频率
cutoff_freq = rho_cutoff / ideal_spectrum_interval
print(f"理想低通滤波器的截止频率为: {cutoff_freq}")
# 生成理想低通滤波器
lowpass_filter = ideal_lowpass_filter((ideal_rows, ideal_cols), cutoff_freq)

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
plt.title("Ideal Image")

# 显示spatial spectrum
plt.subplot(2, 2, 2)
plt.imshow(np.log(np.abs(ideal_image_fourier) + 1), cmap="gray")
plt.axis("off")
plt.title("Spatial Spectrum")

# 显示lowpass filter
plt.subplot(2, 2, 3)
plt.imshow(lowpass_filter, cmap="gray")
plt.axis("off")
plt.title("Lowpass Filter")

# 显示real image
plt.subplot(2, 2, 4)
plt.imshow(np.abs(real_image), cmap="gray")
plt.axis("off")
plt.title("Real Image")

plt.show()
