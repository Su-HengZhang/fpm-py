import numpy as np

from ptych.utils import led_array_dir_cos

################################################################################
# LED阵列照明的参数
################################################################################
# LED准单色光照明, 中心光波长
wavelength = 0.633e-3  # 以mm为单位
# LED 矩阵的行列数目(奇数行，奇数列)
led_rows, led_cols = 5, 5
# LED阵列两灯之间的间隔
led_d = 3.133  # 以mm为单位
# LED灯板到样本的距离
distance_led2sample = 62.66  # 以mm为单位
distance_led2sample = distance_led2sample / led_d  # 以led_d为单位
# LED的方向余弦矩阵
led_dir_cos = led_array_dir_cos(led_rows, led_cols, distance_led2sample)
# LED照明的空间频率矩阵
led_frequency = led_dir_cos / wavelength
# led_frequency = led_frequency / ideal_spectrum_interval  # 以理想像的频谱间隔为单位
# LED照明最大频率间隔, 应为从中心LED到四邻接LED
print(
    f"LED照明最大频率间隔为: {led_frequency[led_rows // 2 + 1, led_cols // 2, 0]:.1f} lp/mm"
)
# LED照明最大频率, 应为4个角落LED的照明
print(f"LED照明最高频率为: {np.sqrt(2) * led_frequency[-1, -1, 0]:.1f} lp/mm")

################################################################################
# 物镜的参数
################################################################################
# 物镜的数值孔径
numerical_aperture = 0.1
# 物镜数值孔径对应的截止频率
cutoff_freq = numerical_aperture / wavelength
print(f"物镜数值孔径对应的截止频率为: {cutoff_freq:.1f} lp/mm")
# 若将显微镜看成一个衍射受限系统, 透过样品的物光经物镜, 管镜最终成像到相机的靶面, 则该光学图像
# 的空间分辨率起决于物镜的数值孔径, 为低分辨率图像.
# 若要求相机能无失真的记录该光学图像, 则相机的像素尺寸的最大值为:
# pixel_size_max = magnify * 1000 / (2 * cutoff_freq)  # 以um为单位
# 其中magnify为成像的放大率

################################################################################
# 傅里叶叠层成像FPM分辨率
################################################################################
# 傅里叶叠层成像的实际截止频率
effective_cutoff_freq = cutoff_freq + np.sqrt(2) * np.abs(led_frequency[0, 0, 0])
print(f"傅里叶叠层成像的截止频率为: {effective_cutoff_freq:.1f} lp/mm")
# 傅里叶叠层成像空间分辨率(以um为单位)
space_resolution = 1000 / (2 * effective_cutoff_freq)
print(f"傅里叶叠层成像空间分辨率: {space_resolution:.3f} um")
