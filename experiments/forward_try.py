from scipy.io import loadmat
import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
from ptych.utils import image_center_region

mat_data = loadmat("data/usaf4um.mat")
f = mat_data["usaf4um"]

ideal_image = image_center_region(f, 1024, 1024)

ideal_image_fourier = fft.fftshift(fft.fft2(fft.ifftshift(ideal_image)))

# 显示ideal image and its spatial spectrum
plt.figure(figsize=(12, 6))
# 显示ideal image
plt.subplot(1, 2, 1)
plt.imshow(ideal_image, cmap="gray")
plt.axis("off")
plt.title("Ideal Image")

# 显示spatial spectrum
plt.subplot(1, 2, 2)
plt.imshow(np.log(np.abs(ideal_image_fourier) + 1), cmap="gray")
plt.axis("off")
plt.title("Spatial Spectrum")

plt.show()
