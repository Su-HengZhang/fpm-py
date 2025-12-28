import torch
from ptych.utils import get_default_device

import matplotlib.pyplot as plt

device = get_default_device()
print(device)

imgs = torch.load("data/imgs.pth", map_location=device)
# imgs = torch.load("data/imgs_0.pth", map_location=device)
rows_led, cols_led, _, _ = imgs.shape
row_cen_led = rows_led // 2
col_cen_led = cols_led // 2

# 提取中间图像
nrows, ncols = 5, 5

# 计算中间图像的起始索引
start_row = row_cen_led - nrows // 2
start_col = col_cen_led - ncols // 2

# 提取中间图像
middle_imgs = imgs[start_row : start_row + nrows, start_col : start_col + ncols]


fig, axs = plt.subplots(nrows, ncols, figsize=(9, 9))
for i in range(nrows):
    for j in range(ncols):
        axs[i, j].imshow(middle_imgs[i, j].cpu().numpy(), cmap="gray")
        axs[i, j].axis("off")

fig.tight_layout()
plt.show()
