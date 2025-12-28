import numpy as np


def extract_center_region(image, roi_rows=None, roi_cols=None):
    """
    以图像中心为基准提取指定大小的区域，自动处理边界情况

    功能说明:
    - 当ROI小于图像尺寸时：从图像中心裁剪出指定大小的区域
    - 当ROI大于图像尺寸时：保持图像中心不变，在边界外补零至指定大小
    - 始终保证原图像中心与提取区域中心完全对齐

    参数:
        image: 输入的2D numpy数组
        roi_rows: 提取区域的行数（默认为image的行数）
        roi_cols: 提取区域的列数（默认为image的列数）

    返回:
        center_region: 提取的中心区域，尺寸为 (roi_rows, roi_cols)

    示例:
        >>> img = np.arange(25).reshape(5, 5)
        >>> small = extract_center_region(img, 3, 3)  # 裁剪
        >>> large = extract_center_region(img, 7, 7)  # 补零
    """
    h, w = image.shape

    # 使用默认值
    roi_rows = h if roi_rows is None else roi_rows
    roi_cols = w if roi_cols is None else roi_cols

    # 计算图像中心
    h_center = h // 2
    w_center = w // 2

    # 计算ROI中心
    roi_r_center = roi_rows // 2
    roi_c_center = roi_cols // 2

    # 计算ROI在原图中的范围（以对齐中心为基准）
    r_min = h_center - roi_r_center
    r_max = r_min + roi_rows
    c_min = w_center - roi_c_center
    c_max = c_min + roi_cols

    # 计算需要补零的量和有效区域
    pad_top = max(0, -r_min)
    pad_bottom = max(0, r_max - h)
    pad_left = max(0, -c_min)
    pad_right = max(0, c_max - w)

    # 计算在原图中实际要提取的区域
    img_r_start = max(0, r_min)
    img_r_end = min(h, r_max)
    img_c_start = max(0, c_min)
    img_c_end = min(w, c_max)

    # 如果不需要补零，直接切片返回
    if pad_top == 0 and pad_bottom == 0 and pad_left == 0 and pad_right == 0:
        return image[img_r_start:img_r_end, img_c_start:img_c_end]

    # 需要补零的情况：创建结果数组并填充
    result = np.zeros((roi_rows, roi_cols), dtype=image.dtype)

    # 计算在结果数组中的填充位置
    res_r_start = pad_top
    res_r_end = roi_rows - pad_bottom
    res_c_start = pad_left
    res_c_end = roi_cols - pad_right

    # 将图像数据复制到结果数组
    result[res_r_start:res_r_end, res_c_start:res_c_end] = image[
        img_r_start:img_r_end, img_c_start:img_c_end
    ]

    return result


# 测试示例
if __name__ == "__main__":
    # 创建一个5x5的测试图像
    test_image = np.arange(25).reshape(5, 5)
    print("原始图像 (5x5):")
    print(test_image)
    print(f"中心位置: ({test_image.shape[0] // 2}, {test_image.shape[1] // 2})")
    print(f"中心值: {test_image[2, 2]}")
    print()

    # 测试1: 提取3x3的中心区域（裁剪）
    result1 = extract_center_region(test_image, roi_rows=3, roi_cols=3)
    print("提取3x3中心区域（裁剪）:")
    print(result1)
    print(f"中心位置: (1, 1), 中心值: {result1[1, 1]} (应为12)")
    print()

    # 测试2: 提取7x7的区域（补零）
    result2 = extract_center_region(test_image, roi_rows=7, roi_cols=7)
    print("提取7x7区域（补零）:")
    print(result2)
    print(f"中心位置: (3, 3), 中心值: {result2[3, 3]} (应为12)")
    print()

    # 测试3: 提取5x8的区域（行不变，列补零）
    result3 = extract_center_region(test_image, roi_rows=5, roi_cols=8)
    print("提取5x8区域（列方向补零）:")
    print(result3)
    print(f"中心位置: (2, 4), 中心值: {result3[2, 4]} (应为12)")
    print()

    # 测试4: 提取8x3的区域（行补零，列裁剪）
    result4 = extract_center_region(test_image, roi_rows=8, roi_cols=3)
    print("提取8x3区域（行方向补零，列裁剪）:")
    print(result4)
    print(f"中心位置: (4, 1), 中心值: {result4[4, 1]} (应为12)")
    print()

    # 测试5: 验证奇偶尺寸组合
    result5 = extract_center_region(test_image, roi_rows=6, roi_cols=6)
    print("提取6x6区域（偶数尺寸补零）:")
    print(result5)
    print(f"原图中心12的位置应该在结果的 (3, 3)")
    print(f"result5[3, 3] = {result5[3, 3]} (应为12)")
    print()
    print()

    # 性能测试
    import time

    large_image = np.random.rand(1000, 1000)

    start = time.time()
    for _ in range(100):
        _ = extract_center_region(large_image, 800, 800)
    print(f"裁剪操作 100次耗时: {time.time() - start:.4f}秒")

    start = time.time()
    for _ in range(100):
        _ = extract_center_region(large_image, 1200, 1200)
    print(f"补零操作 100次耗时: {time.time() - start:.4f}秒")
