

import numpy as np
import matplotlib.pyplot as plt


def read_depth_file(filename, width, height):
    """
    读取深度图文件并返回二维深度数组
    :param filename: 深度图文件路径
    :param width: 图像宽度
    :param height: 图像高度
    :return: 二维深度数组
    """
    # 读取二进制文件中的深度数据
    depth_data = np.fromfile(filename, dtype=np.float32)
    print(width * height)

    # 检查数据大小是否匹配
    if depth_data.size != width * height:
        raise ValueError(f"数据量不匹配：数组大小为 {depth_data.size}，但期望形状为 ({height}, {width})")

    # 将一维数组转换为二维数组
    depth_map = depth_data.reshape((height, width))

    # 将无穷大值设为0
    depth_map[depth_map == np.inf] = 0

    return depth_map


def display_depth_map(depth_map):
    """
    显示深度图
    :param depth_map: 二维深度数组
    """
    plt.imshow(depth_map, cmap='viridis')
    plt.colorbar(label='Depth')
    plt.title('Depth Map Visualization')
    plt.show()


# 设置深度图文件路径和图像尺寸
# 设置深度图文件路径和图像尺寸
depth_file = 'D:\\3DGS_data\\yard\\courtyard_dslr_depth\\courtyard\\ground_truth_depth\\dslr_images\\DSC_0301.JPG'  # 请替换为实际文件路径
image_width = 6048  # 替换为实际图像宽度
image_height = 4032 # 替换为实际图像高度

# 读取深度图数据
depth_map = read_depth_file(depth_file, image_width, image_height)

# 显示深度图
display_depth_map(depth_map)
