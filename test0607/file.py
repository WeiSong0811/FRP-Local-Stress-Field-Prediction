import os
import numpy as np
import matplotlib.pyplot as plt

# 检查并创建必要的文件夹
def ensure_folder_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def load_array_and_plot_image(input_path, target_size, output_path):
    # 从文本文件中读取 NumPy 数组
    reshaped_array = np.loadtxt(input_path, dtype=np.uint8)
    # 确保数组的大小正确
    if reshaped_array.size != np.prod(target_size):
        raise ValueError(f"从文件 {input_path} 读取的数组大小不匹配目标尺寸 {target_size}")
    # 将数组重塑回原始形状
    image_array = reshaped_array.reshape(target_size)
    # 使用 Matplotlib 绘制图像，但不显示坐标轴和空白画布
    fig, ax = plt.subplots(figsize=(target_size[1] / 100, target_size[0] / 100))  # 设置图像尺寸，假设100 DPI
    ax.imshow(image_array)
    ax.axis('off')  # 去掉坐标轴
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 去掉空白画布
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def process_folder(input_folder, output_folder, target_size):
    ensure_folder_exists(output_folder)
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '_reconstructed.jpg')
            # 从txt文件中读取数组并绘制图像
            load_array_and_plot_image(input_path, target_size, output_path)

# 文件夹路径
input_folder = 'C:/Users/weiso/Desktop/DA/test_06072024/new/path_to_your_txt_output_folder'  # 替换为你的输入文件夹路径
output_folder = 'C:/Users/weiso/Desktop/DA/test_06072024/new/new_image_output_folder'  # 替换为存储处理后图像的文件夹路径

# 处理文件夹中的所有.txt文件
process_folder(input_folder, output_folder, target_size=(600, 600, 3))
