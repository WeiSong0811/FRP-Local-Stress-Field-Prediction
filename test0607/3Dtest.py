import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def ensure_folder_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def convert_image_to_array_and_save(image_path, txt_output_path, target_size=(600, 600)):
    # 打开图像
    image = Image.open(image_path)
    
    # 调整图像大小
    image = image.resize(target_size, Image.LANCZOS)
    
    # 将图像转换为 NumPy 数组
    image_array = np.array(image)
    
    # 获取图像的形状
    original_shape = image_array.shape
    
    # 将 NumPy 数组保存为文本文件
    reshaped_array = image_array.reshape(-1, image_array.shape[2])  # 将数组重塑为 (num_pixels, 3) 形式
    np.savetxt(txt_output_path, reshaped_array, fmt='%d')
    
    # 返回原始形状
    return original_shape

def enhance_contrast(image_array):
    # 线性拉伸对比度
    p2, p98 = np.percentile(image_array, (2, 98))
    image_array = np.clip(image_array, p2, p98)
    image_array = (image_array - p2) * 255.0 / (p98 - p2)
    return image_array.astype(np.uint8)

def load_array_and_plot_image(txt_input_path, original_shape, output_image_path):
    # 从文本文件中读取 NumPy 数组
    reshaped_array = np.loadtxt(txt_input_path, dtype=np.uint8)
    
    # 将数组重塑回原始形状
    image_array = reshaped_array.reshape(original_shape)
    
    # 提高图像对比度
    image_array = enhance_contrast(image_array)
    
    # 使用 Matplotlib 绘制图像，但不显示坐标轴和空白画布
    fig, ax = plt.subplots(figsize=(original_shape[1] / 100, original_shape[0] / 100))  # 设置图像尺寸，假设100 DPI
    ax.imshow(image_array)
    ax.axis('off')  # 去掉坐标轴
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 去掉空白画布
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def process_folder(input_folder, txt_output_folder, image_output_folder, target_size=(600, 600)):
    ensure_folder_exists(txt_output_folder)
    ensure_folder_exists(image_output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg'):
            image_path = os.path.join(input_folder, filename)
            txt_output_path = os.path.join(txt_output_folder, os.path.splitext(filename)[0] + '.txt')
            image_output_path = os.path.join(image_output_folder, os.path.splitext(filename)[0] + '_reconstructed.png')
            
            # 转换图像为数组并保存为txt文件
            original_shape = convert_image_to_array_and_save(image_path, txt_output_path, target_size)
            
            # 从txt文件中读取数组并绘制图像
            load_array_and_plot_image(txt_output_path, original_shape, image_output_path)

# 文件夹路径
input_folder = 'C:/Users/weiso/Desktop/DA/test_06072024/new'  # 替换为你的输入文件夹路径
txt_output_folder = 'C:/Users/weiso/Desktop/DA/test_06072024/new/path_to_your_txt_output_folder'  # 替换为存储txt文件的文件夹路径
image_output_folder = 'C:/Users/weiso/Desktop/DA/test_06072024/new/path_to_your_image_output_folder'  # 替换为存储处理后图像的文件夹路径

# 处理文件夹中的所有.jpg文件
process_folder(input_folder, txt_output_folder, image_output_folder, target_size=(600, 600))
