import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def convert_images_to_arrays(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg'):
            image_path = os.path.join(input_folder, filename)
            image = Image.open(image_path).convert('L')
            image_array = np.array(image)
            
            txt_filename = os.path.splitext(filename)[0] + '.txt'
            txt_path = os.path.join(output_folder, txt_filename)
            np.savetxt(txt_path, image_array, fmt='%d')

def process_txt_files(input_folder, output_folder, image_output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(image_output_folder):
        os.makedirs(image_output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            txt_path = os.path.join(input_folder, filename)
            image_array = np.loadtxt(txt_path, dtype=np.uint8)
            mode_result = stats.mode(image_array, axis=None)
            if mode_result.mode < 25:  # 判断图像灰度，较暗图像按照22处理
                processed_array = np.where(image_array > 22, 255, 0)  #大于22的值设置为255，其余的为0
            else:
                processed_array = np.where(image_array > 100, 255, 0)  #大于100的值设置为255，其余的为0
            
            processed_txt_filename = os.path.splitext(filename)[0] + '_processed.txt'
            processed_txt_path = os.path.join(output_folder, processed_txt_filename)
            np.savetxt(processed_txt_path, processed_array, fmt='%d')

            # Save processed image
            image_output_path = os.path.join(image_output_folder, os.path.splitext(filename)[0] + '_processed.jpg')
            plt.imsave(image_output_path, processed_array, cmap='gray')

# 文件夹路径
input_folder = 'C:/Users/weiso/Desktop/DA/test_06072024/new/2D'  # 替换为你的输入文件夹路径
txt_output_folder = 'C:/Users/weiso/Desktop/DA/test_06072024/new/2D/path_to_your_txt_output_folder'  # 替换为存储txt文件的文件夹路径
processed_output_folder = 'C:/Users/weiso/Desktop/DA/test_06072024/new/2D/path_to_your_processed_output_folder'  # 替换为存储处理后txt文件的文件夹路径
image_output_folder = 'C:/Users/weiso/Desktop/DA/test_06072024/new/2D/path_to_your_image_output_folder'  # 替换为存储处理后图像的文件夹路径

# 转换图像为数组并保存为txt
convert_images_to_arrays(input_folder, txt_output_folder)

# 处理txt文件并保存新的txt文件，同时保存处理后的图像
process_txt_files(txt_output_folder, processed_output_folder, image_output_folder)

