import os
from PIL import Image
import matplotlib.pyplot as plt

def convert_to_grayscale(input_folder, output_folder,target_size=(600, 600)):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg'):
            input_image_path = os.path.join(input_folder, filename)
            output_image_path = os.path.join(output_folder, filename)
            
            # 打开输入图像
            color_image = Image.open(input_image_path)
            # 调整图像大小
            color_image = color_image.resize(target_size, Image.LANCZOS)
            # 将图像转换为灰度图像
            grayscale_image = color_image.convert('L')
            
            # 保存灰度图像
            grayscale_image.save(output_image_path)
                        
            print(f"Processed {filename}")

# 示例使用
input_folder = 'C:/Users/weiso/Desktop/DA/test_06072024/new'
output_folder = 'C:/Users/weiso/Desktop/DA/test_06072024/new/2D/'
convert_to_grayscale(input_folder, output_folder,target_size=(600, 600))
