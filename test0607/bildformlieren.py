import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

# 设置图像文件夹路径
folder_path = 'C:/Users/weiso/Desktop/DA/test_06072024'  # 请根据实际路径修改
output_folder_path = 'C:/Users/weiso/Desktop/DA/test_06072024/new/'  # 保存裁剪后图像的文件夹

# 创建输出文件夹（如果不存在）
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# 获取文件夹中的所有 .jpg 文件
image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

# 处理每个图像文件
for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        continue

    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用边缘检测
    edges = cv2.Canny(gray, 50, 150)

    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # 获取所有轮廓的边界框
        x, y, w, h = cv2.boundingRect(np.concatenate(contours))

        # 裁剪图像
        cropped_image = image[y:y+h, x:x+w]

        # 生成新的文件名，包含输出图像的尺寸信息
        base_name = os.path.splitext(image_file)[0]
        new_image_name = f"{base_name}_new_{w}x{h}.jpg"
        new_image_path = os.path.join(output_folder_path, new_image_name)

        # 保存裁剪后的图像
        cv2.imwrite(new_image_path, cropped_image)

        # 打印保存信息
        print(f"Cropped image saved at: {new_image_path}")
    else:
        print(f"No contours found in the image: {image_path}")

