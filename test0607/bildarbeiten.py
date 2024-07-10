import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def preprocess_image(image_path, output_size=(600, 600)):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 二值化
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    
    # 形态学操作去除噪声
    kernel = np.ones((4, 4), np.uint8)  # 更小的卷积核
    processed_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=1)
    processed_image = cv2.morphologyEx(processed_image, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # 调整图像尺寸
    resized_image = cv2.resize(processed_image, output_size, interpolation=cv2.INTER_LINEAR)
    
    return resized_image

def display_image(image, title='Image'):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# 示例图像路径
image_path = 'C:/Users/weiso/Desktop/DA/test_06072024/U-Net test/inputs/0_S13un_new_594x594_processed.jpg'

# 预处理图像
preprocessed_image = preprocess_image(image_path)

# 显示处理后的图像
display_image(preprocessed_image, 'Preprocessed Image')

# 将处理后的图像保存到本地
output_image_path = 'C:/Users/weiso/Desktop/DA/test_06072024/U-Net test/preprocessed_image.jpg'
cv2.imwrite(output_image_path, preprocessed_image)
print(f'Processed image saved to {output_image_path}')
