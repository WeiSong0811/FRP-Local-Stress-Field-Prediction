import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# 加载训练好的U-Net模型
model_path = 'unet_circle_detection.h5'
model = load_model(model_path)

# 预处理图像
def preprocess_image(image_path, target_size=(64, 64)):
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)  # 增加一个维度，用于模型输入
    img = img / 255.0  # 归一化到[0, 1]
    return img

# 后处理预测结果
def postprocess_prediction(pred):
    pred = pred[0, :, :, 0]
    pred = (pred > 0.5).astype(np.uint8)  # 阈值处理，二值化
    return pred

# 目录设置
image_dir = 'circle_images_64'  # 替换为待预测图像的目录
output_dir = 'unet_predictions'  # 替换为保存预测结果的目录

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 遍历目录中的所有图像
for image_name in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_name)
    if image_path.endswith('.png') or image_path.endswith('.jpg'):
        # 预处理图像
        img = preprocess_image(image_path)
        
        # 使用模型进行预测
        prediction = model.predict(img)
        
        # 后处理预测结果
        prediction = postprocess_prediction(prediction)
        
        # 保存预测结果
        plt.imsave(os.path.join(output_dir, f'pred_{image_name}'), prediction, cmap='gray')

print("预测完成。结果保存在:", output_dir)
