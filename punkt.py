import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import random
import os
import gc
import logging


# 创建保存图像的文件夹
output_folder = "circle_images"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

mask_folder = "circle_masks"
if not os.path.exists(mask_folder):
    os.makedirs(mask_folder)

def generate_random_circles_image(size, num_circles, diameter, image_filename, mask_filename):
    try:
        print(f"Starting generation of {image_filename}")
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='blue')  # 设置整个图形的背景颜色为蓝色
        
        # 绘制蓝色背景
        ax.set_facecolor('blue')
        
        radius = diameter / 2
        circles = []
        max_attempts = 1000  # 增加最大尝试次数以确保能够放置所有小球
        attempt = 0
        
        while len(circles) < num_circles and attempt < max_attempts:
            x = random.uniform(radius, size - radius)
            y = random.uniform(radius, size - radius)
            
            # 检查新圆是否与现有圆重叠
            if all(np.sqrt((x - cx)**2 + (y - cy)**2) >= 2 * radius for cx, cy, _ in circles):
                circles.append((x, y, radius))
            attempt += 1
        
        if len(circles) < num_circles:
            logging.warning(f"Could only place {len(circles)} out of {num_circles} circles.")
        
        for x, y, radius in circles:
            circle = Circle((x, y), radius, color='yellow')
            ax.add_patch(circle)
        
        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
        ax.set_aspect('equal', 'box')
        ax.axis('off')
        
        plt.gca().invert_yaxis()
        plt.savefig(image_filename, bbox_inches='tight', pad_inches=0, facecolor=fig.get_facecolor())
        plt.close(fig)  # 显式关闭图形对象以释放内存
        
         # 创建对应的掩膜图像
        mask = np.zeros((size, size), dtype=np.uint8)
        for x, y, radius in circles:
            rr, cc = np.ogrid[:size, :size]
            mask_area = (rr - y)**2 + (cc - x)**2 <= radius**2
            mask[mask_area] = 255  # 将小球区域设置为白色（255）
        
        plt.imsave(mask_filename, mask, cmap='gray')

    except Exception as e:
        print(f"Error generating image {image_filename}: {e}")
    finally:
        plt.close(fig)  # 确保图形对象被关闭
        plt.cla()  # 清除当前轴
        plt.clf()  # 清除当前图形
        plt.close('all')  # 关闭所有图形
        gc.collect()  # 强制进行垃圾回收

# 生成并保存图像
num_images = 1000  # 可以更改为需要生成的图像数量
num_circles = 5  # 可以自定义小球数量
diameter = 20    # 可以自定义小球直径

for i in range(num_images):
    try:
        image_filename = os.path.join(output_folder, f"circle_{i:04d}.jpg")
        mask_filename = os.path.join(mask_folder, f"mask_{i:04d}.png")
        print(f"Generating image {i+1}/{num_images}")
        logging.info(f"Generating image {i+1}/{num_images}")
        generate_random_circles_image(size=64, num_circles=num_circles, diameter=diameter, image_filename=image_filename, mask_filename=mask_filename)
    except Exception as e:
        print(f"Error in loop {i+1}: {e}")

print(f"{num_images} 个图像和掩膜已保存到 {output_folder} 和 {mask_folder} 文件夹中")
