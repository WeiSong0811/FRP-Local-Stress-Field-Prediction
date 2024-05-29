import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import os

# 创建保存图像的文件夹
output_folder = "voronoi_bild"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 函数：生成并保存一个 Voronoi 图像
def generate_and_save_voronoi_image(image_index):
    # 生成一些随机点
    points = np.random.rand(20, 2) * 64  # 在 64x64 范围内生成随机点

    # 创建 Voronoi 图
    vor = Voronoi(points)

    # 绘制 Voronoi 图
    fig, ax = plt.subplots(figsize=(6, 6))

    # 为每个 Voronoi 区域上色
    patches = []
    for region in vor.regions:
        if not -1 in region and len(region) > 0:
            polygon = [vor.vertices[i] for i in region]
            patches.append(Polygon(polygon))

    # 生成随机颜色
    colors = np.random.rand(len(patches), 3)

    # 创建多边形集合
    p = PatchCollection(patches, facecolors=colors, edgecolor='k', alpha=0.6)

    # 添加到绘图中
    ax.add_collection(p)

    # 设置图形参数
    ax.set_xlim(0, 64)
    ax.set_ylim(0, 64)
    ax.set_aspect('equal', 'box')
    # ax.axis('off')  # 不显示轴
    # plt.title('64x64 Voronoi Diagram with Color Blocks')

    # 保存图像
    output_path = os.path.join(output_folder, f"voronoi_{image_index:04d}.jpg")
    plt.savefig(output_path, dpi=64, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

# 生成并保存 1000 个随机 Voronoi 图像
for i in range(1000):
    print(f"Generating image {i+1}/{1000}")
    generate_and_save_voronoi_image(i)

print(f"1000 个 Voronoi 图像已保存到 {output_folder} 文件夹中")
