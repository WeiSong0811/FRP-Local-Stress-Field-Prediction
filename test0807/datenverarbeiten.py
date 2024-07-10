import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from scipy.interpolate import griddata
from matplotlib.colors import ListedColormap, BoundaryNorm
# 读取txt文件，并将数据存储到DataFrame中
txt_file = 'C:/Users/weiso/Desktop/DA/test0807/node_data.txt'  # 替换为你的文件路径

data = np.loadtxt(txt_file, delimiter=',', skiprows=1)
nodeCoords = data[:, :3]
stresses = data[:, 3:9]
strains = data[:, 9:15]

# 检查数据范围，确保数据有效性
print(f"节点坐标范围: X({nodeCoords[:,0].min()}, {nodeCoords[:,0].max()}), Y({nodeCoords[:,1].min()}, {nodeCoords[:,1].max()}), Z({nodeCoords[:,2].min()}, {nodeCoords[:,2].max()})")
print(f"应力范围: S11({stresses[:,0].min()}, {stresses[:,0].max()}), S22({stresses[:,1].min()}, {stresses[:,1].max()}), S33({stresses[:,2].min()}, {stresses[:,2].max()}), S12({stresses[:,3].min()}, {stresses[:,3].max()}), S13({stresses[:,4].min()}, {stresses[:,4].max()}), S23({stresses[:,5].min()}, {stresses[:,5].max()})")
print(f"应变范围: E11({strains[:,0].min()}, {strains[:,0].max()}), E22({strains[:,1].min()}, {strains[:,1].max()}),  E33({strains[:,2].min()}, {strains[:,2].max()}), E12({strains[:,3].min()}, {strains[:,3].max()}), E13({strains[:,4].min()}, {strains[:,4].max()}),E23({strains[:,5].min()}, {strains[:,5].max()})")

# 检测数据的层数，结果显示有11层，也就是有11个不同的z值
# plt.figure()
# plt.scatter(nodeCoords[:,0],nodeCoords[:,2])
# plt.show()

## 筛选出不同的z值对应的数据 ##
# 四舍五入z值到小数点后4位
rounded_z_values = np.round(nodeCoords[:, 2], 4)

# 获取所有唯一的四舍五入后的z值
unique_rounded_z_values = np.unique(rounded_z_values)

# 创建一个目录来保存分类后的文件
output_dir = 'C:/Users/weiso/Desktop/DA/test0807/z_value_files'
os.makedirs(output_dir, exist_ok=True)

# 根据每个四舍五入后的z值进行分类并保存为单独的txt文件
for z_value in unique_rounded_z_values:
    # 筛选出当前四舍五入后z值对应的数据
    mask = rounded_z_values == z_value
    z_data = np.hstack((nodeCoords[mask], stresses[mask], strains[mask]))
    # 保存到txt文件
    output_file = os.path.join(output_dir, f'z_{z_value}.txt')
    np.savetxt(output_file, z_data, delimiter=',', header='x,y,z,s11,s22,s33,s12,s13,s23,e11,e22,e33,e12,e13,e23', comments='')
    
print("文件已根据四舍五入后的z值分类并保存到各自的txt文件中。")

# 文件夹路径
folder_path = 'C:/Users/weiso/Desktop/DA/test0807/z_value_files'

# 获取文件夹下所有的txt文件
txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

# 应力分量标签
stress_labels = ['S11', 'S22', 'S33', 'S12', 'S13', 'S23']


# 定义插值网格的分辨率
grid_resolution = 100  # 提高分辨率

# 初始化最小值和最大值
global_min = float('inf')
global_max = float('-inf')

# 首先遍历所有文件，计算全局最小值和最大值
for txt_file in txt_files:
    # 构建文件路径
    file_path = os.path.join(folder_path, txt_file)
    
    # 读取数据
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    nodeCoords = data[:, :3]
    stresses = data[:, 3:9]
    
    # 更新全局最小值和最大值
    global_min = min(global_min, stresses.min())
    global_max = max(global_max, stresses.max())

# 遍历每个txt文件进行绘图
for txt_file in txt_files:
    # 构建文件路径
    file_path = os.path.join(folder_path, txt_file)
    
    # 读取数据
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    nodeCoords = data[:, :3]
    stresses = data[:, 3:9]
    
    # 获取z值
    z_value = nodeCoords[0, 2]
    
    # 创建插值网格
    grid_x, grid_y = np.meshgrid(
        np.linspace(nodeCoords[:,0].min(), nodeCoords[:,0].max(), grid_resolution),
        np.linspace(nodeCoords[:,1].min(), nodeCoords[:,1].max(), grid_resolution)
    )
    
    # 绘制应力分量的图表
    for i, label in enumerate(stress_labels):
        # 进行插值
        rbf = Rbf(nodeCoords[:,0], nodeCoords[:,1], stresses[:,i], function='cubic')
        grid_z = rbf(grid_x, grid_y)
        
        # 绘制插值图
        plt.figure()
        plt.imshow(grid_z, extent=(nodeCoords[:,0].min(), nodeCoords[:,0].max(), nodeCoords[:,1].min(), nodeCoords[:,1].max()), 
                   origin='lower', cmap='viridis', aspect='auto', vmin=global_min, vmax=global_max)
        plt.colorbar(label=label)
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title(f'Stress Field {label}(z={z_value})')
        
        # 保存图表
        output_file = os.path.join(folder_path, f'{label}(z={z_value}).png')
        plt.savefig(output_file)
        plt.close()
        print(f"{label} 图像已保存: {output_file}")

print("所有图像已生成并保存。")



# 文件夹路径
folder_path = 'C:/Users/weiso/Desktop/DA/test0807/z_value_files'

# 获取文件夹下所有的txt文件
txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

# 应力分量标签
stress_labels = ['S11', 'S22', 'S33', 'S12', 'S13', 'S23']

# 定义插值网格的分辨率
grid_resolution = 100  # 提高分辨率

# 自定义颜色映射
colors = [
    '#000033', '#000066','#00007F', '#0000FF', '#007FFF','#00FFFF' , '#00FF7F','#00FF00', '#7FFF00','#FFFF00', '#FF7F00','#FF0000'  
]
cmap = ListedColormap(colors)

# 遍历每个txt文件进行绘图
for txt_file in txt_files:
    # 构建文件路径
    file_path = os.path.join(folder_path, txt_file)
    
    # 读取数据
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    nodeCoords = data[:, :3]
    stresses = data[:, 3:9]
    
    # 获取z值
    z_value = nodeCoords[0, 2]
    
    # 创建插值网格
    grid_x, grid_y = np.meshgrid(
        np.linspace(nodeCoords[:,0].min(), nodeCoords[:,0].max(), grid_resolution),
        np.linspace(nodeCoords[:,1].min(), nodeCoords[:,1].max(), grid_resolution)
    )
    
    # 绘制应力分量的图表
    for i, label in enumerate(stress_labels):
        # 进行插值
        rbf = Rbf(nodeCoords[:,0], nodeCoords[:,1], stresses[:,i], function='cubic')
        grid_z = rbf(grid_x, grid_y)
        
        # 获取当前应力分量的最小值和最大值
        local_min = grid_z.min()
        local_max = grid_z.max()
        
        # 创建对应的颜色映射
        norm = BoundaryNorm(np.linspace(local_min, local_max, len(colors) + 1), cmap.N)
        
        # 绘制插值图
        plt.figure()
        plt.imshow(grid_z, extent=(nodeCoords[:,0].min(), nodeCoords[:,0].max(), nodeCoords[:,1].min(), nodeCoords[:,1].max()), 
                   origin='lower', cmap=cmap, aspect='auto', norm=norm)
        plt.colorbar(label=label)
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title(f'Stress Field {label}(z={z_value})')
        
        # 保存图表
        output_file = os.path.join(folder_path, f'{label}(z={z_value})new.png')
        plt.savefig(output_file)
        plt.close()
        print(f"{label} 图像已保存: {output_file}")

print("所有图像已生成并保存。")