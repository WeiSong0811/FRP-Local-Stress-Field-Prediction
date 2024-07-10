import os
import numpy as np
import matplotlib.pyplot as plt

def plot_data_from_file(txt_file, output_dir):
    # 读取数据
    data = np.loadtxt(txt_file, delimiter=',', skiprows=1)
    nodeCoords = data[:, :3]
    stresses = data[:, 3:9]
    strains = data[:, 9:15]

    # 检查数据范围，确保数据有效性
    print(f"节点坐标范围: X({nodeCoords[:,0].min()}, {nodeCoords[:,0].max()}), Y({nodeCoords[:,1].min()}, {nodeCoords[:,1].max()}), Z({nodeCoords[:,2].min()}, {nodeCoords[:,2].max()})")
    print(f"应力范围: S11({stresses[:,0].min()}, {stresses[:,0].max()}), S22({stresses[:,1].min()}, {stresses[:,1].max()}), S33({stresses[:,2].min()}, {stresses[:,2].max()}), S12({stresses[:,3].min()}, {stresses[:,3].max()}), S13({stresses[:,4].min()}, {stresses[:,4].max()}), S23({stresses[:,5].min()}, {stresses[:,5].max()})")
    print(f"应变范围: E11({strains[:,0].min()}, {strains[:,0].max()}), E22({strains[:,1].min()}, {strains[:,1].max()}), E33({strains[:,2].min()}, {strains[:,2].max()}), E12({strains[:,3].min()}, {strains[:,3].max()}), E13({strains[:,4].min()}, {strains[:,4].max()}), E23({strains[:,5].min()}, {strains[:,5].max()})")

    # 应力分量标签
    stress_labels = ['S11', 'S22', 'S33', 'S12', 'S13', 'S23']
    
    for i, label in enumerate(stress_labels):
        plt.figure()
        plt.scatter(nodeCoords[:,0], nodeCoords[:,1], c=stresses[:,i], cmap='viridis')
        plt.colorbar(label=label)
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title(f'Stress Field {label}')
        plt.savefig(f"{output_dir}/{label}.png")
        plt.close()
        print(f"{label} 图像已保存")

    # 绘制应变分布图
    plt.figure()
    plt.scatter(nodeCoords[:,0], nodeCoords[:,1], c=strains[:,0], cmap='viridis')  # 假设我们使用 E11 作为应变图
    plt.colorbar(label='E11')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Strain Distribution E11')
    plt.savefig(f"{output_dir}/Strain_Distribution_E11.png")
    plt.close()
    print(f"应变分布图已保存")

# 主程序
if __name__ == "__main__":
    txt_file = 'C:/Users/weiso/Desktop/DA/test0807/node_data.txt'  # 替换为你保存的txt文件路径
    output_dir = 'C:/Users/weiso/Desktop/DA/test0807/data_images'  # 替换为你希望保存图像的文件夹路径

    # 创建保存图像的文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plot_data_from_file(txt_file, output_dir)
