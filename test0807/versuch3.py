import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata
import os
from scipy.interpolate import Rbf

# 定义文件路径
file_path = 'C:/Users/weiso/Desktop/DA/test0807/node_data.txt'  # 替换为你的文件路径

# 读取数据
data = np.loadtxt(file_path, delimiter=',', skiprows=1)  # 根据需要调整delimiter和skiprows
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]
# 假设应力数据从第4列开始
stresses = data[:, 3:9]

# 应力分量标签
stress_labels = ['S11', 'S22', 'S33', 'S12', 'S13', 'S23']

# 文件夹路径
output_folder = 'C:/Users/weiso/Desktop/DA/test0807/utput_folder'  # 替换为输出文件夹路径
os.makedirs(output_folder, exist_ok=True)

# 绘制应力分量的3D散点图
for i, label in enumerate(stress_labels):
    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=3,
            color=stresses[:, i],
            colorscale='Viridis',
            colorbar=dict(title=label)
        )
    )])

    fig.update_layout(title=f'Stress Field {label}',
                      scene=dict(
                          xaxis_title='X Coordinate',
                          yaxis_title='Y Coordinate',
                          zaxis_title='Z Coordinate'
                      ))

    # 保存为HTML文件
    output_file = os.path.join(output_folder, f'{label}_3d_scatter.html')
    fig.write_html(output_file)
    print(f"{label} 三维图像已保存: {output_file}")

print("所有三维图像已生成并保存。")

# 定义插值网格的分辨率
grid_resolution = 100  # 你可以根据需要调整分辨率

# 获取所有唯一的z值
unique_z_values = np.unique(z)

# 计算全局最小值和最大值
global_min = np.min(stresses)
global_max = np.max(stresses)

# 绘制应力分量的插值3D图
for i, label in enumerate(stress_labels):
    fig = go.Figure()

    # 对每个唯一的z值进行插值
    for z_value in unique_z_values:
        mask = z == z_value
        xi, yi = np.linspace(x.min(), x.max(), grid_resolution), np.linspace(y.min(), y.max(), grid_resolution)
        xi, yi = np.meshgrid(xi, yi)
        
        # RBF插值
        rbf = Rbf(x[mask], y[mask], stresses[mask, i], function='linear')
        zi = rbf(xi, yi)

        # 将插值结果添加到图中
        fig.add_trace(go.Surface(
            x=xi, y=yi, z=np.full_like(xi, z_value), surfacecolor=zi,
            cmin=global_min, cmax=global_max, colorscale='Viridis', showscale=False
        ))

    # 添加颜色条
    fig.update_layout(coloraxis_colorbar=dict(
        title=label
    ))

    fig.update_layout(title=f'Stress Field {label} (RBF Interpolated)',
                      scene=dict(
                          xaxis_title='X Coordinate',
                          yaxis_title='Y Coordinate',
                          zaxis_title='Z Coordinate'
                      ))

    # 保存为HTML文件
    output_file = os.path.join(output_folder, f'{label}_3d_interpolated_rbf.html')
    fig.write_html(output_file)
    print(f"{label} RBF插值三维图像已保存: {output_file}")

print("所有RBF插值三维图像已生成并保存。")