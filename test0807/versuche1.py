import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 假设csv文件路径为'points.csv'
csv_file_path = 'C:/Users/weiso/Desktop/DA/test0807/abaqus11.csv'  # 替换为你的CSV文件路径

# 读取csv文件
df_0 = pd.read_csv(csv_file_path)
df = df_0[df_0.iloc[:,8] == 552.073E-06 ]  # 筛选出第一层
# 提取x, y, z坐标
x = df.iloc[:, 6]
y = df.iloc[:, 7]
z = df.iloc[:, 8]

    


# 创建一个新的3D图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制3D散点图
ax.scatter(x, y, z, c='b', marker='o')

# 设置轴标签
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# 设置图形标题
ax.set_title('3D Scatter Plot of Points')

# 显示图形
plt.show()