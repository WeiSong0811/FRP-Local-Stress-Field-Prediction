import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from PIL import Image

# 打开图像并转换为灰度
img = Image.open('C:/Users/weiso/Desktop/DA/test_06072024/U-Net test/inputs/0_S13un_new_594x594_processed.jpg').convert('L')
data = np.array(img)

# 获取图像尺寸
height, width = data.shape

# 生成不规则网格点
y, x = np.mgrid[0:height, 0:width]

# 将图像数据中的非零点作为有效点
points = np.column_stack((x[data > 0], y[data > 0]))
values = data[data > 0]

# 生成规则网格
grid_x, grid_y = np.mgrid[0:width:100j, 0:height:100j]

# 使用插值方法将不规则网格数据插值到规则网格
grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')

# 可视化结果
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(data, cmap='gray')
plt.title('Original Irregular Grid')
plt.subplot(122)
plt.imshow(grid_z.T, extent=(0, width, 0, height), origin='lower', cmap='gray')
plt.title('Interpolated Regular Grid')
plt.show()