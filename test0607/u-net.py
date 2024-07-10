import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

# 定义U-Net模型
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        self.encoder1 = self.conv_block(3, 64)   # 输入3通道，输出64通道
        self.encoder2 = self.conv_block(64, 128) # 输入64通道，输出128通道
        self.encoder3 = self.conv_block(128, 256) # 输入128通道，输出256通道
        self.encoder4 = self.conv_block(256, 512) # 输入256通道，输出512通道
        
        self.bottleneck = self.conv_block(512, 1024) # 输入512通道，输出1024通道
        
        self.decoder4 = self.conv_block(1024 + 512, 512) # 输入1024+512通道，输出512通道
        self.decoder3 = self.conv_block(512 + 256, 256) # 输入512+256通道，输出256通道
        self.decoder2 = self.conv_block(256 + 128, 128) # 输入256+128通道，输出128通道
        self.decoder1 = self.conv_block(128 + 64, 64) # 输入128+64通道，输出64通道
        
        self.final_layer = nn.Conv2d(64, 3, kernel_size=1) # 输入64通道，输出3通道

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.encoder1(x)  # 64通道
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))  # 128通道
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))  # 256通道
        enc4 = self.encoder4(F.max_pool2d(enc3, 2))  # 512通道
        
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))  # 1024通道
        
        dec4 = self.up_concat(bottleneck, enc4)  # 1024+512 = 1536通道
        dec4 = self.decoder4(dec4)  # 512通道
        dec3 = self.up_concat(dec4, enc3)  # 512+256 = 768通道
        dec3 = self.decoder3(dec3)  # 256通道
        dec2 = self.up_concat(dec3, enc2)  # 256+128 = 384通道
        dec2 = self.decoder2(dec2)  # 128通道
        dec1 = self.up_concat(dec2, enc1)  # 128+64 = 192通道
        dec1 = self.decoder1(dec1)  # 64通道
        
        return self.final_layer(dec1)  # 64通道到3通道

    def up_concat(self, x1, x2):
        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        return torch.cat([x1, x2], dim=1)

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform
        self.input_images = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.jpg')])
        self.target_images = sorted([os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith('.jpg')])

    def __len__(self):
        return min(len(self.input_images), len(self.target_images))

    def __getitem__(self, idx):
        input_image = Image.open(self.input_images[idx]).convert('RGB')
        target_image = Image.open(self.target_images[idx]).convert('RGB')

        input_image = input_image.resize((600, 600))
        target_image = target_image.resize((600, 600))
        
        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)
        
        return input_image, target_image

# 加载数据
input_dir = 'C:/Users/weiso/Desktop/DA/test_06072024/U-Net test/inputs'  # 输入图像文件夹
target_dir = 'C:/Users/weiso/Desktop/DA/test_06072024/U-Net test/targets'  # 目标图像文件夹

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor()
])

dataset = CustomDataset(input_dir, target_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# 在训练之前检查输入图像
inputs, targets = next(iter(dataloader))
plt.imshow(inputs[0].cpu().numpy().transpose(1, 2, 0))
plt.title('Input Image')
plt.show()

# 初始化模型、损失函数和优化器
model = UNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# 保存模型
model_path = 'C:/Users/weiso/Desktop/DA/test_06072024/U-Net test/unet_model.pth'
torch.save(model.state_dict(), model_path)
print(f'Model saved to {model_path}')

# 加载模型并进行预测
model = UNet()
model.load_state_dict(torch.load(model_path))
model.eval()

# 在训练结束后检查输出图像
with torch.no_grad():
    for inputs, _ in dataloader:
        outputs = model(inputs)
        output_image = outputs[0].cpu().numpy().transpose(1, 2, 0)
        plt.imshow(output_image)
        plt.title('Output Image')
        plt.show()

