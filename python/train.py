import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from utils import UCIHARDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from model import VGG1D
from model import ResNet1D_18

from model_co import VGG1D_shen
from model_co import ResNet1D_50
'''
# 定义数据增强变换
class TimeSeriesAugmentation:
    def __init__(self, noise_std=0.005, scale_factor=0.05, shift_range=5, mask_prob=0.1, mask_length=10):
        self.noise_std = noise_std
        self.scale_factor = scale_factor
        self.shift_range = shift_range
        self.mask_prob = mask_prob
        self.mask_length = mask_length

    def __call__(self, x):
        # 添加噪声
        noise = torch.randn_like(x) * self.noise_std
        x = x + noise

        # 随机缩放
        scale = 1.0 + torch.randn(1) * self.scale_factor
        x = x * scale

        # 时间平移
        shift = torch.randint(-self.shift_range, self.shift_range, (1,))
        x = torch.roll(x, shifts=shift.item(), dims=1)

        # 随机反转
        if torch.rand(1) < 0.5:
            x = torch.flip(x, dims=[1])

        # 时间掩蔽
        if torch.rand(1) < self.mask_prob:
            start = torch.randint(0, x.shape[1] - self.mask_length, (1,))
            x[:, start:start + self.mask_length] = 0.0

        return x
    
# 数据预处理和加载
train_transform = transforms.Compose([
    transforms.Lambda(lambda x: torch.from_numpy(x)),  # 将 numpy 数组转换为 Tensor
    TimeSeriesAugmentation(noise_std=0.005, scale_factor=0.05, shift_range=5, mask_prob=0.1, mask_length=10),  # 数据增强
])

test_transform = transforms.Compose([
    transforms.Lambda(lambda x: torch.from_numpy(x)),  # 将 numpy 数组转换为 Tensor
])
'''

train_dataset = UCIHARDataset(
    data_path='data/train/X_train.txt',
    label_path='data/train/y_train.txt',
    #transform=train_transform
)

test_dataset = UCIHARDataset(
    data_path='data/test/X_test.txt',
    label_path='data/test/y_test.txt',
    #transform=test_transform
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# 初始化模型、损失函数和优化器
#model = VGG1D(num_classes=6)
#model = ResNet1D_18(num_classes=6)
#model = VGG1D_shen(num_classes=6)
model = ResNet1D_50(num_classes=6)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 使用余弦退火学习率调度器
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)


# 训练模型
num_epochs = 100
device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 记录训练和测试损失
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()

        if (i + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    epoch_train_loss /= len(train_loader)
    train_losses.append(epoch_train_loss)

    # 更新学习率
    scheduler.step()

    # 测试模型
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        epoch_test_loss = 0.0
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            epoch_test_loss += criterion(outputs, labels).item()
        epoch_test_loss /= len(test_loader)
        test_losses.append(epoch_test_loss)
        current_lr = scheduler.get_last_lr()[0]


        print(f'Test Accuracy: {100 * correct / total:.2f}%')
        print(f'Test Loss: {epoch_test_loss:.4f}')
        print(f'Current Learning Rate: {current_lr:.6f}')


    # 保存模型
    torch.save(model.state_dict(), f'models_50/model_{epoch+1}.pth')

# 绘制最终的训练和测试损失曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Test Loss')
plt.legend()
plt.savefig('models_50/final_loss_curve.png')
plt.close()