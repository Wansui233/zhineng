import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from model import VGG1D
from model import ResNet1D_18
from utils import UCIHARDataset
from torch.utils.data import DataLoader

from model_co import VGG1D_shen
from model_co import ResNet1D_50


test_dataset = UCIHARDataset(
    data_path='data/test/X_test.txt',
    label_path='data/test/y_test.txt'
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 加载模型
model = ResNet1D_50(num_classes=6)
model.load_state_dict(torch.load('models_50/model_100.pth'))

# 测试模型
model.eval()
with torch.no_grad():
    y_true = []
    y_pred = []
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(predicted.numpy())
    
    # 计算准确率
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    
    # 打印分类报告
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    
    # 绘制混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.set_style("darkgrid")  # 设置背景为灰色网格
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu")
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.savefig('models_50/final_Matrix.png')
    plt.show()