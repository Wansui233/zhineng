import numpy as np

from utils import UCIHARDataset

import matplotlib.pyplot as plt
import random

# 假设每个志愿者有固定数量的样本，这里需要根据实际数据集调整
# UCI HAR数据集共有30个志愿者，每个志愿者的数据分布在训练集和测试集中
# 这里简单假设数据集中每个志愿者的样本是连续排列的
# 你可能需要根据实际数据集的结构进行调整
NUM_VOLUNTEERS = 30

# 加载数据集
train_dataset = UCIHARDataset(
    data_path='data/train/X_train.txt',
    label_path='data/train/y_train.txt'
)
test_dataset = UCIHARDataset(
    data_path='data/test/X_test.txt',
    label_path='data/test/y_test.txt'
)

# 合并训练集和测试集
all_dataset = train_dataset.data.tolist() + test_dataset.data.tolist()
all_labels = train_dataset.labels.tolist() + test_dataset.labels.tolist()

# 随机选择一名志愿者
volunteer_id = random.randint(1, NUM_VOLUNTEERS)
print(f"随机选择的志愿者编号: {volunteer_id}")

# 假设每个志愿者的样本数量大致相同，这里简单计算每个志愿者的样本范围
samples_per_volunteer = len(all_dataset) // NUM_VOLUNTEERS
start_index = (volunteer_id - 1) * samples_per_volunteer
end_index = start_index + samples_per_volunteer

# 获取该志愿者的所有数据和标签
volunteer_data = all_dataset[start_index:end_index]
volunteer_labels = all_labels[start_index:end_index]

# 每种行为的编号从1到6
for activity_id in range(1, 7):
    # 找到该行为的所有样本
    activity_indices = [i for i, label in enumerate(volunteer_labels) if label == activity_id]
    if activity_indices:
        # 随机选择一个样本
        sample_index = random.choice(activity_indices)
        sample_data = volunteer_data[sample_index]

        # 时域可视化
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(sample_data)
        plt.title(f'Volunteer {volunteer_id}, Activity {activity_id} - Time Domain')
        plt.xlabel('Time Step')
        plt.ylabel('Amplitude')

        # 频域可视化
        fft_data = np.fft.fft(sample_data)
        frequencies = np.fft.fftfreq(len(sample_data))
        plt.subplot(2, 1, 2)
        plt.plot(frequencies, np.abs(fft_data))
        plt.title(f'Volunteer {volunteer_id}, Activity {activity_id} - Frequency Domain')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')

        plt.tight_layout()
        # 保存图像
        plt.savefig(f'volunteer_{volunteer_id}_activity_{activity_id}.png')
        plt.close()