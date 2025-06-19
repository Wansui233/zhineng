import numpy as np

class UCIHARDataset:
    def __init__(self, data_path, label_path=None, transform=None):
        self.data = np.loadtxt(data_path)
        self.labels = np.loadtxt(label_path) if label_path else None
        self.transform = transform  # 添加对 transform 的支持

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 将数据 reshape 成 1D 形式 (1, 561)
        img = self.data[idx].reshape(1, -1).astype(np.float32)
        # 手动归一化数据到 [-1, 1] 范围
        img = (img - np.mean(img)) / np.std(img)
        label = int(self.labels[idx]) - 1 if self.labels is not None else 0  # 将标签转换为从 0 开始

        # 应用变换（如果存在）
        if self.transform:
            img = self.transform(img)
            
        return img, label