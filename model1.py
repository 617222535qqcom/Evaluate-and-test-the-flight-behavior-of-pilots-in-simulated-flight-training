import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.nn import Transformer
from sklearn.metrics import mean_squared_error, mean_absolute_error
# 将数据转换为PyTorch Tensor
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder

import os
import csv
import numpy as np

# 指定源文件夹路径
source_folder_path = 'C:/Users/张和铭/pythonProject55/combined-300'
# 指定目标文件路
destination_file_path = 'C:/Users/张和铭/pythonProject55/output.csv'

# 初始化一个列表用于存储合并后的数据
datas = []
labels = []
# 遍历源文件夹中的每个文件
for filename in os.listdir(source_folder_path):
    if filename.endswith('.csv'):
        source_file_path = os.path.join(source_folder_path, filename)

        with open(source_file_path, mode='r', encoding='utf-8') as source_file:
            reader = csv.reader(source_file)
            next(reader)  # 跳过第一行
            label = os.path.basename(source_file_path)[30]
            d = np.genfromtxt(source_file_path, delimiter=',', skip_header=1, usecols=range(2, 37))
            datas.append(d)
            labels.append(label)


label_encoder = LabelEncoder()
labelss = label_encoder.fit_transform(labels)
X_train, X_test, y_train, y_test = train_test_split(datas, labelss, test_size=0.2, random_state=42)
print(len(X_train), len(X_test), len(datas))

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)


class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out


# 设置模型参数
input_size = 35  # 输入特征维度，假设每个时间步输入大小为28
hidden_size = 64  # RNN隐藏层大小
num_layers = 9  # RNN层数
num_classes = 4  # 分类数

# 初始化模型
model = RNNClassifier(input_size, hidden_size, num_layers, num_classes)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01,eps=1e-4)

# 将模型移到GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 训练模型
num_epochs = 15
# 训练模型
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device, dtype=torch.long)  # 将标签转换为torch.long类型

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}')

print('训练完成')

# 测试模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device, dtype=torch.long)  # 将标签转换为torch.long类型
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'测试集准确率: {100 * correct / total}%')
