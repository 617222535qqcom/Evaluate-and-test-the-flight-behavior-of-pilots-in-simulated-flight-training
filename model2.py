import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
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
source_folder_path = 'C:/Users/张和铭/pythonProject55/combined025'
# 指定目标文件路径
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


# # 读取CSV数据
# df = pd.read_csv('devSubjsFeatMat_2.csv')
#
# # 提取特征和标签
# X = df.iloc[:, 3:96].values  # 提取第4列到第96列作为特征数据
# y = df.iloc[:, 2].values     # 提取第3列作为标签数据
# ylabel = []
# for i in range(y.shape[0]):
#     a = int(y[i].split("-")[1][1])
#     ylabel.append(a)
# # 划分训练集和测试集

label_encoder = LabelEncoder()
labelss = label_encoder.fit_transform(labels)
X_train, X_test, y_train, y_test = train_test_split(datas, labelss, test_size=0.2, random_state=42)
print(len(X_train), len(X_test), len(datas))

X_train_tensor1 = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 将所有为0的元素替换为NaN，这样在计算均值和标准差时可以忽略这些元素
# X_train_tensor1[X_train_tensor1 == 0] = float('nan')
# X_test_tensor1[X_test_tensor1 == 0] = float('nan')

mean = X_train_tensor1.mean()
std = X_train_tensor1.std()

# 2. 应用标准化公式
X_train_tensor2 = (X_train_tensor1 - mean) / std
# X_train_tensor2[X_train_tensor2 == 0] = float('nan')
X_train_tensor = X_train_tensor2

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)


# 定义LSTM模型类
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # 取序列最后一个时间步的输出进行分类
        return out



# 初始化模型、损失函数和优化器
input_size = 35  # CSV文件每行数据的列数
hidden_size = 256
num_layers = 1
num_classes = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)


# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device, dtype=torch.long)  # 将标签转换为torch.long类型
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# 在测试集上评估模型
model.eval()
total_correct = 0
total_samples = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

accuracy = total_correct / total_samples
print(f'Accuracy on test set: {accuracy:.2%}')