from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
import torch.nn as nn

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder

import os
import csv
import numpy as np
# 指定源文件夹路径
source_folder_path = 'C:/Users/张和铭/pythonProject55/combined-300'
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

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 将所有为0的元素替换为NaN，这样在计算均值和标准差时可以忽略这些元素
# X_train_tensor1[X_train_tensor1 == 0] = float('nan')
# X_test_tensor1[X_test_tensor1 == 0] = float('nan')


# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)


# 定义ConvLSTM模型
class ConvLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(ConvLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.conv1 = nn.Conv2d(in_channels=input_size, out_channels=32, kernel_size=(2, 2), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2,2), padding=(1, 1))
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1))
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.permute(0, 2, 1, 3)  # 调整维度以适应LSTM的输入要求
        batch_size, seq_len, channels, height, width = x.size()
        x = x.view(batch_size, seq_len, channels * height * width)
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1, :, :])
        return out


# 初始化模型和损失函数
input_size = 35  # 输入数据的特征数
hidden_size = 64  # LSTM隐藏层大小
num_layers = 2  # LSTM层数
num_classes = 4  # 四分类任务

model = ConvLSTM(input_size, hidden_size, num_layers, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 训练函数
def train(model, criterion, optimizer, train_loader, val_loader, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data.float())  # 前向传播
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * data.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}")

        # 验证函数
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in val_loader:
                outputs = model(data.float())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = correct / total
        print(f"Epoch [{epoch + 1}/{num_epochs}], Val Accuracy: {val_accuracy:.2%}")

    print("Finished Training")


# 测试函数
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data.float())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = correct / total
    print(f"Test Accuracy: {test_accuracy:.2%}")


# 开始训练和测试
train(model, criterion, optimizer, train_loader, test_loader, num_epochs=10)
test(model, test_loader)