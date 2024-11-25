import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score


class SensorDataset(Dataset):
    def __init__(self, data_dir, scaler=None):
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform([f[30] for f in self.file_list])
        self.scaler = scaler

        # 读取所有数据以便在训练集上进行标准化
        self.data = []
        for file in self.file_list:
            file_path = os.path.join(self.data_dir, file)
            data = pd.read_csv(file_path, header=None)
            data = data.apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)
            self.data.append(data)

        self.data = np.stack(self.data)

        # 如果没有提供scaler，则在训练集上进行标准化
        if self.scaler is None:
            self.scaler = StandardScaler()
            num_samples, seq_len, num_features = self.data.shape
            self.data = self.data.reshape(-1, num_features)
            self.scaler.fit(self.data)
            self.data = self.scaler.transform(self.data)
            self.data = self.data.reshape(num_samples, seq_len, num_features)
        else:
            num_samples, seq_len, num_features = self.data.shape
            self.data = self.data.reshape(-1, num_features)
            self.data = self.scaler.transform(self.data)
            self.data = self.data.reshape(num_samples, seq_len, num_features)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

data_dir = 'C:/Users/张和铭/pythonProject55/combined-400'

full_dataset = SensorDataset(data_dir)

# 划分训练集、验证集和测试集
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

# 使用训练集的scaler对验证集和测试集进行标准化
scaler = full_dataset.scaler
val_dataset = SensorDataset(data_dir, scaler=scaler)
test_dataset = SensorDataset(data_dir, scaler=scaler)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=128, nhead=8, num_encoder_layers=3, dim_feedforward=512):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # (batch_size, seq_len, d_model)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, d_model)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # (batch_size, d_model)
        x = self.fc(x)  # (batch_size, num_classes)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = TransformerModel(input_dim=37, num_classes=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 45

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)

    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0.0   
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * data.size(0)
            val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader.dataset)
    val_accuracy = accuracy_score(val_labels, val_preds)

    print(
        f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

# 测试集评估
model.eval()
test_preds = []
test_labels = []
with torch.no_grad():
    for data, labels in test_loader:
        data, labels = data.to(device), labels.to(device)
        outputs = model(data)
        test_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

test_accuracy = accuracy_score(test_labels, test_preds)
print(f'Test Accuracy: {test_accuracy:.4f}')

