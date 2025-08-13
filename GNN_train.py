import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data as GeoData
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import DispatchGNN
from data_utils import DispatchGraphDataset, data_preparation, construct_graph_data
import pickle

# 1. 数据读取与特征构造,保存编码器和标准化器
input_csv = "data_train.csv"
df_train, le_target, le_truck, scaler = data_preparation(input_csv, fit=True)
# 保存编码器和标准化器
with open('model/le_target.pkl', 'wb') as f:
    pickle.dump(le_target, f)
with open('model/le_truck.pkl', 'wb') as f:
    pickle.dump(le_truck, f)
with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# 2. 图数据构建
samples, edge_index = construct_graph_data(df_train)

# 3. 模型定义
node_in_dim = 2      # 节点特征维度（被作为目的地数量、节点类型）
truck_in_dim = 3     # 车辆特征维度（当前目标、卡车ID、任务状态）
hidden_dim = 32      # 隐藏层维度
num_classes = 43     # 节点数量（即目标类别数量）
model = DispatchGNN(node_in_dim, truck_in_dim, hidden_dim, num_classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 数据集与加载器
train_samples, test_samples = train_test_split(samples, test_size=0.2, random_state=42)
train_dataset = DispatchGraphDataset(train_samples, edge_index)
test_dataset = DispatchGraphDataset(test_samples, edge_index)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1)

# 5. 训练与测试函数
def train_epoch(model, loader):
    model.train()
    total_loss = 0
    for data in tqdm(loader, desc="Train"):
        data = data.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.y.size(0)
    return total_loss / len(loader.dataset)

def test_epoch(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(loader, desc="Test"):
            data = data.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == data.y).sum().item()
            total += data.y.size(0)
    return correct / total

# 6. 训练主循环与模型保存
train_acc_list = []
test_acc_list = []
best_acc = 0.0

max_epochs = 200

for epoch in range(max_epochs):
    loss = train_epoch(model, train_loader)
    train_acc = test_epoch(model, train_loader)
    test_acc = test_epoch(model, test_loader)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), "model/best_dispatch_gnn.pt")
        print(f"保存当前最优模型到 model/best_dispatch_gnn.pt")
torch.save(model.state_dict(), "model/last_dispatch_gnn.pt")
print(f"保存最后一次训练模型到 model/last_dispatch_gnn.pt")

# 7. 绘制训练和测试准确率随训练轮次变化的图像
plt.figure(figsize=(10,6))
plt.plot(train_acc_list, label='Train Accuracy')
plt.plot(test_acc_list, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train/Test Accuracy Curve')
plt.legend()
plt.grid(True)
plt.savefig('model/accuracy_curve.png')
