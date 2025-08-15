import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data as GeoData
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import DispatchGNN, WeightedRewardLoss
from data_utils import DispatchGraphDataset, data_preparation, construct_graph_data
import pickle
import os


# 1. 数据读取与特征构造,保存编码器和标准化器
input_csv = "data_train.csv"
# df = data_preparation(input_csv)
df_train, le_target, le_truck, scaler = data_preparation(input_csv, fit=True)
# 保存编码器和标准化器
os.makedirs('model_6', exist_ok=True)
with open('model_6/le_target.pkl', 'wb') as f:
    pickle.dump(le_target, f)
with open('model_6/le_truck.pkl', 'wb') as f:
    pickle.dump(le_truck, f)
with open('model_6/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# 2. 图数据构建
samples, edge_index = construct_graph_data(df_train)

# 3. 模型定义
node_in_dim = 2      # 节点特征维度（被作为目的地数量、节点类型）
truck_in_dim = 4     # 车辆特征维度（卡车ID、任务状态、上一个目标、停留时间）
hidden_dim = 32      # 隐藏层维度
num_classes = 43     # 节点数量（即目标类别数量）
model = DispatchGNN(node_in_dim, truck_in_dim, hidden_dim, num_classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 使用新的自定义损失函数，并传入新的 move_weight 和调整后的惩罚权重
criterion = WeightedRewardLoss(
    move_weight=2.0,      # 移动样本基础权重
    mbs_penalty=5.0,      # Move-But-Stay 惩罚 (最高)
    mbr_penalty=3.0,      # Move-But-Wrong 惩罚 (较高)
    sbm_penalty=1.5,      # Stay-But-Move 惩罚 (普通)
    device=device
)

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
        loss = criterion(outputs, data)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.y.size(0)
    return total_loss / len(loader.dataset)

def test_epoch(model, loader, criterion):
    model.eval()
    
    total_loss = 0
    
    # Overall metrics
    total_samples = 0
    total_correct = 0
    
    # Stay metrics
    total_stay = 0
    correct_stay = 0
    sbm_errors = 0 # Stay but predicted Move
    
    # Move metrics
    total_move = 0
    correct_move = 0
    mbs_errors = 0 # Move but predicted Stay
    mbr_errors = 0 # Move but predicted wRong
    
    with torch.no_grad():
        for data in tqdm(loader, desc="Evaluate"):
            data = data.to(device)
            outputs = model(data)
            loss = criterion(outputs, data)
            total_loss += loss.item() * data.y.size(0)
            
            _, predicted = torch.max(outputs, 1)
            
            total_samples += data.y.size(0)
            total_correct += (predicted == data.y).sum().item()
            
            # Identify stay/move ground truth
            is_stay = (data.y == data.cur_targets)
            is_move = ~is_stay
            
            # Identify stay/move predictions
            pred_is_stay = (predicted == data.cur_targets)
            pred_is_move = ~pred_is_stay

            # Accumulate stay metrics
            total_stay += is_stay.sum().item()
            correct_stay += (predicted[is_stay] == data.y[is_stay]).sum().item()
            sbm_errors += pred_is_move[is_stay].sum().item()

            # Accumulate move metrics
            total_move += is_move.sum().item()
            correct_move += (predicted[is_move] == data.y[is_move]).sum().item()
            mbs_errors += pred_is_stay[is_move].sum().item()
            mbr_errors += (pred_is_move[is_move] & (predicted[is_move] != data.y[is_move])).sum().item()

    # Calculate final metrics
    metrics = {
        'loss': total_loss / total_samples if total_samples > 0 else 0,
        'total_acc': total_correct / total_samples if total_samples > 0 else 0,
        'stay_acc': correct_stay / total_stay if total_stay > 0 else 0,
        'move_acc': correct_move / total_move if total_move > 0 else 0,
        'sbm_rate': sbm_errors / total_stay if total_stay > 0 else 0,
        'mbs_rate': mbs_errors / total_move if total_move > 0 else 0,
        'mbr_rate': mbr_errors / total_move if total_move > 0 else 0,
    }
    return metrics

# 6. 训练主循环与模型保存
history = {
    'train_loss': [], 'test_loss': [],
    'train_acc': [], 'test_acc': [],
    'stay_acc': [], 'move_acc': [],
    'sbm': [], 'mbs': [], 'mbr': []
}
best_acc = 0.0
max_epochs = 200

for epoch in range(max_epochs):
    train_loss = train_epoch(model, train_loader)
    train_metrics = test_epoch(model, train_loader, criterion)
    test_metrics = test_epoch(model, test_loader, criterion)

    # Store history
    history['train_loss'].append(train_loss)
    history['test_loss'].append(test_metrics['loss'])
    history['train_acc'].append(train_metrics['total_acc'])
    history['test_acc'].append(test_metrics['total_acc'])
    history['stay_acc'].append(test_metrics['stay_acc'])
    history['move_acc'].append(test_metrics['move_acc'])
    history['sbm'].append(test_metrics['sbm_rate'])
    history['mbs'].append(test_metrics['mbs_rate'])
    history['mbr'].append(test_metrics['mbr_rate'])

    print(f"--- Epoch {epoch+1}/{max_epochs} ---")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_metrics['total_acc']:.4f}")
    print(f"Test Loss:  {test_metrics['loss']:.4f}, Test Acc:  {test_metrics['total_acc']:.4f}")
    print(f"Stay Acc:   {test_metrics['stay_acc']:.4f}, Move Acc:  {test_metrics['move_acc']:.4f}")
    print(f"Errors -> SBM: {test_metrics['sbm_rate']:.4f}, MBS: {test_metrics['mbs_rate']:.4f}, MBR: {test_metrics['mbr_rate']:.4f}")

    if test_metrics['total_acc'] > best_acc:
        best_acc = test_metrics['total_acc']
        torch.save(model.state_dict(), "model_6/best_dispatch_gnn.pt")
        print(f"保存当前最优模型到 model_6/best_dispatch_gnn.pt (Acc: {best_acc:.4f})")

torch.save(model.state_dict(), "model_6/last_dispatch_gnn.pt")
print(f"保存最后一次训练模型到 model_6/last_dispatch_gnn.pt")

# 7. 绘制训练和测试准确率随训练轮次变化的图像
fig, axs = plt.subplots(2, 2, figsize=(20, 12))
fig.suptitle('Model Training Analysis', fontsize=22)

# 定义统一的字号
title_fontsize = 18
label_fontsize = 16
legend_fontsize = 16
tick_fontsize = 16 # 坐标轴刻度字号

# Plot Loss
axs[0, 0].plot(history['train_loss'], label='Train Loss')
axs[0, 0].plot(history['test_loss'], label='Test Loss')
axs[0, 0].set_title('Loss Curve', fontsize=title_fontsize)
axs[0, 0].set_xlabel('Epoch', fontsize=label_fontsize)
axs[0, 0].set_ylabel('Loss', fontsize=label_fontsize)
axs[0, 0].tick_params(axis='both', which='major', labelsize=tick_fontsize)
axs[0, 0].legend(fontsize=legend_fontsize)
axs[0, 0].grid(True)

# Plot Accuracy
axs[0, 1].plot(history['train_acc'], label='Train Accuracy')
axs[0, 1].plot(history['test_acc'], label='Test Accuracy')
axs[0, 1].set_title('Overall Accuracy Curve', fontsize=title_fontsize)
axs[0, 1].set_xlabel('Epoch', fontsize=label_fontsize)
axs[0, 1].set_ylabel('Accuracy', fontsize=label_fontsize)
axs[0, 1].tick_params(axis='both', which='major', labelsize=tick_fontsize)
axs[0, 1].legend(fontsize=legend_fontsize)
axs[0, 1].grid(True)

# Plot Stay vs Move Accuracy
axs[1, 0].plot(history['stay_acc'], label='Stay Accuracy')
axs[1, 0].plot(history['move_acc'], label='Move Accuracy')
axs[1, 0].set_title('Stay vs. Move Accuracy (Test)', fontsize=title_fontsize)
axs[1, 0].set_xlabel('Epoch', fontsize=label_fontsize)
axs[1, 0].set_ylabel('Accuracy', fontsize=label_fontsize)
axs[1, 0].tick_params(axis='both', which='major', labelsize=tick_fontsize)
axs[1, 0].legend(fontsize=legend_fontsize)
axs[1, 0].grid(True)

# Plot Error Rates
axs[1, 1].plot(history['sbm'], label='SBM (Stay -> Move)')
axs[1, 1].plot(history['mbs'], label='MBS (Move -> Stay)')
axs[1, 1].plot(history['mbr'], label='MBR (Move -> Wrong)')
axs[1, 1].set_title('Error Rates (Test)', fontsize=title_fontsize)
axs[1, 1].set_xlabel('Epoch', fontsize=label_fontsize)
axs[1, 1].set_ylabel('Error Rate', fontsize=label_fontsize)
axs[1, 1].tick_params(axis='both', which='major', labelsize=tick_fontsize)
axs[1, 1].legend(fontsize=legend_fontsize)
axs[1, 1].grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('model_6/accuracy_and_metrics_curve.png')
print("已保存详细的性能曲线图到 model_6/accuracy_and_metrics_curve.png")