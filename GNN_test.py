import torch
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch.optim as optim
from model import DispatchGNN
from data_utils import DispatchGraphDataset, data_preparation, construct_graph_data
import pickle
import matplotlib.pyplot as plt

input_csv = "data_test_4.csv"
os.makedirs('result', exist_ok=True)

# 1. 数据读取与特征构造
# 加载编码器和标准化器
with open('model/le_target.pkl', 'rb') as f:
    le_target = pickle.load(f)
with open('model/le_truck.pkl', 'rb') as f:
    le_truck = pickle.load(f)
with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
df_test = data_preparation(input_csv, le_target, le_truck, scaler, fit=False)

# 2. 图数据构建
samples_test, edge_index = construct_graph_data(df_test)

# 3. 模型定义
node_in_dim = 2  # 节点特征维度（被作为目的地数量、节点类型）
truck_in_dim = 3 # 车辆特征维度（当前目标、前一个目标、卡车ID、任务状态）
hidden_dim = 32 # 隐藏层维度
num_classes = 43 # 目标类别数量（节点数量）43
model = DispatchGNN(node_in_dim, truck_in_dim, hidden_dim, num_classes,)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = nn.CrossEntropyLoss() # 损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001) # 优化器

#4. 测试集,加载器,最优模型
test_dataset_new = DispatchGraphDataset(samples_test, edge_index)
test_loader_new = DataLoader(test_dataset_new, batch_size=1)
model.load_state_dict(torch.load("model_2/best_dispatch_gnn.pt"))
model.eval()

#5. 测试并输出结果，同时统计各样本准确率和整体准确率
output_lines = []
total = 0
correct = 0
sample_acc_list = []

with torch.no_grad():
    for idx, data in enumerate(test_loader_new):
        data = data.to(device)
        outputs = model(data)
        pred = outputs.argmax(dim=1).cpu().numpy()
        truck_ids = le_truck.inverse_transform(data.truck_indices.cpu().numpy())
        target_names = le_target.inverse_transform(pred)
        true_targets = le_target.inverse_transform(data.y.cpu().numpy())
        sample_total = len(truck_ids)
        sample_correct = sum(tname == ttrue for tname, ttrue in zip(target_names, true_targets))
        sample_acc = sample_correct / sample_total if sample_total > 0 else 0
        sample_acc_list.append(sample_acc)
        output_lines.append(f"样本{idx}:")
        for tid, tname, ttrue in zip(truck_ids, target_names, true_targets):
            output_lines.append(f"  车辆ID: {tid}，预测下一个终点: {tname}，真实终点: {ttrue}")
        output_lines.append(f"  样本准确率: {sample_acc:.4f}")
        total += sample_total
        correct += sample_correct

overall_acc = correct / total if total > 0 else 0
output_lines.append(f"整体预测准确率: {overall_acc:.4f}")
with open('result/model_testdata_result.txt', 'w', encoding='utf-8') as f:
    for line in output_lines:
        f.write(line + '\n')
print(f"整体预测准确率: {overall_acc:.4f}")


# 可视化各样本准确率（柱状图），整体准确率用不同颜色
plt.figure(figsize=(12,6))
bars = plt.bar(range(len(sample_acc_list)), sample_acc_list, color='skyblue', label='Overall Accuracy')
# 整体准确率用红色横线
plt.axhline(y=correct/total, color='red', linestyle='--', linewidth=2, label=f'Overall Accuracy: {correct/total:.4f}')
plt.xlabel('sample_ID')
plt.ylabel('sample_accuracy')
plt.title('Prediction accuracy distribution across samples')
plt.legend()
plt.savefig('result/mode_testdata.png')
plt.show()
