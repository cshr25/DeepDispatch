import torch
import torch.nn.functional as F
import json
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from model import MiningGNN
from data_utils import build_graph_from_scene

def train():
    # 1. 加载数据
    with open('test_data'
    '.json') as f:
        raw_data = json.load(f)
    
    # 2. 构建图数据集
    all_graphs = []
    max_shovels = 0
    for timestamp, scene in raw_data.items():
        graph = build_graph_from_scene(scene)
        # 检查是否有有效标签
        if graph is not None and (graph.y != -1).sum() > 0:
            all_graphs.append(graph)
            max_shovels = max(max_shovels, len(graph.shovel_ids))
        else:
            print(f"[跳过] 时间戳 {timestamp} 没有有效target_id的卡车")
    print(f"成功加载 {len(all_graphs)} 个有效图，最大挖机数为 {max_shovels}")

    if not all_graphs:
        raise ValueError("没有有效的训练数据！请检查target_id匹配情况")

    # 3. 数据集划分
    split_idx = int(0.8 * len(all_graphs))
    train_graphs = all_graphs[:split_idx]
    val_graphs = all_graphs[split_idx:]
    
    # 4. 训练设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MiningGNN(max_shovels=max_shovels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 5. 创建数据加载器
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=32)
    
    best_val_acc = 0.0
    train_acc_list = []
    val_acc_list = []

    # 6. 训练循环
    for epoch in range(1000):
        if epoch == 0 or epoch % 10 == 0:
            print("日志指标说明：Train Loss=训练集损失, Train Acc=训练集准确率, Val Loss=验证集损失, Val Acc=验证集准确率")
        model.train()
        train_loss = train_acc = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            truck_counts = batch.num_trucks.tolist() if hasattr(batch.num_trucks, 'tolist') else list(batch.num_trucks)
            start = 0
            targets_list = []
            out_list = []
            for count in truck_counts:
                end = start + count
                targets_list.append(batch.y[start:end])
                out_list.append(model(batch)[start:end])
                start = end
            targets = torch.cat(targets_list)
            out = torch.cat(out_list)

            valid_mask = targets != -1
            if valid_mask.sum() == 0:
                print("[Train] 跳过无有效标签的batch")
                continue

            loss = F.nll_loss(out[valid_mask], targets[valid_mask])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pred = out[valid_mask].argmax(dim=1)
            train_acc += (pred == targets[valid_mask]).float().mean().item()
        
        train_acc_epoch = train_acc / len(train_loader)
        train_acc_list.append(train_acc_epoch)

        # 验证
        model.eval()
        val_loss = val_acc = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                truck_counts = batch.num_trucks.tolist() if hasattr(batch.num_trucks, 'tolist') else list(batch.num_trucks)
                start = 0
                targets_list = []
                out_list = []
                for count in truck_counts:
                    end = start + count
                    targets_list.append(batch.y[start:end])
                    out_list.append(model(batch)[start:end])
                    start = end
                targets = torch.cat(targets_list)
                out = torch.cat(out_list)

                valid_mask = targets != -1
                if valid_mask.sum() == 0:
                    print("[Val] 跳过无有效标签的batch")
                    continue

                val_loss += F.nll_loss(out[valid_mask], targets[valid_mask]).item()
                val_acc += (out[valid_mask].argmax(dim=1) == targets[valid_mask]).float().mean().item()
        
        val_acc_epoch = val_acc / len(val_loader)
        val_acc_list.append(val_acc_epoch)

        print(f"Epoch {epoch:03d} | "
              f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_acc_epoch:.4f} | "
              f"Val Loss: {val_loss/len(val_loader):.4f} | "
              f"Val Acc: {val_acc_epoch:.4f}")
        if val_acc_epoch > best_val_acc:
            best_val_acc = val_acc_epoch
            torch.save(model.state_dict(), 'mining_gnn_model_best.pth')
            print(f"已保存当前最优模型，Val Acc={best_val_acc:.4f}")
    torch.save(model.state_dict(), 'mining_gnn_model_last.pth')
    print("训练完成，模型参数已保存为 mining_gnn_model_last.pth")

    # 绘制准确率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_acc_list, label='Train Accuracy')
    plt.plot(val_acc_list, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train & Validation Accuracy Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_curve.png')
    plt.show()

if __name__ == "__main__":
    train()