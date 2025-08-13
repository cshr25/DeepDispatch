import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class DispatchGNN(nn.Module): # 定义图神经网络模型
    def __init__(self, node_in_dim, truck_in_dim, hidden_dim, num_classes):
        super().__init__()
        self.node_gnn = GCNConv(node_in_dim, hidden_dim)
        self.truck_fc = nn.Linear(truck_in_dim, hidden_dim)
        self.combine_fc = nn.Linear(hidden_dim*2, hidden_dim)
        self.out_fc = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
    def forward(self, data):
        node_emb = self.relu(self.node_gnn(data.x, data.edge_index)) 
        truck_emb = self.relu(self.truck_fc(data.truck_x))
        node_emb_for_truck = node_emb[data.cur_targets]
        comb = torch.cat([truck_emb, node_emb_for_truck], dim=1)
        comb = self.relu(self.combine_fc(comb))
        out = self.out_fc(comb)
        return out
