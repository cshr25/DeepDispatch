import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class MiningGNN(nn.Module):
    def __init__(self, max_shovels=4):
        super().__init__()
        self.max_shovels = max_shovels
        self.node_encoder = nn.Linear(5, 32)  # 节点特征维度改为5
        self.conv1 = GATConv(32, 32, edge_dim=4)  # 边特征维度改为4
        self.conv2 = GATConv(32, 32, edge_dim=4)
        self.decoder = nn.Linear(32, max_shovels)

    def forward(self, data):
        h = F.relu(self.node_encoder(data.x))
        h = F.relu(self.conv1(h, data.edge_index, data.edge_attr))
        h = self.conv2(h, data.edge_index, data.edge_attr)

        if hasattr(data, 'batch') and data.batch is not None:
            truck_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            for i in range(data.num_graphs):
                graph_mask = (data.batch == i)
                graph_indices = graph_mask.nonzero(as_tuple=True)[0]
                num_trucks = data.num_trucks[i].item()
                truck_mask[graph_indices[:num_trucks]] = True
            truck_features = h[truck_mask]
        else:
            truck_features = h[:data.num_trucks]

        if truck_features.dim() == 1:
            truck_features = truck_features.unsqueeze(0)

        out = self.decoder(truck_features)
        return F.log_softmax(out, dim=1)
