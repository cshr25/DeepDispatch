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


class WeightedRewardLoss(nn.Module):
    """
    自定义损失函数，对不同类型的错误施加不同的惩罚权重。
    - MBS (Move But Stay): 实际移动，但预测停留。 (最高惩罚)
    - MBR (Move But Wrong): 实际移动，预测也移动，但目标错误。(较高惩罚)
    - SBM (Stay But Move): 实际停留，但预测移动。(普通惩罚)
    """
    def __init__(self, move_weight=2.0, mbs_penalty=4.0, mbr_penalty=3.0, sbm_penalty=1.5, device='cpu'):
        super().__init__()
        self.move_weight = move_weight # 新增：移动样本的基础权重
        self.mbs_penalty = mbs_penalty
        self.mbr_penalty = mbr_penalty
        self.sbm_penalty = sbm_penalty
        self.device = device

    def forward(self, outputs, data):
        targets = data.y
        current_targets = data.cur_targets
        base_loss = F.cross_entropy(outputs, targets, reduction='none')
        weights = torch.ones_like(base_loss, device=self.device)
        _, predicted = torch.max(outputs, 1)
        is_true_stay = (targets == current_targets)
        is_true_move = ~is_true_stay
        is_pred_stay = (predicted == current_targets)
        is_pred_move = ~is_pred_stay
        weights[is_true_move] = self.move_weight
        mbs_mask = is_true_move & is_pred_stay
        weights[mbs_mask] = self.mbs_penalty
        mbr_mask = is_true_move & is_pred_move & (predicted != targets)
        weights[mbr_mask] = self.mbr_penalty
        sbm_mask = is_true_stay & is_pred_move
        weights[sbm_mask] = self.sbm_penalty      
        weighted_loss = base_loss * weights
        return weighted_loss.mean()
