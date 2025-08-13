import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data as GeoData, Batch
from sklearn.preprocessing import LabelEncoder, StandardScaler

def data_preparation(input_csv, le_target=None, le_truck=None, scaler=None, fit=True):
    df = pd.read_csv(input_csv)
    df = df.sort_values(['report_truck_id', 'header_stamp_sec'])
    df['next_target_id'] = df.groupby('report_truck_id')['result.pose_id'].shift(-1)
    df = df.dropna(subset=['next_target_id'])

    if fit:
        le_target = LabelEncoder()
        le_truck = LabelEncoder()
        scaler = StandardScaler()
        df['cur_target_enc'] = le_target.fit_transform(df['result.pose_id'])
        df['next_target_enc'] = le_target.transform(df['next_target_id'])
        df['truck_id_enc'] = le_truck.fit_transform(df['report_truck_id'])
        df['task_state_norm'] = scaler.fit_transform(df[['report_truck_info_task_state_value']])
        return df, le_target, le_truck, scaler
    else:
        df['cur_target_enc'] = le_target.transform(df['result.pose_id'])
        df['next_target_enc'] = le_target.transform(df['next_target_id'])
        df['truck_id_enc'] = le_truck.transform(df['report_truck_id'])
        df['task_state_norm'] = scaler.transform(df[['report_truck_info_task_state_value']])
        return df

def construct_graph_data(df, num_nodes=43):
    node_types = np.array([0]*27 + [1]*16)  # 0:排队点, 1:调度终点 节点特征
    adj_matrix = np.loadtxt('adj_matrix.csv', delimiter=',', dtype=int)  # 邻接矩阵
    # 图的边索引
    edge_index = []  # 边索引
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i, j] == 1:
                edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    samples = []  # 创建训练样本列表
    grouped = df.groupby('header_stamp_sec')  # 按时间戳分组
    for ts, group in grouped:  # 每个时间戳一个样本
        lock_count = np.array([sum(group['cur_target_enc'] == i) for i in range(num_nodes)], dtype=np.float32).reshape(-1, 1)  # 每个终点被车辆锁定的数量
        node_type_feat = node_types.reshape(-1, 1).astype(np.float32)  # 0/1类别
        node_features = np.concatenate([lock_count, node_type_feat], axis=1)  # shape: [num_nodes, 2]
        # 车辆特征：卡车ID、当前时间戳目标、前一个时间戳目标、任务状态值、下一个时间戳目标、任务状态值
        trucks = group['truck_id_enc'].values 
        task_states = group['task_state_norm'].values
        cur_targets = group['cur_target_enc'].values
        next_targets = group['next_target_enc'].values
        truck_features = np.stack([trucks, task_states, cur_targets], axis=1)  # shape: [num_trucks, 3]
        samples.append({
            'node_features': node_features,  # shape: [num_nodes, 2] 节点特征（被作为目的地数量、节点类型）
            'truck_features': truck_features,  # shape: [num_trucks, 3] 车辆特征 （卡车ID、任务状态、当前目标）
            'truck_indices': trucks,  # shape: [num_trucks] 卡车ID编码
            'next_targets': next_targets,  # shape: [num_trucks] 下一个目标编码
            'cur_targets': cur_targets  # shape: [num_trucks] 当前目标编码
        })
    return samples, edge_index

class DispatchGraphDataset(Dataset): # 自定义数据集类
    def __init__(self, samples, edge_index):
        self.samples = samples 
        self.edge_index = edge_index
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx] # 获取样本
        x = torch.tensor(s['node_features'], dtype=torch.float32) # 节点特征
        edge_idx = self.edge_index # 边索引
        truck_x = torch.tensor(s['truck_features'], dtype=torch.float32) # 车辆特征
        y = torch.tensor(s['next_targets'], dtype=torch.long) # 下一个目标编码
        truck_indices = torch.tensor(s['truck_indices'], dtype=torch.long) # 卡车ID编码
        cur_targets = torch.tensor(s['cur_targets'], dtype=torch.long) # 当前目标编码
        data = GeoData(x=x, edge_index=edge_idx) # 构造图数据对象
        data.truck_x = truck_x
        data.y = y
        data.truck_indices = truck_indices
        data.cur_targets = cur_targets
        return data