import torch
import math
from torch_geometric.data import Data

def validate_scene(scene_data):
    """验证场景数据是否有效"""
    shovels_ids = {s['shovel_id'] for s in scene_data.get('shovels', [])}
    for truck in scene_data.get('trucks', []):
        if truck['target_id'] not in shovels_ids:
            print(f"警告: 卡车 {truck['truck_id']} 的 target_id={truck['target_id']} 不存在于挖机列表")
            return False
    return True

def normalize_position(x, y):
    """坐标归一化（基于矿区实际范围）"""
    min_x, max_x = 540000, 550000
    min_y, max_y = 4780000, 4790000
    return (x - min_x)/(max_x - min_x), (y - min_y)/(max_y - min_y)

def get_truck_features(truck):
    """卡车节点特征：位置、速度、是否空载"""
    x_norm, y_norm = normalize_position(truck['position']['x'], truck['position']['y'])
    features = [
        x_norm,  # 归一化x坐标
        y_norm,  # 归一化y坐标
        truck['speed'] / 20.0,  # 归一化速度
        0.0 if truck['task_state']['code'] == 3 else 1.0,  # 是否空载
        0.0  # 补零
    ]
    return features

def get_shovel_features(shovel):
    """挖机节点特征：位置、工作状态、队列长度、装载率"""
    x_norm, y_norm = normalize_position(shovel['position']['x'], shovel['position']['y'])
    return [
        x_norm,  # 归一化x坐标
        y_norm,  # 归一化y坐标
        float(shovel['working_status']),  # 工作状态
        min(shovel['queue_length'] / 5.0, 1.0),  # 归一化队列长度
        shovel['loading_rate'] / 250.0  # 归一化装载率
    ]

def build_graph_from_scene(scene_data):
    """
    构建场景图数据
    节点特征不包含目标挖机信息，避免标签泄漏。
    边为卡车与所有挖机全连接，边特征包含距离、挖机状态、队列长度、装载率。
    """
    trucks = scene_data.get("trucks", [])
    shovels = scene_data.get("shovels", [])
    shovel_ids = sorted({s['shovel_id'] for s in shovels})
    shovel_id_to_idx = {id: idx for idx, id in enumerate(shovel_ids)}

    # 节点特征
    truck_features = [get_truck_features(truck) for truck in trucks]
    x_truck = torch.tensor(truck_features, dtype=torch.float) if truck_features else torch.empty((0, 5))
    shovel_features = [get_shovel_features(shovel) for shovel in shovels]
    x_shovel = torch.tensor(shovel_features, dtype=torch.float) if shovel_features else torch.empty((0, 5))

    # 边构造（卡车与所有挖机全连接）
    edge_index = []
    edge_attr = []
    for truck_idx, truck in enumerate(trucks):
        for shovel_idx, shovel in enumerate(shovels):
            edge_index.append([truck_idx, len(trucks) + shovel_idx])
            distance = math.sqrt(
                (truck['position']['x'] - shovel['position']['x'])**2 +
                (truck['position']['y'] - shovel['position']['y'])**2
            )
            edge_attr.append([
                distance / 500.0,                        # 距离
                float(shovel['working_status']),          # 挖机工作状态
                min(shovel['queue_length'] / 5.0, 1.0),  # 队列长度
                shovel['loading_rate'] / 250.0           # 装载率
            ])

    # 生成标签（卡车目标挖机的索引）
    y_labels = []
    for truck in trucks:
        if truck.get('target_id') in shovel_id_to_idx:
            y_labels.append(shovel_id_to_idx[truck['target_id']])
        else:
            y_labels.append(-1)  # 无效target_id时用-1标记

    return Data(
        x=torch.cat([x_truck, x_shovel], dim=0),
        edge_index=torch.tensor(edge_index).t().contiguous() if edge_index else torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float) if edge_attr else torch.empty((0, 4)),
        y=torch.tensor(y_labels, dtype=torch.long),  # y: 卡车目标挖机索引（对应 shovel_ids 下标）
        num_trucks=len(trucks),
        shovel_ids=shovel_ids
    )

def build_temporal_dataset(raw_data):
    """处理时间序列数据"""
    return [build_graph_from_scene(scene) for timestamp, scene in raw_data.items()]