# DeepDispatch

基于深度图神经网络的矿山全局调度拟合系统

## 项目概述

DeepDispatch 是一个使用图神经网络(GNN)拟合现有矿山卡车调度的智能系统。该系统将矿山场景建模为图结构，其中卡车和挖机作为节点，通过图注意力网络(GAT)预测最优的卡车-挖机分配策略。

## 项目结构

```
DeepDispatch/
├── README.md           # 项目说明文档
├── model.py           # GNN模型定义 (MiningGNN)
├── data_utils.py      # 数据预处理工具
├── train.py           # 主训练脚本
└── example_data.json  # 示例数据
```

### 核心模块说明

- **model.py**: 包含 `MiningGNN` 类，实现基于图注意力网络的调度模型
- **data_utils.py**: 数据预处理模块，将JSON格式的矿山场景数据转换为PyTorch Geometric图对象
- **train.py**: 主程序，包含完整的训练流程、模型评估和结果可视化
- **example_data.json**: 矿山场景数据样例，包含卡车位置、速度、任务状态和挖机信息

## GNN设计架构

### 图结构设计

- **节点类型**:
  - 卡车节点：包含位置(x,y)、速度、装载状态等特征
  - 挖机节点：包含位置(x,y)、工作状态、队列长度、装载率等特征
- **边连接**: 卡车与所有挖机全连接（二分图结构）
- **边特征**: 距离、挖机工作状态、队列长度、装载率

### 模型架构

```python
MiningGNN(
  ├── node_encoder: Linear(5, 32)     # 节点特征编码
  ├── conv1: GATConv(32, 32)          # 第一层图注意力
  ├── conv2: GATConv(32, 32)          # 第二层图注意力  
  └── decoder: Linear(32, max_shovels) # 输出层
)
```

### 特征工程

- **坐标归一化**: 基于矿区实际范围 (540000-550000, 4780000-4790000)
- **速度归一化**: 除以最大速度20.0
- **状态编码**: 卡车装载状态、挖机工作状态的二进制编码
- **距离特征**: 卡车到挖机的欧几里得距离，归一化到[0,1]

## 环境配置

### 依赖要求

```bash
pip install torch
pip install torch-geometric
pip install matplotlib
```

### 数据格式

输入数据为JSON格式，包含时间戳、卡车信息和挖机信息：

```json
{
  "timestamp": {
    "trucks": [{
      "truck_id": "string",
      "position": {"x": float, "y": float},
      "speed": float,
      "task_state": {"code": int},
      "target_id": "string"
    }],
    "shovels": [{
      "shovel_id": "string",
      "position": {"x": float, "y": float}, 
      "working_status": int,
      "queue_length": int,
      "loading_rate": float
    }]
  }
}
```

## 使用方法

### 训练模型

```bash
cd DeepDispatch
python train.py
```

### 训练过程

1. 从 `test_data.json` 加载数据
2. 构建图数据集，过滤无效标签
3. 80/20划分训练验证集
4. 1000轮训练，Adam优化器，学习率0.001
5. 保存最优模型和最终模型
6. 生成准确率曲线图

### 输出文件

- `mining_gnn_model_best.pth`: 验证集上表现最佳的模型
- `mining_gnn_model_last.pth`: 最终训练完成的模型
- `accuracy_curve.png`: 训练和验证准确率曲线图

## 技术特点

- **标签泄漏防护**: 节点特征不包含目标挖机信息
- **批处理支持**: 支持变长图的批量训练
- **无效数据处理**: 自动过滤target_id不匹配的样本
- **可视化输出**: 自动生成训练过程可视化图表

## 更新日志

- **2025/7/25**: 上传第一版代码，实现基础GNN调度模型
