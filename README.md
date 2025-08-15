# GNN-based Vehicle Dispatch Prediction

本项目基于图神经网络（GNN）实现车辆调度终点预测，适用于多车辆、多终点的调度场景。项目包含数据预处理、模型训练、模型测试、结果可视化等完整流程。

## 目录结构

```
.
├── GNN_train.py                # 训练主程序
├── GNN_train2.py               # 训练主程序（更新）
├── GNN_test.py                 # 测试主程序
├── model.py                    # GNN模型定义
├── data_utils.py               # 数据处理与图数据构建
├── data_train.csv              # 训练数据
├── data_test.csv               # 测试数据
├── adj_matrix.csv              # 图结构邻接矩阵
├── result/
│   └── result.txt              # 测试结果输出
├── model/
│   ├── best_dispatch_gnn.pt    # 最优模型参数
│   ├── last_dispatch_gnn.pt    # 最后一次训练模型参数
│   ├── le_target.pkl           # 目标编码器
│   ├── le_truck.pkl            # 车辆编码器
│   └── scaler.pkl              # 标准化器
```

## 快速开始

### 1. 安装依赖

建议使用 Python 3.8+，并提前安装如下依赖：

```bash
pip install torch torch-geometric scikit-learn matplotlib tqdm pandas numpy
```

### 2. 数据准备

- `data_train.csv`：训练数据，包含车辆ID、时间戳、任务状态、目标ID等字段。
- `data_test.csv`：测试数据，结构同训练数据。
- `adj_matrix.csv`：43x43的邻接矩阵，定义图结构。

### 3. 训练模型

运行训练脚本，自动保存模型和编码器：

```bash
python GNN_train.py
```

- 训练过程中会保存最优模型、最后一次模型、编码器和标准化器到 `model/` 文件夹。
- 训练和测试准确率曲线保存在 `model/accuracy_curve.png`。

### 4. 测试模型

运行测试脚本，输出预测结果和可视化：

```bash
python GNN_test.py
```

- 预测结果（每辆车的预测与真实终点、各样本准确率、整体准确率）保存在 `result/result.txt`。
- 各样本准确率柱状图保存在 `result/accuracy_curve.png`。

## 主要文件说明

- **GNN_train.py**：训练流程，包括数据处理、模型训练、准确率曲线绘制、模型与编码器保存。
- **GNN_test.py**：加载模型和编码器，对测试集进行预测并输出详细结果与可视化。
- **model.py**：核心GNN模型结构，包含节点特征聚合、车辆特征映射、特征融合与输出层。
- **data_utils.py**：数据预处理、编码器/标准化器适配、图数据构建、PyG数据集定义。

## 结果输出

- `result/result.txt`：每个样本的预测详情、准确率、整体准确率。
- `result/accuracy_curve.png`：各样本准确率分布柱状图，红线为整体准确率。

## 备注

- 训练和测试脚本完全解耦，编码器和标准化器通过 `pickle` 文件传递，保证测试时与训练一致。
- 支持自定义数据和图结构，只需替换对应 csv 文件即可。

## 更新日志
- 2025年8月13日，上传该分支的第一版代码。
- 2025年8月15日，上传GNN_train2.py，修改车辆特征、自定义损失函数、增加更多的评估指标
