# ECGI-FEM 项目文档

**Author:** 贾浩真
**Email:** 1623888385@qq.com

本项目是一个用于心电成像（ECGI）研究的综合平台，结合了基于有限元方法（FEM）的3D正/反问题求解器与深度学习方法。项目主要功能包括心脏电生理的正向仿真、缺血反演、数据生成以及基于机器学习的辅助诊断模型开发。

## 目录结构说明

```
ecgi-fem/
├── forward_inverse_3d/         # 3D 正问题与反问题求解核心代码
│   ├── convergence/            # 收敛性测试脚本
│   ├── data/                   # 数据存放目录 (raw_data 包含原始几何与激活数据)
│   ├── forward/                # 正问题求解器 (计算体表电势)
│   ├── inverse/                # 反问题求解器 (推断心脏电活动/缺血区域)
│   └── mesh/                   # 网格生成与处理脚本
├── machine_learning/           # 机器学习/深度学习模块
│   ├── create_dataset/         # 训练与测试数据集生成脚本
│   ├── dl_method/              # 深度学习模型实现
│   ├── feature_extraction/     # 特征提取工具
│   ├── check_dataset/          # 数据集检查工具
│   └── check_result/           # 结果检查与评估
├── utils/                      # 通用工具库 (数学计算、可视化、信号处理等)
│   ├── simulate_tools.py       # 仿真相关工具
│   ├── visualize_tools.py      # 可视化相关工具
│   └── ...
└── README.md                   # 项目说明文档
```

## 环境配置

本项目推荐在 **Linux** 环境或 **Windows Subsystem for Linux 2 (WSL2)** 下运行，特别是依赖 FEniCSx (dolfinx) 的部分。

### 1. 系统要求
*   **推荐系统**: Ubuntu 22.04 LTS (通过 WSL2 或原生安装)
*   **注意**: 不建议直接在 Windows 下配置 FEniCSx 环境，因为配置复杂度较高且可能遇到兼容性问题。

### 2. Python 依赖

主要依赖库包括：

#### 科学计算与有限元 (FEM)
用于正反问题求解核心算法：
*   **FEniCSx (dolfinx)**: 版本 0.8.0 (核心有限元求解器)
*   **Gmsh**: 网格生成工具
*   **PyVista**: 3D 可视化交互工具
*   **MPI4Py & PETSc4Py**: 并行计算与线性代数求解器支持

#### 机器学习 (Machine Learning)
用于数据驱动任务：
*   **PyTorch**: 深度学习框架
*   **NumPy, SciPy**: 基础科学计算
*   **Scikit-learn**: 传统机器学习算法与数据预处理
*   **h5py**: 大规模数据存储与读取

### 3. 安装指南 (详细步骤)

#### 第一步：WSL2 环境准备 (Windows 用户)

由于 FEniCSx 在 Windows 原生环境下支持有限，强烈建议使用 WSL2 (Ubuntu 22.04)。

```powershell
# 在 Windows PowerShell (管理员) 中运行：
wsl --install -d Ubuntu-22.04
# 安装完成后重启电脑，并按照提示设置 Linux 用户名和密码
```

#### 第二步：Conda 环境配置

建议使用 `conda` (推荐使用 Miniforge) 创建隔离环境。

**命令行直接安装 (推荐)**

```bash
# 1. 安装 Miniforge (如果尚未安装)
# wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
# bash Miniforge3-Linux-x86_64.sh

# 2. 创建环境并安装 FEM 核心库 (Dolfinx v0.8.0)
# 注意：使用 conda-forge 通道以确保兼容性
conda create -n ecgi-fem -c conda-forge python=3.10 fenics-dolfinx=0.8.0 mpich pyvista gmsh

# 3. 激活环境
conda activate ecgi-fem

# 4. 安装深度学习与数据处理库 
# CPU 版本 (示例):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# 或者 GPU 版本 (如果 WSL2 配置了 GPU 穿透，请访问 pytorch.org 获取具体命令):
# pip install torch torchvision torchaudio

# 5. 安装其他常用工具
pip install h5py matplotlib scikit-learn pandas tqdm scipy jupyterlab
```

#### 第三步：环境验证

安装完成后，可以在终端运行 python 并输入以下代码检查是否成功：

```python
import dolfinx
print(f"Dolfinx version: {dolfinx.__version__}")
import torch
print(f"Torch version: {torch.__version__}")
try:
    from dolfinx.io import gmshio
    print("Gmshio loaded successfully.")
except ImportError as e:
    print(f"Gmshio import failed: {e}")
```

## 功能使用指南

### 一、数据生成 (Data Generation)

#### 1. 网格生成 (Mesh Generation)
原始几何数据位于 `forward_inverse_3d/data/raw_data`。在进行仿真前，需生成有限元网格。
*   **脚本路径**: `forward_inverse_3d/mesh/`
*   **常用脚本**:
    *   `create_mesh_ecgsim_multi_conduct.py`: 生成具有不同电导率区域的网格

#### 2. 机器学习数据集构建
针对心肌缺血定位任务，使用专门的脚本生成训练集和测试集。
*   **脚本路径**: `machine_learning/create_dataset/`
*   **常用脚本**:
    *   `create_dataset_ischemia_v.py/create_dataset_healthy_v.py`: 生成心肌缺血跨膜电压数据
    *   `create_dataset_ischemia_d.py`: 生成缺血相关的体表电压数据
    *   `create_dataset_ischemia_d_standard.py`: 生成标准十二导联体表电压数据
    *   `create_dataset_ischemia_d_noisy.py`: 生成噪声数据
    *   `create_dataset_ischemia_d_processed.py`: 生成噪声数据

### 二、正/反问题求解 (Forward & Inverse Problems)

#### 正问题 (Forward Problem)
计算给定心脏跨膜电位 (TMP) 分布下的体表电位分布。
*   **主要代码**: `forward_inverse_3d/forward/forward_coupled.py`
    *   实现了心-身耦合模型的电位计算。

#### 反问题 (Inverse Problem)
根据体表电位测量值，推断心脏的电生理状态（如定位缺血区域、重建电位图）。
*   **主要代码**: `forward_inverse_3d/inverse/`
    *   `inverse_ischemia_one_timeframe.py`: 单时间帧的缺血反演
    *   `inverse_ischemia_multi_timeframe_activation_known.py`: 已知激活时间的多时间帧反演

### 三、相关工具 (Utils)

`utils/` 目录下包含了项目中复用的核心功能模块：
*   `analytic_tool.py`: 解析解计算工具，用于验证数值解。
*   `simulate_tools.py`: 包含刚度矩阵构建 (`build_Mi`, `build_M`) 等仿真核心函数。
*   `visualize_tools.py`: 用于结果的 3D 可视化。
*   `error_metrics_tools.py`: 误差分析与评估指标计算。
*   `transmembrane_potential_tools.py`: 跨膜电位相关的处理工具。

---
*文档更新日期: 2026-01-24*