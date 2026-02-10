import os
import numpy as np
import logging
import h5py
import torch
from glob import glob

from utils.ECGDimReducer_tools import ECGReducerFactory


def save_reduced_data(features, labels, save_dir, file_name):
    """保存降维后的数据"""
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, file_name)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("X", data=features, compression="gzip")
        f.create_dataset("y", data=labels, compression="gzip")

    logging.info(f"✅ 已保存 {out_path}")


def process_dataset(input_dir, output_dir, reducer, x_key="X", y_key="y"):
    """读取 input_dir 下所有 .h5 文件，使用 reducer 转换后写入 output_dir"""
    if not os.path.exists(input_dir):
        logging.warning(f"输入目录不存在: {input_dir}")
        return

    data_files = sorted(
        os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".h5")
    )

    for file in data_files:
        logging.info(f"正在处理文件: {file}")
        try:
            with h5py.File(file, "r") as data:
                X_data = data[x_key][()]
                y_data = data[y_key][()]

            # 变换数据
            X_reduced = reducer.transform(X_data)

            # 使用原始文件名
            file_name = os.path.basename(file)

            if X_reduced.size > 0:
                save_reduced_data(
                    X_reduced,
                    y_data,
                    output_dir,
                    file_name,
                )
            else:
                logging.warning(f"文件 {file} 转换后为空")

        except Exception as e:
            logging.error(f"处理文件 {file} 失败: {e}")


def create_reduced_dataset(
    case_name,
    severity,
    reducer,
    method_name,
    dir_prefix='machine_learning/data/Ischemia_Dataset/',
):
    # 处理 d64_processed_dataset -> d64_{method_name}_dataset
    process_dataset(
        input_dir=os.path.join(
            dir_prefix, f"{case_name}/{severity}/d64_processed_dataset/"
        ),
        output_dir=os.path.join(
            dir_prefix, f"{case_name}/{severity}/d64_{method_name}_dataset/"
        ),
        reducer=reducer,
    )


def collect_training_data(dirs, x_key="X"):
    """收集所有目录下的数据用于训练，确保文件顺序与 H5Dataset 一致"""
    logging.info("正在收集训练数据...")

    all_files = []
    for d in dirs:
        if not os.path.exists(d):
            continue
        # 获取目录下所有h5文件路径
        all_files.extend(glob(os.path.join(d, "*.h5")))

    # 全局排序，确保顺序与 deep_learning_tools.py 中的 H5Dataset 一致
    all_files = sorted(all_files)

    X_list = []
    for path in all_files:
        with h5py.File(path, "r") as f:
            X = f[x_key][()]
            X_list.append(X)

    if not X_list:
        raise ValueError("未找到任何训练数据")

    X_all = np.concatenate(X_list, axis=0)
    logging.info(f"训练数据收集完成，总形状: {X_all.shape}")
    return X_all


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    method_name = "cnn128"
    dir_prefix = "machine_learning/data/Ischemia_Dataset/"
    case_name_list = ['normal_male', 'normal_male2']
    severities = ['mild', 'severe', 'healthy']

    # 1. 初始化 Reducer 参数
    kwargs = {}
    if method_name == "cnn_ae":
        kwargs = {}
    elif method_name == "drpca":
        kwargs = {}
    elif method_name == "cnn128":
        kwargs = {}

    logging.info(f"初始化 {method_name} (kwargs={kwargs})...")
    reducer = ECGReducerFactory.create(method_name, **kwargs)

    # 2. 准备训练数据路径
    training_dirs = []
    for case in case_name_list:
        for sev in severities:
            path = os.path.join(dir_prefix, f"{case}/{sev}/d64_processed_dataset/")
            training_dirs.append(path)

    # 3. 训练 (Global Fit)
    try:
        X_all = collect_training_data(training_dirs)

        # 模仿 deep_learning_tools.py 中的 random_split 逻辑
        total_len = len(X_all)
        test_len = int(total_len * 0.2)
        train_len = total_len - test_len

        g = torch.Generator()
        g.manual_seed(42)

        # 获取随机排列的索引
        indices = torch.randperm(total_len, generator=g).tolist()
        train_indices = indices[:train_len]

        # 根据索引获取训练集
        X_train = X_all[train_indices]

        logging.info(f"开始训练 {method_name}...")
        reducer.fit(X_train)
        logging.info("模型训练完成。")

        # 释放内存
        del X_all
        del X_train

        # 4. 逐个文件转换并保存 (Transform Files)
        for severity in severities:
            for case_name in case_name_list:
                logging.info(f"开始转换目录: {case_name} - {severity}")
                create_reduced_dataset(
                    case_name=case_name,
                    severity=severity,
                    reducer=reducer,
                    method_name=method_name,
                    dir_prefix=dir_prefix,
                )

    except Exception as e:
        logging.error(f"程序执行出错: {e}")
