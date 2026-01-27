import os
import numpy as np
import logging
import h5py
import re


def save_concatenated_data(features, labels, save_dir, file_name):
    """保存拼接后的数据"""
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, file_name)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("X", data=features, compression="gzip")
        f.create_dataset("y", data=labels, compression="gzip")

    logging.info(f"✅ 已保存 {out_path}")


def get_file_id(filename):
    """从文件名中提取末尾的三个数字作为ID"""
    # 匹配 .h5 前的数字
    match = re.search(r'(\d+)\.h5$', filename)
    if match:
        digits = match.group(1)
        # 取最后三位
        if len(digits) >= 3:
            return digits[-3:]
        else:
            return digits.zfill(3)
    return None


def process_dataset_concatenation(
    input_dir1, input_dir2, output_dir, x_key="X", y_key="y"
):
    """
    读取 input_dir1 和 input_dir2 下的文件（通过文件名末尾3位数字匹配），
    将 X 数据沿时间维度 (axis=1) 拼接，也就是 (B, T1, D) + (B, T2, D) -> (B, T1+T2, D)
    然后写入 output_dir
    """
    if not os.path.exists(input_dir1):
        logging.warning(f"输入目录1不存在: {input_dir1}")
        return
    if not os.path.exists(input_dir2):
        logging.warning(f"输入目录2不存在: {input_dir2}")
        return

    # 获取 dataset1 中的所有 h5 文件
    data_files1 = sorted(f for f in os.listdir(input_dir1) if f.endswith(".h5"))
    # 获取 dataset2 中的所有 h5 文件
    data_files2 = sorted(f for f in os.listdir(input_dir2) if f.endswith(".h5"))

    # 建立 id -> filename 的映射
    id_map2 = {}
    for f in data_files2:
        fid = get_file_id(f)
        if fid:
            id_map2[fid] = f

    for file_name1 in data_files1:
        fid1 = get_file_id(file_name1)
        if not fid1:
            logging.warning(f"无法从文件 {file_name1} 中提取ID，跳过")
            continue

        if fid1 not in id_map2:
            logging.warning(
                f"目录2中缺少ID为 {fid1} 的对应文件 (源文件: {file_name1})，跳过"
            )
            continue

        file_name2 = id_map2[fid1]

        file_path1 = os.path.join(input_dir1, file_name1)
        file_path2 = os.path.join(input_dir2, file_name2)

        logging.info(f"正在处理匹配文件: {file_name1} <-> {file_name2}")
        try:
            with h5py.File(file_path1, "r") as f1, h5py.File(file_path2, "r") as f2:
                X1 = f1[x_key][()]
                y1 = f1[y_key][()]

                X2 = f2[x_key][()]
                y2 = f2[y_key][()]

            # 检查标签是否一致
            if not np.array_equal(y1, y2):
                logging.warning(
                    f"文件 {file_name1} 和 {file_name2} 的标签在两个数据集中不一致！将使用目录1的标签。"
                )

            # 检查数据维度
            if X1.ndim != 3 or X2.ndim != 3:
                logging.warning(
                    f"文件数据维度不符合 (B, T, D) 预期: X1={X1.shape}, X2={X2.shape}. 将尝试按 axis=1 拼接。"
                )

            # 校验 B 和 D 维度是否一致
            # 假设 shape 是 (B, T, D) -> 拼接 T (axis=1)
            # 所以 axis 0 (B) 和 axis 2 (D) 必须相同
            if X1.shape[0] != X2.shape[0]:
                logging.error(
                    f"BatchSize不匹配: {file_name1}({X1.shape}) vs {file_name2}({X2.shape})"
                )
                continue
            if X1.ndim == 3 and X2.ndim == 3 and X1.shape[2] != X2.shape[2]:
                logging.error(
                    f"特征维度(D)不匹配: {file_name1}({X1.shape}) vs {file_name2}({X2.shape})"
                )
                continue

            # 执行拼接
            X_concat = np.concatenate((X1, X2), axis=1)

            logging.info(f"  拼接完成: {X1.shape} + {X2.shape} -> {X_concat.shape}")

            # 使用文件1的文件名保存，或者根据需要修改保存名
            save_concatenated_data(
                X_concat,
                y1,
                output_dir,
                file_name1,
            )

        except Exception as e:
            logging.error(f"处理文件 {file_name1} 失败: {e}")


def create_concat_dataset(
    case_name,
    severity,
    dataset1_name,
    dataset2_name,
    output_dataset_name,
    dir_prefix='machine_learning/data/Ischemia_Dataset/',
):
    input_dir1 = os.path.join(dir_prefix, f"{case_name}/{severity}/{dataset1_name}/")
    input_dir2 = os.path.join(dir_prefix, f"{case_name}/{severity}/{dataset2_name}/")
    output_dir = os.path.join(
        dir_prefix, f"{case_name}/{severity}/{output_dataset_name}/"
    )

    process_dataset_concatenation(input_dir1, input_dir2, output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # TODO: 请在此处修改你要拼接的两个源数据集名称
    DATASET_1_NAME = "d64_cnn_ae_dataset"
    DATASET_2_NAME = "d64_features_dataset"
    OUTPUT_DATASET_NAME = "d64_cnn_ae_features_dataset"

    case_name_list = ['normal_male', 'normal_male2']
    severities = ['mild', 'severe', 'healthy']

    for case_name in case_name_list:
        for severity in severities:
            logging.info(f"Dataset Concat: {case_name} - {severity}")
            create_concat_dataset(
                case_name=case_name,
                severity=severity,
                dataset1_name=DATASET_1_NAME,
                dataset2_name=DATASET_2_NAME,
                output_dataset_name=OUTPUT_DATASET_NAME,
            )
