import os
import numpy as np
import logging
import h5py

from utils.signal_processing_tools import batch_extract_statistical_features


def save_feature_data(features, seg_ids, save_dir, partial_idx):
    features_array = np.array(features)
    seg_ids_array = np.array(seg_ids)

    partial_file = os.path.join(save_dir, f"features_part_{partial_idx:03d}.h5")
    os.makedirs(save_dir, exist_ok=True)

    with h5py.File(partial_file, "w") as f:
        f.create_dataset("X", data=features_array, compression="gzip")
        f.create_dataset("y", data=seg_ids_array, compression="gzip")

    logging.info(f"✅ 已保存 {partial_file}")


def process_dataset(input_dir, output_dir, fs=1000):
    """读取 input_dir 下所有 .h5 文件，提取特征后写入 output_dir"""
    if not os.path.exists(input_dir):
        logging.warning(f"输入目录不存在: {input_dir}")
        return

    data_files = sorted(
        os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".h5")
    )

    for file_idx, file in enumerate(data_files):
        logging.info(f"正在处理文件: {file}")
        try:
            with h5py.File(file, "r") as data:
                d_data = data["X"][:]
                seg_ids = data["y"][:]

            # 提取特征
            # batch_extract_features 返回 (B, D, F) 和 feature_names
            # features, feature_names = batch_extract_features(d_data, fs=fs)
            features, feature_names = batch_extract_statistical_features(d_data, fs=fs)

            if features.size > 0:
                save_feature_data(
                    features,
                    seg_ids,
                    output_dir,
                    file_idx,
                )
            else:
                logging.warning(f"文件 {file} 未提取到有效特征")

        except Exception as e:
            logging.error(f"处理文件 {file} 失败: {e}")


def create_features_dataset(
    case_name,
    severity,
    dir_prefix='machine_learning/data/Ischemia_Dataset/',
):
    # 处理 d12_noisy_dataset -> d12_features_dataset
    process_dataset(
        input_dir=os.path.join(
            dir_prefix, f"{case_name}/{severity}/d12_noisy_dataset/"
        ),
        output_dir=os.path.join(
            dir_prefix, f"{case_name}/{severity}/d12_statistical_features_dataset/"
        ),
    )
    # 处理 d64_noisy_dataset -> d64_features_dataset
    process_dataset(
        input_dir=os.path.join(
            dir_prefix, f"{case_name}/{severity}/d64_noisy_dataset/"
        ),
        output_dir=os.path.join(
            dir_prefix, f"{case_name}/{severity}/d64_statistical_features_dataset/"
        ),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    case_name_list = ['normal_male', 'normal_male2']
    for severity in ['mild', 'severe', 'healthy']:
        for case_name in case_name_list:
            logging.info(f"开始处理: {case_name} - {severity}")
            create_features_dataset(case_name=case_name, severity=severity)
