import os
import numpy as np
import logging
import h5py
from utils.signal_processing_tools import add_noise_based_on_snr, normalize_ecg_zscore


def save_partial_bsp_data(bsp_results, seg_ids, save_dir, partial_idx):
    bsp_array = np.array(bsp_results)
    seg_ids_array = np.array(seg_ids)
    partial_file = os.path.join(save_dir, f"d_part_{partial_idx:03d}.h5")
    os.makedirs(save_dir, exist_ok=True)
    with h5py.File(partial_file, "w") as f:
        f.create_dataset("X", data=bsp_array, compression="gzip")
        f.create_dataset("y", data=seg_ids_array, compression="gzip")
    logging.info(f"✅ 已保存 {partial_file}")


def process_dataset(input_dir, output_dir, snr_db=40):
    """读取 input_dir 下所有 .h5 文件，加噪后写入 output_dir"""
    data_files = sorted(
        os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".h5")
    )

    for file_idx, file in enumerate(data_files):
        with h5py.File(file, "r") as data:
            d_data = data["X"][:]
            seg_ids = data["y"][:]

        processed_results = []
        seg_results = []

        for d, seg_id in zip(d_data, seg_ids):
            try:
                noise_d = add_noise_based_on_snr(d, snr=snr_db)
                processed_d = normalize_ecg_zscore(noise_d)
                processed_results.append(processed_d)
                seg_results.append(seg_id)
            except Exception as e:
                logging.error(f"处理数据失败: {e}")

        if processed_results:
            save_partial_bsp_data(
                processed_results,
                seg_results,
                output_dir,
                file_idx,
            )
            logging.info(f"✅ 已处理文件 {file}")


def create_d_dataset_with_pretreatment(
    case_name,
    severity,
    dir_prefix='machine_learning/data/Ischemia_Dataset/',
):
    process_dataset(
        input_dir=os.path.join(dir_prefix, f"{case_name}/{severity}/d12_dataset/"),
        output_dir=os.path.join(
            dir_prefix, f"{case_name}/{severity}/d12_processed_dataset/"
        ),
    )
    process_dataset(
        input_dir=os.path.join(dir_prefix, f"{case_name}/{severity}/d64_dataset/"),
        output_dir=os.path.join(
            dir_prefix, f"{case_name}/{severity}/d64_processed_dataset/"
        ),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    case_name_list = ['normal_male', 'normal_male2']
    for severity in ['mild', 'severe', 'healthy']:
        for case_name in case_name_list:
            create_d_dataset_with_pretreatment(case_name=case_name, severity=severity)
