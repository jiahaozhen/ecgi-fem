import os
import numpy as np
import logging
import h5py
from utils.signal_processing_tools import (
    transfer_bsp_to_standard12lead,
    transfer_bsp_to_standard64lead,
)


def save_partial_bsp_data(bsp_results, seg_ids, save_dir, partial_idx):
    bsp_array = np.array(bsp_results)
    seg_ids_array = np.array(seg_ids)
    partial_file = os.path.join(save_dir, f"d_part_{partial_idx:03d}.h5")
    os.makedirs(save_dir, exist_ok=True)
    with h5py.File(partial_file, "w") as f:
        f.create_dataset("X", data=bsp_array, compression="gzip")
        f.create_dataset("y", data=seg_ids_array, compression="gzip")
    logging.info(f"✅ 已保存 {partial_file}")


def create_standard_d_dataset(
    case_name,
    severity,
    dir_prefix='machine_learning/data/Ischemia_Dataset/',
):
    d_data_files = [
        os.path.join(dir_prefix, f"{case_name}/{severity}/d_dataset/", f)
        for f in os.listdir(
            os.path.join(dir_prefix, f"{case_name}/{severity}/d_dataset/")
        )
        if f.endswith('.h5')
    ]
    d_data_files.sort()

    # Process data file-by-file to reduce memory usage
    for file_idx, file in enumerate(d_data_files):
        with h5py.File(file, "r") as data:
            d_data = data["X"][:]
            seg_ids = data["y"][:]

        standard12_d_results = []
        standard64_d_results = []
        bsp_seg_ids = []
        for i, (d, seg_id) in enumerate(zip(d_data, seg_ids)):
            try:
                standard12_d = transfer_bsp_to_standard12lead(d, case_name=case_name)
                standard64_d = transfer_bsp_to_standard64lead(d, case_name=case_name)
                standard12_d_results.append(standard12_d)
                standard64_d_results.append(standard64_d)
                bsp_seg_ids.append(seg_id)
            except Exception as e:
                logging.error(f"处理数据失败: {e}")

        if standard12_d_results and standard64_d_results:
            save_partial_bsp_data(
                standard12_d_results,
                bsp_seg_ids,
                os.path.join(dir_prefix, f"{case_name}/{severity}/d12_dataset/"),
                file_idx,
            )
            save_partial_bsp_data(
                standard64_d_results,
                bsp_seg_ids,
                os.path.join(dir_prefix, f"{case_name}/{severity}/d64_dataset/"),
                file_idx,
            )
            logging.info(f"✅ 已处理文件 {file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    case_name_list = ['normal_male', 'normal_male2', 'normal_young_male']
    for severity in ['mild', 'severe']:
        for case_name in case_name_list:
            create_standard_d_dataset(case_name=case_name, severity=severity)
