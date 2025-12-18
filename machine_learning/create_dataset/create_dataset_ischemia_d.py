import os
import numpy as np
import logging
import h5py
from forward_inverse_3d.forward.forward_coupled_matrix_form import compute_d_from_tmp


def save_partial_bsp_data(bsp_results, seg_ids, save_dir, partial_idx):
    bsp_array = np.array(bsp_results)
    seg_ids_array = np.array(seg_ids)
    partial_file = os.path.join(save_dir, f"d_part_{partial_idx:03d}.h5")
    os.makedirs(save_dir, exist_ok=True)
    with h5py.File(partial_file, "w") as f:
        f.create_dataset("X", data=bsp_array, compression="gzip")
        f.create_dataset("y", data=seg_ids_array, compression="gzip")
    logging.info(f"✅ 已保存 {partial_file}")


def create_ischemia_d_dataset(
    case_name,
    severity,
    dir_prefix='machine_learning/data/Ischemia_Dataset/',
):
    logging.info("开始生成心肌缺血D数据集")

    v_data_files = [
        os.path.join(dir_prefix, f"{case_name}/{severity}/v_dataset/", f)
        for f in os.listdir(
            os.path.join(dir_prefix, f"{case_name}/{severity}/v_dataset/")
        )
        if f.endswith('.h5')
    ]
    v_data_files.sort()

    # Process data file-by-file to reduce memory usage
    for file_idx, file in enumerate(v_data_files):
        with h5py.File(file, "r") as data:
            v_data = data["X"][:]
            seg_ids = data["y"][:]

        bsp_results = []
        bsp_seg_ids = []
        for i, (v, seg_id) in enumerate(zip(v_data, seg_ids)):
            try:
                bsp = compute_d_from_tmp(case_name, v, allow_cache=True)
                bsp_results.append(bsp)
                bsp_seg_ids.append(seg_id)
            except Exception as e:
                logging.error(f"处理数据失败: {e}")

        if bsp_results:
            save_partial_bsp_data(
                bsp_results,
                bsp_seg_ids,
                os.path.join(dir_prefix, f"{case_name}/{severity}/d_dataset/"),
                file_idx,
            )
            logging.info(f"✅ 已处理文件 {file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    case_name_list = ['normal_male', 'normal_male2']
    for severity in ['mild', 'severe', 'healthy']:
        for case_name in case_name_list:
            create_ischemia_d_dataset(case_name=case_name, severity=severity)
