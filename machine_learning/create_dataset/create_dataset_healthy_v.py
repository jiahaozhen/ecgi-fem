import os
import logging
import h5py

import numpy as np
from tqdm import tqdm
from forward_inverse_3d.reaction_diffusion.simulate_reaction_diffusion import (
    compute_v_based_on_reaction_diffusion,
)
from utils.simulate_tools import get_activation_dict


# Function to generate ischemia data
def generate_ischemia_data(
    case_name,
    save_dir_prefix='machine_learning/data/Ischemia_Dataset/',
    gdim=3,
    T=500,
    step_per_timeframe=1,
    save_interval=200,
    partial_idx=0,
):
    save_dir = os.path.join(save_dir_prefix, f"{case_name}/healthy/v_dataset/")
    os.makedirs(save_dir, exist_ok=True)
    logging.info("开始生成心肌缺血数据集")
    mesh_file = f'forward_inverse_3d/data/mesh/mesh_{case_name}.msh'

    activation_dict = get_activation_dict(case_name, mode='FREEWALL')

    all_v_results = []
    all_seg_ids = []

    num_cases = 100

    v, _, _ = compute_v_based_on_reaction_diffusion(
        mesh_file,
        gdim=gdim,
        T=T,
        step_per_timeframe=step_per_timeframe,
        activation_dict_origin=activation_dict,
    )

    for case_idx in tqdm(range(num_cases), desc="生成健康心脏数据"):
        try:

            label = np.zeros(17, dtype=np.int64)

            all_v_results.append(v.copy())

            all_seg_ids.append(label.tolist())

            # 定期保存数据以节省内存
            if (case_idx + 1) % save_interval == 0:
                save_partial_data(all_v_results, all_seg_ids, save_dir, partial_idx)
                partial_idx += 1
                all_v_results = []
                all_seg_ids = []

        except Exception as e:
            logging.error(f"处理数据失败: {e}")

    if all_v_results:
        save_partial_data(all_v_results, all_seg_ids, save_dir, partial_idx)
        logging.info("✅ 已保存最后文件")


# Function to save partial data
def save_partial_data(v_results, seg_ids, save_dir, partial_idx):
    X = np.array(v_results)
    y = np.array(seg_ids)
    partial_file = os.path.join(save_dir, f"v_part_{partial_idx:03d}.h5")
    with h5py.File(partial_file, "w") as f:
        f.create_dataset("X", data=X, compression="gzip")
        f.create_dataset("y", data=y, compression="gzip")
    logging.info(f"✅ 已保存 {partial_file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    case_name_list = ['normal_male', 'normal_male2']
    for case_name in case_name_list:
        generate_ischemia_data(case_name=case_name)
