import os
import logging
import h5py

from dolfinx.io import gmshio
from dolfinx.mesh import create_submesh
from mpi4py import MPI
import numpy as np
from tqdm import tqdm
from forward_inverse_3d.reaction_diffusion.simulate_reaction_diffusion import (
    compute_v_based_on_reaction_diffusion,
)
from utils.ventricular_segmentation_tools import (
    distinguish_epi_endo,
    lv_17_segmentation_from_mesh,
    get_ischemia_segment,
)
from utils.simulate_tools import get_activation_dict


# Function to generate ischemia data
def generate_ischemia_data(
    case_name,
    severity,
    save_dir_prefix='machine_learning/data/Ischemia_Dataset/',
    gdim=3,
    T=500,
    step_per_timeframe=1,
    save_interval=200,
    partial_idx=0,
):
    save_dir = os.path.join(save_dir_prefix, f"{case_name}/{severity}/v_dataset/")
    os.makedirs(save_dir, exist_ok=True)
    logging.info("开始生成心肌缺血数据集")
    mesh_file = f'forward_inverse_3d/data/mesh/mesh_{case_name}.msh'

    activation_dict = get_activation_dict(case_name, mode='FREEWALL')

    segment_ids = lv_17_segmentation_from_mesh(mesh_file, gdim=gdim)
    domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
    tdim = domain.topology.dim
    subdomain_ventricle, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))
    valid_mask = segment_ids != -1
    center_ischemia_list = subdomain_ventricle.geometry.x[valid_mask]
    epi_endo_marker = distinguish_epi_endo(mesh_file, gdim=gdim)

    # 参数定义
    ischemia_epi_endo_list = [[1, 0], [0, -1], [1, 0, -1]]
    ischemia_epi_endo_rep = ['110', '011', '111']
    if severity == 'mild':
        radius_ischemia_list = [15]
        u_peak_ischemia_val_list = [0.9]
        u_rest_ischemia_val_list = [0.1]
    else:
        radius_ischemia_list = [30]
        u_peak_ischemia_val_list = [0.8]
        u_rest_ischemia_val_list = [0.2]
    all_v_results = []
    all_metadata = []
    all_labels = []

    total_loops = (
        len(center_ischemia_list)
        * len(radius_ischemia_list)
        * len(ischemia_epi_endo_list)
        * len(u_peak_ischemia_val_list)
    )

    with tqdm(total=total_loops, desc="生成心肌缺血数据集", dynamic_ncols=True) as pbar:
        for center_ischemia in center_ischemia_list:
            for radius_ischemia in radius_ischemia_list:
                for ischemia_epi_endo, rep_epi_endo in zip(
                    ischemia_epi_endo_list, ischemia_epi_endo_rep
                ):
                    for u_peak_ischemia_val, u_rest_ischemia_val in zip(
                        u_peak_ischemia_val_list, u_rest_ischemia_val_list
                    ):
                        try:
                            label = get_ischemia_segment(
                                subdomain_ventricle.geometry.x,
                                segment_ids,
                                epi_endo_marker,
                                center_ischemia,
                                radius_ischemia,
                                ischemia_epi_endo,
                            )
                            is_all_zero = all(v == 0 for v in label)
                            if is_all_zero:
                                pbar.update(1)
                                continue
                            v, _, _ = compute_v_based_on_reaction_diffusion(
                                mesh_file,
                                gdim=gdim,
                                ischemia_flag=True,
                                ischemia_epi_endo=ischemia_epi_endo,
                                center_ischemia=center_ischemia,
                                radius_ischemia=radius_ischemia,
                                T=T,
                                step_per_timeframe=step_per_timeframe,
                                u_peak_ischemia_val=u_peak_ischemia_val,
                                u_rest_ischemia_val=u_rest_ischemia_val,
                                activation_dict_origin=activation_dict,
                            )
                            metadata = {
                                "center": center_ischemia,
                                "radius": radius_ischemia,
                                "peak": u_peak_ischemia_val,
                                "rest": u_rest_ischemia_val,
                                "epi_endo": rep_epi_endo,
                            }
                            all_metadata.append(metadata)
                            all_v_results.append(v)
                            all_labels.append(label)
                            pbar.update(1)

                            if len(all_v_results) >= save_interval:
                                save_partial_data(
                                    all_v_results,
                                    all_labels,
                                    all_metadata,
                                    save_dir,
                                    partial_idx,
                                )
                                partial_idx += 1
                                all_v_results.clear()
                                all_labels.clear()
                                all_metadata.clear()
                        except Exception as e:
                            logging.error(f"数据生成失败: {e}")

    if all_v_results:
        save_partial_data(
            all_v_results, all_labels, all_metadata, save_dir, partial_idx
        )
        logging.info("✅ 已保存最后文件")


# Function to save partial data
def save_partial_data(v_results, labels, metadata_list, save_dir, partial_idx):
    X = np.array(v_results)
    y = np.array(labels)
    center = np.array([m["center"] for m in metadata_list])
    radius = np.array([m["radius"] for m in metadata_list])
    peak = np.array([m["peak"] for m in metadata_list])
    rest = np.array([m["rest"] for m in metadata_list])
    epi_endo = [m["epi_endo"] for m in metadata_list]
    partial_file = os.path.join(save_dir, f"v_part_{partial_idx:03d}.h5")
    with h5py.File(partial_file, "w") as f:
        f.create_dataset("X", data=X, compression="gzip")
        f.create_dataset("y", data=y, compression="gzip")
        f.create_dataset("center", data=center, compression="gzip")
        f.create_dataset("radius", data=radius, compression="gzip")
        f.create_dataset("peak", data=peak, compression="gzip")
        f.create_dataset("rest", data=rest, compression="gzip")
        f.create_dataset("epi_endo", data=epi_endo, compression="gzip")
    logging.info(f"✅ 已保存 {partial_file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    case_name_list = ['normal_male', 'normal_male2']
    for severity in ['mild', 'severe']:
        for case_name in case_name_list:
            generate_ischemia_data(case_name=case_name, severity=severity)
