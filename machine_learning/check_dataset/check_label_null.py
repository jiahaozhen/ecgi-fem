import logging

from dolfinx.io import gmshio
from dolfinx.mesh import create_submesh
from mpi4py import MPI
from utils.ventricular_segmentation_tools import (
    distinguish_epi_endo,
    lv_17_segmentation_from_mesh,
    get_ischemia_segment,
)

CNT = 0


# Function to generate ischemia data
def generate_ischemia_data(
    case_name,
    severity,
    gdim=3,
):
    global CNT
    mesh_file = f'forward_inverse_3d/data/mesh/mesh_{case_name}.msh'

    segment_ids = lv_17_segmentation_from_mesh(mesh_file, gdim=gdim)
    domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
    tdim = domain.topology.dim
    subdomain_ventricle, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))
    valid_mask = segment_ids != -1
    center_ischemia_list = subdomain_ventricle.geometry.x[valid_mask]
    epi_endo_marker = distinguish_epi_endo(mesh_file, gdim=gdim)

    # 参数定义
    ischemia_epi_endo_list = [[1, 0], [0, 1], [-1, 0, 1]]
    if severity == 'mild':
        radius_ischemia_list = [15]
    else:
        radius_ischemia_list = [30]

    for center_ischemia in center_ischemia_list:
        for radius_ischemia in radius_ischemia_list:
            for ischemia_epi_endo in ischemia_epi_endo_list:
                try:
                    label = get_ischemia_segment(
                        subdomain_ventricle.geometry.x,
                        segment_ids,
                        epi_endo_marker,
                        center_ischemia,
                        radius_ischemia,
                        ischemia_epi_endo,
                        ratio_threshold=0.1,
                    )
                    is_all_zero = all(v == 0 for v in label)
                    if is_all_zero:
                        CNT += 1

                except Exception as e:
                    logging.error(f"数据生成失败: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    case_name_list = ['normal_male', 'normal_male2']
    for severity in ['mild', 'severe']:
        for case_name in case_name_list:
            generate_ischemia_data(case_name=case_name, severity=severity)
    print(CNT)
