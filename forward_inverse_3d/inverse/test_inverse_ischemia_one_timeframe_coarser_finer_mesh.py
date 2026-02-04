import numpy as np
from forward_inverse_3d.inverse.inverse_ischemia_one_timeframe import (
    ischemia_inversion,
)
from utils.mesh_tools import (
    extract_data_from_mesh1_to_mesh2,
    extract_data_from_submesh1_to_submesh2,
)
from utils.helper_function import visualize_result_dict


if __name__ == '__main__':
    case_index = 1

    origin_mesh = r'forward_inverse_3d/data/mesh/mesh_normal_male.msh'
    mesh_file = r'forward_inverse_3d/data/mesh/mesh_normal_male_lc60.msh'
    d_file = f'forward_inverse_3d/data/inverse/{case_index}/u_data_ischemia.npy'
    v_file = f'forward_inverse_3d/data/inverse/{case_index}/v_data_ischemia.npy'

    d = np.load(d_file)[0]
    v = np.load(v_file)[0]

    d = extract_data_from_mesh1_to_mesh2(origin_mesh, mesh_file, d)
    v = extract_data_from_submesh1_to_submesh2(origin_mesh, mesh_file, v)

    result_dict = ischemia_inversion(
        mesh_file=mesh_file,
        d_data=d,
        v_data=v,
        print_message=True,
        transmural_flag=False,
        total_iter=200,
        alpha1=1e-10,
    )

    visualize_result_dict(result_dict)
