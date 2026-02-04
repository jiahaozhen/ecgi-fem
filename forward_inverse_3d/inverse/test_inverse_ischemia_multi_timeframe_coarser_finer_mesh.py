import numpy as np
from forward_inverse_3d.inverse.inverse_ischemia_multi_timeframe_activation_known import (
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
    phi_1_file = f'forward_inverse_3d/data/inverse/{case_index}/phi_1_data_ischemia.npy'
    phi_2_file = f'forward_inverse_3d/data/inverse/{case_index}/phi_2_data_ischemia.npy'

    d = np.load(d_file)
    v = np.load(v_file)
    phi_1_exact = np.load(phi_1_file)
    phi_2_exact = np.load(phi_2_file)

    d = extract_data_from_mesh1_to_mesh2(origin_mesh, mesh_file, d)
    v = extract_data_from_submesh1_to_submesh2(origin_mesh, mesh_file, v)
    phi_1_exact = extract_data_from_submesh1_to_submesh2(
        origin_mesh, mesh_file, phi_1_exact
    )
    phi_2_exact = extract_data_from_submesh1_to_submesh2(
        origin_mesh, mesh_file, phi_2_exact
    )

    # rest inversion
    time_sequence = np.arange(0, 50, 5)

    # activation inversion
    # time_sequence = np.arange(0, 1200, 60)

    result_dict = ischemia_inversion(
        mesh_file=mesh_file,
        d_data=d,
        v_data=v,
        phi_1_exact=phi_1_exact,
        phi_2_exact=phi_2_exact,
        time_sequence=time_sequence,
        alpha1=1e-2,
        total_iter=100,
        print_message=True,
        transmural_flag=False,
    )

    visualize_result_dict(result_dict)
