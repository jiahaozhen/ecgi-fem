import numpy as np
from forward_inverse_3d.inverse.inverse_ischemia_multi_timeframe_activation_known import (
    ischemia_inversion,
)
from utils.error_metrics_tools import compute_error_with_marker
from utils.mesh_tools import (
    extract_data_from_mesh1_to_mesh2,
    extract_data_from_submesh1_to_submesh2,
)
from utils.helper_function import visualize_result_dict


if __name__ == '__main__':
    case_index = 1

    origin_mesh = r'forward_inverse_3d/data/mesh/mesh_normal_male.msh'
    d_file = f'forward_inverse_3d/data/inverse/{case_index}/u_data_ischemia.npy'
    v_file = f'forward_inverse_3d/data/inverse/{case_index}/v_data_ischemia.npy'
    phi_1_file = f'forward_inverse_3d/data/inverse/{case_index}/phi_1_data_ischemia.npy'
    phi_2_file = f'forward_inverse_3d/data/inverse/{case_index}/phi_2_data_ischemia.npy'

    d_origin = np.load(d_file)
    v_origin = np.load(v_file)
    phi_1_exact_origin = np.load(phi_1_file)
    phi_2_exact_origin = np.load(phi_2_file)

    for lc in [48, 56, 64]:
        mesh_file = f'forward_inverse_3d/data/mesh/mesh_normal_male_lc{lc}.msh'
        print(f'\n=== Inversion on mesh with lc = {lc} ===\n')

        d = extract_data_from_mesh1_to_mesh2(origin_mesh, mesh_file, d_origin)
        v = extract_data_from_submesh1_to_submesh2(origin_mesh, mesh_file, v_origin)
        phi_1_exact = extract_data_from_submesh1_to_submesh2(
            origin_mesh, mesh_file, phi_1_exact_origin
        )
        phi_2_exact = extract_data_from_submesh1_to_submesh2(
            origin_mesh, mesh_file, phi_2_exact_origin
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
            print_message=False,
            transmural_flag=False,
        )

        error_metric = compute_error_with_marker(
            result_dict['marker_result'],
            result_dict['marker_exact'],
        )
        print(f'Error Metric (Dice Coefficient) for mesh lc = {lc}')
        print(f'Center Of Mass Error: {error_metric[0]}')
        print(f'Hausdorff Distance: {error_metric[1]}')
        print(f'SN false negative: {error_metric[2]}')
        print(f'SP false positive: {error_metric[3]}')

        # visualize_result_dict(result_dict)
