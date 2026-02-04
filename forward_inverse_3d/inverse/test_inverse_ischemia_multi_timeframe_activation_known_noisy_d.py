import numpy as np
from forward_inverse_3d.inverse.inverse_ischemia_multi_timeframe_activation_known import (
    ischemia_inversion,
)
from utils.error_metrics_tools import compute_error_with_marker
from utils.helper_function import visualize_result_dict

if __name__ == '__main__':
    case_index = 1

    mesh_file = r'forward_inverse_3d/data/mesh/mesh_normal_male.msh'
    v_file = f'forward_inverse_3d/data/inverse/{case_index}/v_data_ischemia.npy'
    phi_1_file = f'forward_inverse_3d/data/inverse/{case_index}/phi_1_data_ischemia.npy'
    phi_2_file = f'forward_inverse_3d/data/inverse/{case_index}/phi_2_data_ischemia.npy'

    v = np.load(v_file)
    phi_1_exact = np.load(phi_1_file)
    phi_2_exact = np.load(phi_2_file)

    # rest inversion
    time_sequence = np.arange(0, 50, 5)

    # activation inversion
    # time_sequence = np.arange(0, 1200, 60)

    for noise_level in [10, 20, 30]:
        print(f'\n=== Inversion with noise level (SNR): {noise_level} dB ===\n')

        d_file_noisy = (
            f'forward_inverse_3d/data/inverse/{case_index}/'
            + f'u_data_ischemia_{noise_level}dB.npy'
        )
        d_noisy = np.load(d_file_noisy)

        result_dict = ischemia_inversion(
            mesh_file=mesh_file,
            d_data=d_noisy,
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
        print(f'Error Metric (Dice Coefficient) for noise level {noise_level} dB')
        print(f'Center Of Mass Error: {error_metric[0]}')
        print(f'Hausdorff Distance: {error_metric[1]}')
        print(f'SN false negative: {error_metric[2]}')
        print(f'SP false positive: {error_metric[3]}')

        # visualize_result_dict(result_dict)
