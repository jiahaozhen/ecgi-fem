import numpy as np
from dolfinx.mesh import create_submesh
from dolfinx.fem import functionspace
from dolfinx.io import gmshio
from mpi4py import MPI
from forward_inverse_3d.inverse.inverse_ischemia_multi_timeframe_activation_known import (
    ischemia_inversion,
)
from utils.error_metrics_tools import compute_error_with_marker
from utils.signal_processing_tools import add_noise_based_on_snr
from utils.transmembrane_potential_tools import (
    get_activation_time_from_v,
    compute_phi_with_activation,
)

if __name__ == '__main__':

    case_index = 1

    mesh_file = r'forward_inverse_3d/data/mesh/mesh_normal_male.msh'
    d_file = f'forward_inverse_3d/data/inverse/{case_index}/u_data_ischemia.npy'
    v_file = f'forward_inverse_3d/data/inverse/{case_index}/v_data_ischemia.npy'
    phi_1_file = f'forward_inverse_3d/data/inverse/{case_index}/phi_1_data_ischemia.npy'

    d = np.load(d_file)
    v = np.load(v_file)
    phi_1_exact = np.load(phi_1_file)

    domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=3)
    subdomain_ventricle, _, _, _ = create_submesh(domain, 3, cell_markers.find(2))
    V = functionspace(subdomain_ventricle, ("Lagrange", 1))

    activation_time = get_activation_time_from_v(v)

    # rest inversion
    time_sequence = np.arange(0, 50, 5)

    # activation inversion
    # time_sequence = np.arange(0, 1200, 60)

    for noise_level in [10, 20, 30]:

        print(
            f'\n=== Inversion with noisy activation time, noise level: {noise_level}dB ===\n'
        )

        activation_time_noisy = add_noise_based_on_snr(
            activation_time,
            snr=noise_level,
        )

        activation_time_noisy = np.where(
            activation_time_noisy < 0,
            0,
            activation_time_noisy,
        )

        activation_time_noisy = np.where(
            activation_time_noisy > v.shape[0] - 1,
            v.shape[0] - 1,
            activation_time_noisy,
        )

        activation_time_noisy = np.floor(activation_time_noisy).astype(int)

        phi_2_noisy = compute_phi_with_activation(
            activation_time_noisy,
            v.shape[0],
            V,
        )

        result_dict = ischemia_inversion(
            mesh_file=mesh_file,
            d_data=d,
            v_data=v,
            phi_1_exact=phi_1_exact,
            phi_2_exact=phi_2_noisy,
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
