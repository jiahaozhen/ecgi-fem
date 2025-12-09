import numpy as np
from forward_inverse_3d.inverse.inverse_ischemia_multi_timeframe_activation_known import (
    ischemia_inversion,
)

if __name__ == '__main__':
    mesh_file = r'forward_inverse_3d/data/mesh/mesh_normal_male.msh'
    d_file = r'forward_inverse_3d/data/inverse/u_data_ischemia_30dB.npy'
    v_file = r'forward_inverse_3d/data/inverse/v_data_ischemia.npy'
    phi_1_file = r'forward_inverse_3d/data/inverse/phi_1_data_ischemia.npy'
    phi_2_file = r'forward_inverse_3d/data/inverse/phi_2_data_ischemia.npy'

    d = np.load(d_file)
    v = np.load(v_file)
    phi_1_exact = np.load(phi_1_file)
    phi_2_exact = np.load(phi_2_file)

    time_sequence = np.arange(0, 50, 5)
    ischemia_inversion(
        mesh_file=mesh_file,
        d_data=d,
        v_data=v,
        phi_1_exact=phi_1_exact,
        phi_2_exact=phi_2_exact,
        time_sequence=time_sequence,
        plot_flag=True,
        print_message=True,
        transmural_flag=True,
    )
