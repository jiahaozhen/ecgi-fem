import numpy as np
import os
from forward_inverse_3d.forward.forward_coupled_ischemia import forward_tmp
from forward_inverse_3d.reaction_diffusion.simulate_reaction_diffusion import (
    compute_v_based_on_reaction_diffusion,
)
from utils.signal_processing_tools import add_noise_based_on_snr
from utils.transmembrane_potential_tools import compute_phi_with_v
from utils.simulate_tools import get_activation_dict

case_name_list = ['normal_male', 'normal_male2', 'normal_young_male']
case_name = case_name_list[0]
mesh_file = f'forward_inverse_3d/data/mesh/mesh_{case_name}.msh'

case_dict = {
    '1': {
        'center_ischemia': np.array([78.6, 64.1, 9.6]),
        'radius_ischemia': 20,
    },
    '2': {
        'center_ischemia': np.array([32.1, 71.7, 15]),
        'radius_ischemia': 25,
    },
    '3': {
        'center_ischemia': np.array([8.9, 53.4, -33.5]),
        'radius_ischemia': 25,
    },
    '4': {
        'center_ischemia': np.array([80.4, 19.7, -15.0]),
        'radius_ischemia': 20,
    },
}

ischemia_epi_endo = [-1, 0, 1]
activation_dict = get_activation_dict(case_name, mode='FREEWALL')

for case_id, case_info in case_dict.items():
    v_data, ischemia_marker, V = compute_v_based_on_reaction_diffusion(
        mesh_file,
        ischemia_flag=True,
        ischemia_epi_endo=ischemia_epi_endo,
        center_ischemia=case_info['center_ischemia'],
        radius_ischemia=case_info['radius_ischemia'],
        activation_dict_origin=activation_dict,
    )

    phi_1, phi_2 = compute_phi_with_v(v_data, ischemia_marker, V)

    # sample data
    u_data, _ = forward_tmp(mesh_file, v_data)

    # save data
    file_name_prefix = f'forward_inverse_3d/data/inverse/{case_id}/'
    os.makedirs(os.path.dirname(file_name_prefix), exist_ok=True)
    file_name_suffix = '_data'

    u_file_name = file_name_prefix + 'u' + file_name_suffix + '_ischemia.npy'
    v_file_name = file_name_prefix + 'v' + file_name_suffix + '_ischemia.npy'
    phi_1_file_name = file_name_prefix + 'phi_1' + file_name_suffix + '_ischemia.npy'
    phi_2_file_name = file_name_prefix + 'phi_2' + file_name_suffix + '_ischemia.npy'

    u_data_10dB = add_noise_based_on_snr(u_data, snr=10)
    u_data_20dB = add_noise_based_on_snr(u_data, snr=20)
    u_data_30dB = add_noise_based_on_snr(u_data, snr=30)

    np.save(u_file_name, u_data)
    np.save(v_file_name, v_data)
    np.save(phi_1_file_name, phi_1)
    np.save(phi_2_file_name, phi_2)
    np.save(u_file_name.replace('.npy', '_10dB.npy'), u_data_10dB)
    np.save(u_file_name.replace('.npy', '_20dB.npy'), u_data_20dB)
    np.save(u_file_name.replace('.npy', '_30dB.npy'), u_data_30dB)
