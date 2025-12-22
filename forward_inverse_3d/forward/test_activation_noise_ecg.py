from forward_inverse_3d.reaction_diffusion.simulate_reaction_diffusion import (
    compute_v_based_on_reaction_diffusion,
)
from forward_inverse_3d.forward.forward_coupled_matrix_form import compute_d_from_tmp
from utils.visualize_tools import compare_bsp_on_standard12lead
from utils.simulate_tools import get_activation_dict

case_name_list = ['normal_male', 'normal_male2', 'normal_young_male']
case_name = case_name_list[0]
mesh_file = f'forward_inverse_3d/data/mesh/mesh_{case_name}.msh'
T = 500
step_per_timeframe = 8

activation_dict_1 = get_activation_dict(case_name, mode='FREEWALL', add_noise=False)
activation_dict_2 = get_activation_dict(case_name, mode='FREEWALL', add_noise=True)

v_data_1, _, _ = compute_v_based_on_reaction_diffusion(
    mesh_file,
    T=T,
    step_per_timeframe=step_per_timeframe,
    activation_dict_origin=activation_dict_1,
)
v_data_2, _, _ = compute_v_based_on_reaction_diffusion(
    mesh_file,
    T=T,
    step_per_timeframe=step_per_timeframe,
    activation_dict_origin=activation_dict_2,
)

d_data_1 = compute_d_from_tmp(case_name, v_data_1, allow_cache=True)
d_data_2 = compute_d_from_tmp(case_name, v_data_2, allow_cache=True)

import multiprocessing

p1 = multiprocessing.Process(
    target=compare_bsp_on_standard12lead,
    args=(d_data_1, d_data_2),
    kwargs={
        'case_name': case_name,
        'labels': ['Denoise', 'Noise'],
        "step_per_timeframe": step_per_timeframe,
        "filter_flag": False,
    },
)
p1.start()

p1.join()
