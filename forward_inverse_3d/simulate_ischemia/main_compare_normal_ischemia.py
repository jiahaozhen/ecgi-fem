# 基于有限元正过程比较正常心脏与缺血心脏的12导联心电图
from forward_inverse_3d.reaction_diffusion.simulate_reaction_diffusion import (
    compute_v_based_on_reaction_diffusion,
)
from forward_inverse_3d.forward.forward_coupled_matrix_form import compute_d_from_tmp
from utils.visualize_tools import plot_val_on_mesh, compare_bsp_on_standard12lead
from utils.simulate_tools import get_activation_dict
import numpy as np

case_name_list = ['normal_male', 'normal_male2', 'normal_young_male']
case_name = case_name_list[1]
mesh_file = f'forward_inverse_3d/data/mesh/mesh_{case_name}.msh'
T = 500
step_per_timeframe = 8

activation_dict = get_activation_dict(case_name, mode='FREEWALL')

v_data_ischemia, _, _ = compute_v_based_on_reaction_diffusion(
    mesh_file,
    ischemia_flag=True,
    T=T,
    step_per_timeframe=step_per_timeframe,
    activation_dict_origin=activation_dict,
    center_ischemia=np.array([8.91, 3.59, -5.34]),
    radius_ischemia=15,
    ischemia_epi_endo=[0, -1],
)
v_data_normal, _, _ = compute_v_based_on_reaction_diffusion(
    mesh_file,
    ischemia_flag=False,
    T=T,
    step_per_timeframe=step_per_timeframe,
    activation_dict_origin=activation_dict,
)

d_data_ischemia = compute_d_from_tmp(case_name, v_data_ischemia, allow_cache=True)
d_data_normal = compute_d_from_tmp(case_name, v_data_normal, allow_cache=True)

import multiprocessing

p1 = multiprocessing.Process(
    target=plot_val_on_mesh,
    args=(mesh_file, v_data_ischemia[0]),
    kwargs={
        "target_cell": 2,
        "name": "v_ischemia",
        "title": "v on ventricle with ischemia",
        "f_val_flag": True,
    },
)
p2 = multiprocessing.Process(
    target=compare_bsp_on_standard12lead,
    args=(d_data_normal, d_data_ischemia),
    kwargs={
        'case_name': case_name,
        'labels': ['Normal', 'Ischemia'],
        "step_per_timeframe": step_per_timeframe,
        "filter_flag": False,
    },
)
p1.start()
p2.start()

p1.join()
p2.join()
