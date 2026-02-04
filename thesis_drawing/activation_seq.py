import numpy as np
from utils.simulate_tools import get_activation_dict
from forward_inverse_3d.reaction_diffusion.simulate_reaction_diffusion import (
    compute_v_based_on_reaction_diffusion,
)
from utils.visualize_tools import plot_vals_on_mesh

case_name_list = ['normal_male', 'normal_male2', 'normal_young_male']
case_name = case_name_list[0]
mesh_file = f'forward_inverse_3d/data/mesh/mesh_{case_name}.msh'
T = 500
step_per_timeframe = 4

activation_dict = get_activation_dict(case_name, mode='FREEWALL')

v_val, _, _ = compute_v_based_on_reaction_diffusion(
    mesh_file,
    ischemia_flag=True,
    T=T,
    step_per_timeframe=step_per_timeframe,
    activation_dict_origin=activation_dict,
    center_ischemia=np.array([78.6, 64.1, 9.6]),
    radius_ischemia=20,
    ischemia_epi_endo=[1, 0, -1],
    u_rest_ischemia_val=0.2,
    u_peak_ischemia_val=0.8,
)
plotter = plot_vals_on_mesh(
    mesh_file=mesh_file,
    val_2d=v_val[: 120 * step_per_timeframe],
    target_cell=2,
    f_val_flag=True,
    title=" ",
    show=False,
    off_screen=True,
)

plotter.screenshot("thesis_drawing/figs/activation_seq.png")
