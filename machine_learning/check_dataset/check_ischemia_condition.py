from utils.visualize_tools import plot_val_on_mesh
from forward_inverse_3d.reaction_diffusion.simulate_reaction_diffusion import (
    compute_v_based_on_reaction_diffusion,
)
from utils.simulate_tools import get_activation_dict
import numpy as np

case_name_list = ['normal_male', 'normal_male2', 'normal_young_male']
case_name = case_name_list[1]
mesh_file = f'forward_inverse_3d/data/mesh/mesh_{case_name}.msh'

ischemia_epi_endo = [0, -1]
# center_ischemia = np.array([8.91, 3.59, -5.34])
center_ischemia = np.array([5.5, 4.5, -3.4])
radius_ischemia = 15
activation_dict = get_activation_dict(case_name, mode='FREEWALL')

v, ischemia_marker, V = compute_v_based_on_reaction_diffusion(
    mesh_file,
    gdim=3,
    ischemia_flag=True,
    ischemia_epi_endo=ischemia_epi_endo,
    center_ischemia=center_ischemia,
    radius_ischemia=radius_ischemia,
    T=1,
    activation_dict_origin=activation_dict,
)

plot_val_on_mesh(mesh_file, ischemia_marker, target_cell=2, f_val_flag=True)
