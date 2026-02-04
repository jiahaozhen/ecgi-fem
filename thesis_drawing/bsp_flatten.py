from forward_inverse_3d.reaction_diffusion.simulate_reaction_diffusion import (
    compute_v_based_on_reaction_diffusion,
)
from forward_inverse_3d.forward.forward_coupled_matrix_form import compute_d_from_tmp
from utils.simulate_tools import get_activation_dict
from utils.signal_processing_tools import (
    project_bsp_on_surface,
    transfer_bsp_to_standard300lead,
)
import numpy as np

case_name_list = ['normal_male', 'normal_male2']
case_name = case_name_list[0]
mesh_file = f'forward_inverse_3d/data/mesh/mesh_{case_name}.msh'
T = 500
step_per_timeframe = 1

activation_dict = get_activation_dict(case_name, mode='FREEWALL')

v_data_ischemia, _, _ = compute_v_based_on_reaction_diffusion(
    mesh_file,
    ischemia_flag=True,
    T=T,
    step_per_timeframe=step_per_timeframe,
    activation_dict_origin=activation_dict,
    center_ischemia=np.array([8.91, 3.59, -5.34]),
    radius_ischemia=20,
    ischemia_epi_endo=[0, -1],
    u_peak_ischemia_val=0.9,
    u_rest_ischemia_val=0.1,
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

d_data_normal = transfer_bsp_to_standard300lead(d_data_normal)
d_data_ischemia = transfer_bsp_to_standard300lead(d_data_ischemia)

timeframe_idx = 150
surface_d_normal = project_bsp_on_surface(
    d_data_normal,
    length=100,
    width=50,
)[timeframe_idx]
surface_d_ischemia = project_bsp_on_surface(
    d_data_ischemia,
    length=100,
    width=50,
)[timeframe_idx]

# plot heatmaps
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
plt.subplot(2, 1, 1)
plt.title('Normal Surface D')
plt.imshow(
    surface_d_normal,
    cmap='viridis',
    aspect='auto',
)
plt.colorbar()
plt.subplot(2, 1, 2)
plt.title('Ischemia Surface D')
plt.imshow(
    surface_d_ischemia,
    cmap='viridis',
    aspect='auto',
)
plt.colorbar()
plt.tight_layout()
plt.show()
