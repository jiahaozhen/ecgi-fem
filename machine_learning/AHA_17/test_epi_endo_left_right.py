from utils.ventricular_segmentation_tools import distinguish_left_right_endo_epi
from utils.visualize_tools import plot_val_on_mesh

case_name_list = ['normal_male', 'normal_male2', 'normal_young_male']
case_name = case_name_list[2]
mesh_file = f'forward_inverse_3d/data/mesh/mesh_{case_name}.msh'
gdim = 3

marker = distinguish_left_right_endo_epi(mesh_file, gdim=gdim)

print(
    "Epi:",
    len(marker[marker == 1]),
    "Mid:",
    len(marker[marker == 0]),
    "Endo:",
    len(marker[(marker == -1) | (marker == -2)]),
)

plot_val_on_mesh(
    mesh_file,
    marker,
    target_cell=2,
    name="Epi-Endo Marker",
    title="Epi-Endo Marker (1: Epi, 0: Mid, -1: Left Endo, -2: Right Endo)",
)
