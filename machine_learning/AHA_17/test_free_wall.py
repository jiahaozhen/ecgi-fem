from utils.ventricular_segmentation_tools import get_free_wall_region
from utils.visualize_tools import scatter_on_mesh

case_name_list = ['normal_male', 'normal_male2', 'normal_young_male']
case_name = case_name_list[0]
mesh_file = f'forward_inverse_3d/data/mesh/mesh_{case_name}.msh'
gdim = 3

free_wall_pts = get_free_wall_region(
    mesh_file, gdim, theta_min=-60, theta_max=60, h_min=0.5, h_max=1
)

scatter_on_mesh(mesh_file, free_wall_pts, target_cell=2, title='free wall pts')
