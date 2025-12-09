from utils.ventricular_segmentation_tools import get_IVS_region
from utils.visualize_tools import scatter_on_mesh

case_name_list = ['normal_male', 'normal_male2', 'normal_young_male']
case_name = case_name_list[0]
mesh_file = f'forward_inverse_3d/data/mesh/mesh_{case_name}.msh'
gdim = 3

_, ivs_points, _, _ = get_IVS_region(mesh_file, gdim=gdim, threshold=18.0)

scatter_on_mesh(mesh_file, ivs_points, target_cell=2, title='IVS pts')
