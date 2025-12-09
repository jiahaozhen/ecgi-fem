from utils.visualize_tools import scatter_on_mesh
from utils.simulate_tools import get_activation_dict
import numpy as np

if __name__ == "__main__":
    case_name_list = ['normal_male', 'normal_male2', 'normal_young_male']
    case_name = case_name_list[0]
    mesh_file = f'forward_inverse_3d/data/mesh/mesh_{case_name}.msh'

    activation_dict = get_activation_dict(case_name, mode='IVS', threshold=60)

    target_pts = np.asarray(list(activation_dict.values()))

    scatter_on_mesh(mesh_file, target_pts, target_cell=2)
