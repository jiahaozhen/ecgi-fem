import numpy as np
from utils.ventricular_segmentation_tools import lv_17_segmentation_from_mesh
from utils.visualize_tools import visualize_bullseye_segment, plot_val_on_mesh

case_name_list = ['normal_male', 'normal_male2', 'normal_young_male']
case_name = case_name_list[0]
mesh_file = f'forward_inverse_3d/data/mesh/mesh_{case_name}.msh'
gdim = 3

seg_ids = lv_17_segmentation_from_mesh(mesh_file, gdim=gdim)
seg_ids += 1

import multiprocessing

p1 = multiprocessing.Process(
    target=visualize_bullseye_segment, kwargs={'values': np.arange(17)}
)
p2 = multiprocessing.Process(
    target=plot_val_on_mesh, args=(mesh_file, seg_ids, gdim, 2, 'segemnt', 'AHA 17')
)

p1.start()
p2.start()

p1.join()
p2.join()
