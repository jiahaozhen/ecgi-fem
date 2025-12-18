from dolfinx.io import gmshio
from dolfinx.mesh import create_submesh
from mpi4py import MPI
import numpy as np
from utils.ventricular_segmentation_tools import (
    distinguish_epi_endo,
    lv_17_segmentation_from_mesh,
    get_ischemia_segment,
)
from utils.visualize_tools import plot_val_on_mesh

case_name = 'normal_male'
mesh_file = f'forward_inverse_3d/data/mesh/mesh_{case_name}.msh'
gdim = 3


domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
tdim = domain.topology.dim
subdomain_ventricle, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))

ischemia_epi_endo_list = [1, 0]
segment_ids = lv_17_segmentation_from_mesh(mesh_file, gdim=gdim)
epi_endo_marker = distinguish_epi_endo(mesh_file, gdim=gdim)

center_ischemia = subdomain_ventricle.geometry.x[100]
radius_ischemia = 30
epi_endo_ischemia = [1, 0, -1]
label = get_ischemia_segment(
    subdomain_ventricle.geometry.x,
    segment_ids,
    epi_endo_marker,
    center_ischemia,
    radius_ischemia,
    epi_endo_ischemia,
)
print("缺血位置分区标签：", label)

import multiprocessing

p1 = multiprocessing.Process(
    target=plot_val_on_mesh, args=(mesh_file, segment_ids, gdim, 2, 'segemnt', 'AHA 17')
)
p1.start()
p1.join()
