from dolfinx.io import gmshio
from dolfinx.mesh import create_submesh
from mpi4py import MPI
import numpy as np
from utils.ventricular_segmentation_tools import (
    distinguish_epi_endo,
    lv_17_segmentation_from_mesh,
    get_ischemia_segment,
)
from forward_inverse_3d.reaction_diffusion.simulate_reaction_diffusion import (
    compute_v_based_on_reaction_diffusion,
)
from utils.visualize_tools import plot_val_on_mesh
from utils.simulate_tools import get_activation_dict


case_name = 'normal_male2'
mesh_file = f'forward_inverse_3d/data/mesh/mesh_{case_name}.msh'
gdim = 3

domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
tdim = domain.topology.dim
subdomain_ventricle, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))

segment_ids = lv_17_segmentation_from_mesh(mesh_file, gdim=gdim)
epi_endo_marker = distinguish_epi_endo(mesh_file, gdim=gdim)

# center_ischemia = center_ischemia_list[300]
center_ischemia = np.array([41.23333333, 71.63333333, 5.0])
radius_ischemia = 15
epi_endo_ischemia = [1, 0, -1]

_, marker, _ = compute_v_based_on_reaction_diffusion(
    mesh_file,
    gdim=gdim,
    ischemia_flag=True,
    ischemia_epi_endo=epi_endo_ischemia,
    center_ischemia=center_ischemia,
    radius_ischemia=radius_ischemia,
    T=1,
    activation_dict_origin=get_activation_dict(case_name, mode='FREEWALL'),
)

label = get_ischemia_segment(
    subdomain_ventricle.geometry.x,
    segment_ids,
    epi_endo_marker,
    center_ischemia,
    radius_ischemia,
    epi_endo_ischemia,
)

label_idx = np.where(label == 1)[0]
seg_ids_labeled = np.where(np.isin(segment_ids, label_idx), segment_ids, -1)

import multiprocessing

p1 = multiprocessing.Process(
    target=plot_val_on_mesh,
    args=(mesh_file, marker, gdim, 2, 'marker', 'Ischemia Marker', True),
)
p2 = multiprocessing.Process(
    target=plot_val_on_mesh,
    args=(mesh_file, seg_ids_labeled, gdim, 2, 'segemnt', 'Ischemia Label'),
)
p1.start()
p2.start()

p1.join()
p2.join()
