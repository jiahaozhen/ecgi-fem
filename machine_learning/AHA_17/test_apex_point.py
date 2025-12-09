from dolfinx.io import gmshio
from dolfinx.mesh import create_submesh
from mpi4py import MPI
import numpy as np
from utils.ventricular_segmentation_tools import get_ring_pts, get_apex_from_annulus_pts
from utils.visualize_tools import scatter_on_mesh

case_name_list = ['normal_male', 'normal_male2', 'normal_young_male']
case_name = case_name_list[0]
mesh_file = f'forward_inverse_3d/data/mesh/mesh_{case_name}.msh'
gdim = 3

domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
tdim = domain.topology.dim
subdomain_ventricle, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))

points = subdomain_ventricle.geometry.x
_, _, left_ring_pts, _ = get_ring_pts(mesh_file, gdim=gdim)
apex_point = get_apex_from_annulus_pts(points, left_ring_pts)

scatter_on_mesh(
    mesh_file,
    np.vstack([left_ring_pts, apex_point]),
    target_cell=2,
    title='apex on ventricle',
)
