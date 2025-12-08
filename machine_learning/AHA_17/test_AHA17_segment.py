from dolfinx.io import gmshio
from dolfinx.mesh import create_submesh
from dolfinx.plot import vtk_mesh
from mpi4py import MPI
import pyvista
import numpy as np
from utils.ventricular_segmentation_tools import lv_17_segmentation_from_mesh
from utils.visualize_tools import visualize_bullseye_segment

case_name_list = ['normal_male', 'normal_male2', 'normal_young_male']
case_name = case_name_list[0]
mesh_file = f'forward_inverse_3d/data/mesh/mesh_{case_name}.msh'
gdim = 3

seg_ids = lv_17_segmentation_from_mesh(mesh_file, gdim=gdim)

domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
tdim = domain.topology.dim
subdomain_ventricle, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))

visualize_bullseye_segment(np.arange(17))

plotter = pyvista.Plotter()
grid = pyvista.UnstructuredGrid(*vtk_mesh(subdomain_ventricle, tdim))
grid.point_data["f"] = seg_ids
grid.set_active_scalars("f")

plotter.add_mesh(grid, show_edges=True)
plotter.view_yz()
plotter.add_axes()
plotter.show()
