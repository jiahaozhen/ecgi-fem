from dolfinx.io import gmshio
from dolfinx.mesh import create_submesh
from dolfinx.plot import vtk_mesh
import pyvista
from mpi4py import MPI
from utils.simulate_tools import get_activation_dict
import numpy as np

if __name__ == "__main__":
    case_name_list = ['normal_male', 'normal_male2', 'normal_young_male']
    case_name = case_name_list[0]
    mesh_file = f'forward_inverse_3d/data/mesh/mesh_{case_name}.msh'

    activation_dict = get_activation_dict(case_name, mode='FREEWALL')

    target_pts = np.asarray(list(activation_dict.values()))

    gdim = 3
    target_cell = 2  # ventricle
    domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
    tdim = domain.topology.dim
    subdomain_ventricle, _, _, _ = create_submesh(
        domain, tdim, cell_markers.find(target_cell)
    )
    plotter = pyvista.Plotter(off_screen=False)
    grid = pyvista.UnstructuredGrid(*vtk_mesh(subdomain_ventricle, tdim))
    plotter.add_mesh(grid, show_edges=True)

    plotter.add_points(
        target_pts, color='red', point_size=10, render_points_as_spheres=True
    )

    plotter.view_yz()
    plotter.add_axes()
    print(plotter.camera_position)
    plotter.show()  # for debug purpose
