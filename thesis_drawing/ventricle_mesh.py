from dolfinx.io import gmshio
from dolfinx.mesh import create_submesh
from dolfinx.plot import vtk_mesh
from mpi4py import MPI
import pyvista


def main():
    case_name = "normal_male"
    mesh_file = f'forward_inverse_3d/data/mesh/mesh_{case_name}.msh'
    domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=3)
    cells = cell_markers.find(2)
    tdim = domain.topology.dim
    subdomain, _, _, _ = create_submesh(domain, tdim, cells)
    grid = pyvista.UnstructuredGrid(*vtk_mesh(subdomain, tdim))
    plotter = pyvista.Plotter(off_screen=True)
    plotter.add_mesh(grid, show_edges=True)
    plotter.view_yz()
    plotter.screenshot(
        "thesis_drawing/figs/mesh_ventricle.png", transparent_background=True
    )
    # plotter.show(auto_close=False)


if __name__ == '__main__':
    main()
