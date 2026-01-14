import numpy as np
import pyvista
from mpi4py import MPI
from dolfinx.io import gmshio
from dolfinx.mesh import create_submesh
from dolfinx.plot import vtk_mesh

from utils.ventricular_segmentation_tools import lv_17_segmentation_from_mesh
from utils.visualize_tools import visualize_bullseye_segment


def plot_val_on_mesh_custom(
    mesh_file,
    val,
    gdim=3,
    target_cell=None,
    name="f",
    title="Function on Mesh",
):
    domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
    if target_cell is not None:
        cells = cell_markers.find(target_cell)
        subdomain, _, _, _ = create_submesh(domain, domain.topology.dim, cells)
        domain = subdomain

    tdim = domain.topology.dim
    grid = pyvista.UnstructuredGrid(*vtk_mesh(domain, tdim))
    grid.point_data[name] = val
    grid.set_active_scalars(name)

    plotter = pyvista.Plotter(off_screen=True)
    plotter.add_mesh(grid, show_edges=True)
    plotter.add_title(title)

    # 设置固定视角
    plotter.view_yz()
    plotter.camera.azimuth = 180
    plotter.camera.elevation = -45  # 足位视图
    plotter.add_axes()

    # 保存截图
    plotter.screenshot(f"{title.replace(' ', '_')}.png")
    print(f"Screenshot saved as {title.replace(' ', '_')}.png")

    # plotter.show(auto_close=False)


case_name_list = ['normal_male', 'normal_male2', 'normal_young_male']
case_name = case_name_list[1]
mesh_file = f'forward_inverse_3d/data/mesh/mesh_{case_name}.msh'
gdim = 3

seg_ids = lv_17_segmentation_from_mesh(mesh_file, gdim=gdim)
seg_ids += 1

import multiprocessing

p1 = multiprocessing.Process(
    target=visualize_bullseye_segment, kwargs={'values': np.arange(17)}
)
p2 = multiprocessing.Process(
    target=plot_val_on_mesh_custom,
    args=(mesh_file, seg_ids, gdim, 2, 'segemnt', 'AHA 17'),
)

p1.start()
p2.start()

p1.join()
p2.join()
