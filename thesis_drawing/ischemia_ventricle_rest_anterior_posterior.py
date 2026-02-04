from dolfinx.io import gmshio
from dolfinx.mesh import create_submesh
from dolfinx.plot import vtk_mesh
from dolfinx.fem import Function
import pyvista
from mpi4py import MPI
from forward_inverse_3d.reaction_diffusion.simulate_reaction_diffusion import (
    compute_v_based_on_reaction_diffusion,
)
from utils.simulate_tools import get_activation_dict
from utils.function_tools import eval_function
import numpy as np

case_name_list = ['normal_male', 'normal_male2', 'normal_young_male']
case_name = case_name_list[0]
mesh_file = f'forward_inverse_3d/data/mesh/mesh_{case_name}.msh'
T = 1
step_per_timeframe = 1

activation_dict = get_activation_dict(case_name, mode='FREEWALL')

v_val, _, V = compute_v_based_on_reaction_diffusion(
    mesh_file,
    ischemia_flag=True,
    T=T,
    step_per_timeframe=step_per_timeframe,
    activation_dict_origin=activation_dict,
    center_ischemia=np.array([78.6, 64.1, 9.6]),
    radius_ischemia=20,
    ischemia_epi_endo=[1, 0, -1],
    u_rest_ischemia_val=0.1,
    u_peak_ischemia_val=0.9,
)
gdim = 3
target_cell = 2  # ventricle
domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
tdim = domain.topology.dim
subdomain_ventricle, _, _, _ = create_submesh(
    domain, tdim, cell_markers.find(target_cell)
)
v = Function(V)
v.x.array[:] = v_val[0]

plotter = pyvista.Plotter(off_screen=True)
grid = pyvista.UnstructuredGrid(*vtk_mesh(subdomain_ventricle, tdim))
grid.point_data["v"] = eval_function(v, subdomain_ventricle.geometry.x)
grid.set_active_scalars("v")
plotter.add_mesh(grid, show_edges=True, scalars='v', scalar_bar_args={'vertical': True})
plotter.view_yz()
# plotter.remove_scalar_bar()
plotter.screenshot(f'thesis_drawing/figs/ischemia_ventricle_rest_anterior.png')


position, focal_point, view_up = plotter.camera_position

new_position = (
    focal_point[0] - (position[0] - focal_point[0]),
    position[1],
    position[2],
)

plotter.camera_position = [new_position, focal_point, view_up]
plotter.render()
plotter.screenshot(f'thesis_drawing/figs/ischemia_ventricle_rest_posterior.png')
