from dolfinx.io import gmshio
from dolfinx.fem import functionspace, form, Function, assemble_scalar
from dolfinx.mesh import create_submesh, locate_entities_boundary
import numpy as np
from ufl import Measure
from mpi4py import MPI
import matplotlib.pyplot as plt
import multiprocessing
from forward_inverse_3d.forward.forward_coupled_ischemia import forward_tmp
from utils.transmembrane_potential_tools import G_tau
from utils.visualize_tools import plot_val_on_surface

# in forward problem, u based on v is recommended
mesh_file = r'forward_inverse_3d/data/mesh/mesh_normal_male.msh'
v_data_file = r'forward_inverse_3d/data/inverse/v_data_ischemia.npy'
d_data_file = r'forward_inverse_3d/data/inverse/u_data_ischemia_30dB.npy'
phi_1_file = r'forward_inverse_3d/data/inverse/phi_1_data_ischemia.npy'
phi_2_file = r'forward_inverse_3d/data/inverse/phi_2_data_ischemia.npy'

total_time = 100
step_per_timeframe = 4

# paramter
a1 = -90  # no active no ischemia
a2 = -80  # no active ischemia
a3 = 10  # active no ischemia
a4 = 0  # active ischemia
tau = 5

# mesh of Body
domain, cell_markers, facet_markers = gmshio.read_from_msh(
    mesh_file, MPI.COMM_WORLD, gdim=3
)
tdim = domain.topology.dim
# mesh of Heart
subdomain_ventricle, ventricle_to_torso, _, _ = create_submesh(
    domain, tdim, cell_markers.find(2)
)
domain_boundary = locate_entities_boundary(
    domain, tdim - 3, lambda x: np.full(x.shape[1], True, dtype=bool)
)

# function space
V1 = functionspace(domain, ("Lagrange", 1))
V2 = functionspace(subdomain_ventricle, ("Lagrange", 1))

# phi_1 phi_2 G_phi delta_phi delta_deri_phi
phi_1 = Function(V2)
phi_2 = Function(V2)
G_phi_1 = Function(V2)
G_phi_2 = Function(V2)

# function u w d
u = Function(V1)
d = Function(V1)
v = Function(V2)
# define d's value on the boundary
u_exact_all_time = np.load(d_data_file)

# scalar c
ds = Measure('ds', domain=domain)
c1_element = (d - u) * ds
c2_element = 1 * ds
form_c1 = form(c1_element)
form_c2 = form(c2_element)

# scalar loss
loss_element = 0.5 * (u - d) ** 2 * ds
form_loss = form(loss_element)

# exact v
v_exact_all_time = np.load(v_data_file)[0 : total_time * step_per_timeframe]
v_result_all_time = np.full_like(v_exact_all_time, 0.0)

# exact phi_1 phi_2
phi_1_exact_all_time = np.load(phi_1_file)[0 : total_time * step_per_timeframe]
phi_2_exact_all_time = np.load(phi_2_file)[0 : total_time * step_per_timeframe]

for i in range(total_time * step_per_timeframe):
    # u based v from phi_1 phi_2
    phi_1.x.array[:] = phi_1_exact_all_time[i]
    phi_2.x.array[:] = phi_2_exact_all_time[i]
    G_phi_1.x.array[:] = G_tau(phi_1.x.array, tau)
    G_phi_2.x.array[:] = G_tau(phi_2.x.array, tau)
    v_result_all_time[i] = (
        a1 * G_phi_2.x.array + a3 * (1 - G_phi_2.x.array)
    ) * G_phi_1.x.array + (a2 * G_phi_2.x.array + a4 * (1 - G_phi_2.x.array)) * (
        1 - G_phi_1.x.array
    )
u_result_all_time, _ = forward_tmp(mesh_file, v_result_all_time)

plt.plot(v_exact_all_time[:, 100])
plt.plot(v_result_all_time[:, 100])
plt.show()

cc = []
loss_list = []
for i in range(total_time * step_per_timeframe):
    u.x.array[:] = u_result_all_time[i]
    d.x.array[:] = u_exact_all_time[i]
    adjustment = assemble_scalar(form_c1) / assemble_scalar(form_c2)
    u_result_all_time[i] = u.x.array + adjustment
    u.x.array[:] = u.x.array + adjustment
    loss = assemble_scalar(form_loss) / assemble_scalar(form_c2)
    loss_list.append(loss)
    cc.append(np.corrcoef(v_result_all_time[i], v_exact_all_time[i])[0, 1])
cc = np.array(cc)
loss_list = np.array(loss_list)
time = np.arange(0, total_time, 1 / step_per_timeframe)
plt.subplot(1, 2, 1)
plt.plot(time, cc)
plt.subplot(1, 2, 2)
plt.plot(time, loss_list)
plt.show()


timeframe = 200

p1 = multiprocessing.Process(
    target=plot_val_on_surface,
    args=(domain, u_result_all_time[timeframe], V1, 'result'),
)
p2 = multiprocessing.Process(
    target=plot_val_on_surface, args=(domain, u_exact_all_time[timeframe], V1, 'exact')
)
p3 = multiprocessing.Process(
    target=plot_val_on_surface,
    args=(
        domain,
        np.abs(u_exact_all_time[timeframe] - u_result_all_time[timeframe]),
        V1,
        'error',
    ),
)
p1.start()
p2.start()
p3.start()
p1.join()
p2.join()
p3.join()
