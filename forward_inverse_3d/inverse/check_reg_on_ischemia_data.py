from dolfinx.io import gmshio
from dolfinx.fem import functionspace, form, Function, assemble_scalar
from dolfinx.mesh import create_submesh, locate_entities_boundary
import numpy as np
from ufl import grad, Measure, sqrt, inner
from mpi4py import MPI
import matplotlib.pyplot as plt

from utils.transmembrane_potential_tools import delta_tau

gdim = 3
mesh_file = r'forward_inverse_3d/data/mesh/mesh_normal_male.msh'
phi_1_file = r'forward_inverse_3d/data/inverse/phi_1_data_ischemia.npy'
phi_2_file = r'forward_inverse_3d/data/inverse/phi_2_data_ischemia.npy'

# mesh of Body
domain, cell_markers, facet_markers = gmshio.read_from_msh(
    mesh_file, MPI.COMM_WORLD, gdim=gdim
)
tdim = domain.topology.dim
# mesh of Heart
subdomain_ventricle, ventricle_to_torso, _, _ = create_submesh(
    domain, tdim, cell_markers.find(2)
)
sub_node_num = subdomain_ventricle.topology.index_map(0).size_local
sub_domain_boundary = locate_entities_boundary(
    subdomain_ventricle, tdim - 3, lambda x: np.full(x.shape[1], True, dtype=bool)
)

# function space
V = functionspace(subdomain_ventricle, ("Lagrange", 1))
v = Function(V)
phi_1 = Function(V)
phi_2 = Function(V)
delta_phi_1 = Function(V)
delta_phi_2 = Function(V)
alpha1 = 1e-4
alpha2 = 1e-4
dx2 = Measure("dx", domain=subdomain_ventricle)
tau = 1

# load data
phi_1_data = np.load(phi_1_file)
phi_2_data = np.load(phi_2_file)

time_total = phi_1_data.shape[0]
reg_element = (
    alpha1 * delta_phi_1 * sqrt(inner(grad(phi_1), grad(phi_1)) + 1e-8) * dx2
    + alpha2 * delta_phi_2 * sqrt(inner(grad(phi_2), grad(phi_2)) + 1e-8) * dx2
)
reg_time = []
for i in range(time_total):
    phi_1.x.array[:] = phi_1_data[i]
    phi_2.x.array[:] = phi_2_data[i]
    delta_phi_1.x.array[:] = delta_tau(phi_1_data[i], tau)
    delta_phi_2.x.array[:] = delta_tau(phi_2_data[i], tau)
    reg_time.append(assemble_scalar(form(reg_element)))

plt.plot(np.arange(0, time_total / 4, 1 / 4), reg_time)
plt.show()
