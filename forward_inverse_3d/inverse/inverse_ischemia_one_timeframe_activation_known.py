from dolfinx.io import gmshio
from dolfinx.fem import functionspace, form, Function, assemble_scalar
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
from dolfinx.mesh import create_submesh
from ufl import TestFunction, TrialFunction, dot, grad, Measure, derivative, sqrt, inner
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import multiprocessing
from utils.error_metrics_tools import compute_error_phi
from utils.transmembrane_potential_tools import G_tau, delta_tau, delta_deri_tau
from utils.helper_function import find_vertex_with_neighbour_less_than_0
from utils.simulate_tools import build_M, build_Mi
from utils.visualize_tools import (
    plot_loss_and_cm,
    plot_f_on_domain,
    plot_val_on_surface,
)


# if activation zone is known, find ischemia zone
def ischemia_inversion(
    mesh_file,
    d_data,
    v_data,
    phi_1_exact,
    phi_2_exact,
    timeframe,
    gdim=3,
    sigma_i=0.4,
    sigma_e=0.8,
    sigma_t=0.8,
    a1=-90,
    a2=-80,
    a3=10,
    a4=0,
    tau=10,
    alpha1=1e-4,
    phi_initial=None,
    total_iter=300,
    multi_flag=True,
    plot_flag=False,
    print_message=False,
    transmural_flag=False,
):
    # mesh of Body
    domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
    tdim = domain.topology.dim
    # mesh of Heart
    subdomain_ventricle, ventricle_to_torso, _, _ = create_submesh(
        domain, tdim, cell_markers.find(2)
    )

    # function space
    V1 = functionspace(domain, ("Lagrange", 1))
    V2 = functionspace(subdomain_ventricle, ("Lagrange", 1))

    M = build_M(
        domain,
        cell_markers,
        multi_flag=multi_flag,
        condition=None,
        sigma_i=sigma_i,
        sigma_e=sigma_e,
        sigma_t=sigma_t,
    )

    Mi = build_Mi(subdomain_ventricle, condition=None, sigma_i=sigma_i)

    # phi delta_phi delta_deri_phi
    phi_1 = Function(V2)
    phi_2 = Function(V2)
    G_phi_1 = Function(V2)
    G_phi_2 = Function(V2)
    delta_phi_1 = Function(V2)
    delta_phi_2 = Function(V2)
    delta_deri_phi_1 = Function(V2)
    delta_deri_phi_2 = Function(V2)

    # function u v w d
    u = Function(V1)
    w = Function(V1)
    v = Function(V2)
    d = Function(V1)

    # define d's value on the boundary
    d.x.array[:] = d_data[timeframe]

    # matrix A_u
    u1 = TestFunction(V1)
    v1 = TrialFunction(V1)
    dx1 = Measure("dx", domain=domain)
    a_element = dot(grad(v1), dot(M, grad(u1))) * dx1
    bilinear_form_a = form(a_element)
    A_u = assemble_matrix(bilinear_form_a)
    A_u.assemble()
    solver = PETSc.KSP().create()
    solver.setOperators(A_u)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.ILU)

    # vector b_u
    dx2 = Measure("dx", domain=subdomain_ventricle)
    b_u_element = -dot(grad(v1), dot(Mi, grad(v))) * dx2
    entity_map = {domain._cpp_object: ventricle_to_torso}
    linear_form_b_u = form(b_u_element, entity_maps=entity_map)
    b_u = create_vector(linear_form_b_u)

    # scalar c
    ds = Measure('ds', domain=domain)
    c1_element = (d - u) * ds
    c2_element = 1 * ds
    form_c1 = form(c1_element)
    form_c2 = form(c2_element)

    # scalar loss
    loss_element = 0.5 * (u - d) ** 2 * ds
    reg_element = (
        alpha1 * delta_phi_1 * sqrt(inner(grad(phi_1), grad(phi_1)) + 1e-8) * dx2
    )
    form_loss = form(loss_element)
    form_reg = form(reg_element)

    # vector b_w
    b_w_element = u1 * (u - d) * ds
    linear_form_b_w = form(b_w_element)
    b_w = create_vector(linear_form_b_w)

    # vector direction
    u2 = TestFunction(V2)
    j_p = (
        -(a1 - a2 - a3 + a4)
        * delta_phi_1
        * delta_phi_2
        * u2
        * dot(grad(w), dot(Mi, grad(phi_2)))
        * dx2
        - (a1 - a2)
        * delta_deri_phi_1
        * G_phi_2
        * u2
        * dot(grad(w), dot(Mi, grad(phi_1)))
        * dx2
        - (a1 - a2) * delta_phi_1 * G_phi_2 * dot(grad(w), dot(Mi, grad(u2))) * dx2
        - (a3 - a4)
        * delta_deri_phi_1
        * (1 - G_phi_2)
        * u2
        * dot(grad(w), dot(Mi, grad(phi_1)))
        * dx2
        - (a3 - a4)
        * delta_phi_1
        * (1 - G_phi_2)
        * dot(grad(w), dot(Mi, grad(u2)))
        * dx2
    )
    reg_p = alpha1 * derivative(
        delta_phi_1 * sqrt(inner(grad(phi_1), grad(phi_1)) + 1e-8) * dx2, phi_1, u2
    )
    form_J_p = form(j_p, entity_maps=entity_map)
    form_Reg_p = form(reg_p, entity_maps=entity_map)
    J_p = create_vector(form_J_p)
    Reg_p = create_vector(form_Reg_p)

    # fix phi_2
    phi_2.x.array[:] = phi_2_exact[timeframe]
    G_phi_2.x.array[:] = G_tau(phi_2.x.array, tau)
    delta_phi_2.x.array[:] = delta_tau(phi_2.x.array, tau)
    delta_deri_phi_2.x.array[:] = delta_deri_tau(phi_2.x.array, tau)

    # initial phi_1
    if phi_initial is None:
        phi_1.x.array[:] = np.full(phi_1.x.array.shape, tau / 2)
    else:
        phi_1.x.array[:] = phi_initial

    G_phi_1.x.array[:] = G_tau(phi_1.x.array, tau)
    delta_phi_1.x.array[:] = delta_tau(phi_1.x.array, tau)

    # compute v from phi_1 and phi_2
    v.x.array[:] = (
        a1 * G_phi_2.x.array + a3 * (1 - G_phi_2.x.array)
    ) * G_phi_1.x.array + (a2 * G_phi_2.x.array + a4 * (1 - G_phi_2.x.array)) * (
        1 - G_phi_1.x.array
    )

    # get u from p
    with b_u.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b_u, linear_form_b_u)
    solver.solve(b_u, u.vector)
    # adjust u
    adjustment = assemble_scalar(form_c1) / assemble_scalar(form_c2)
    u.x.array[:] = u.x.array + adjustment

    loss_per_iter = []
    cm_cmp_per_iter = []

    k = 0
    while True:
        delta_phi_1.x.array[:] = delta_tau(phi_1.x.array, tau)
        delta_deri_phi_1.x.array[:] = delta_deri_tau(phi_1.x.array, tau)

        # cost function
        loss = assemble_scalar(form_loss) + assemble_scalar(form_reg)
        loss_per_iter.append(loss)
        cm_cmp_per_iter.append(
            compute_error_phi(phi_1.x.array, phi_1_exact[timeframe], V2)
        )

        # get w from u
        with b_w.localForm() as loc_w:
            loc_w.set(0)
        assemble_vector(b_w, linear_form_b_w)
        solver.solve(b_w, w.vector)

        # compute partial derivative of p from w
        with J_p.localForm() as loc_J:
            loc_J.set(0)
        assemble_vector(J_p, form_J_p)
        with Reg_p.localForm() as loc_R:
            loc_R.set(0)
        assemble_vector(Reg_p, form_Reg_p)
        J_p = J_p + Reg_p

        # print information
        if print_message:
            print('iteration:', k)
            print('loss_residual:', assemble_scalar(form_loss))
            print('loss_reg:', assemble_scalar(form_reg))
            print('J_p', np.linalg.norm(J_p.array))
            print('cm_cmp', cm_cmp_per_iter[-1])

        # check if the condition is satisfied
        if k > total_iter or np.linalg.norm(J_p.array) < 1e-2:
            break
        k = k + 1

        # updata p from partial derivative
        dir_p = -J_p.array.copy()
        phi_v = phi_1.x.array[:].copy()

        # origin value
        alpha = 1
        gamma = 0.8
        c = 0.1
        step_search = 0
        while True:
            # adjust p
            phi_1.x.array[:] = phi_v + alpha * dir_p
            # compute u
            G_phi_1.x.array[:] = G_tau(phi_1.x.array, tau)
            v.x.array[:] = (
                a1 * G_phi_2.x.array + a3 * (1 - G_phi_2.x.array)
            ) * G_phi_1.x.array + (
                a2 * G_phi_2.x.array + a4 * (1 - G_phi_2.x.array)
            ) * (
                1 - G_phi_1.x.array
            )
            with b_u.localForm() as loc_b:
                loc_b.set(0)
            assemble_vector(b_u, linear_form_b_u)
            solver.solve(b_u, u.vector)
            # adjust u
            adjustment = assemble_scalar(form_c1) / assemble_scalar(form_c2)
            u.x.array[:] = u.x.array + adjustment

            # compute loss
            loss_new = assemble_scalar(form_loss) + assemble_scalar(form_reg)
            loss_cmp = loss_new - (loss + c * alpha * J_p.array.dot(dir_p))
            alpha = gamma * alpha
            step_search = step_search + 1
            if step_search > 100 or loss_cmp < 0:
                if transmural_flag:
                    # for p < 0, make its neighbor smaller
                    neighbour_idx, _ = find_vertex_with_neighbour_less_than_0(
                        subdomain_ventricle, phi_1
                    )
                    # make them smaller
                    phi_1.x.array[neighbour_idx] = np.where(
                        phi_1.x.array[neighbour_idx] >= 0,
                        phi_1.x.array[neighbour_idx] - tau / total_iter,
                        phi_1.x.array[neighbour_idx],
                    )
                    # compute u
                    G_phi_1.x.array[:] = G_tau(phi_1.x.array, tau)
                    v.x.array[:] = (
                        a1 * G_phi_2.x.array + a3 * (1 - G_phi_2.x.array)
                    ) * G_phi_1.x.array + (
                        a2 * G_phi_2.x.array + a4 * (1 - G_phi_2.x.array)
                    ) * (
                        1 - G_phi_1.x.array
                    )
                    with b_u.localForm() as loc_b:
                        loc_b.set(0)
                    assemble_vector(b_u, linear_form_b_u)
                    solver.solve(b_u, u.vector)
                    # adjust u
                    adjustment = assemble_scalar(form_c1) / assemble_scalar(form_c2)
                    u.x.array[:] = u.x.array + adjustment
                break

    if not plot_flag:
        return phi_1, assemble_scalar(form_loss), assemble_scalar(form_reg)

    marker_result = Function(V2)
    marker_result.x.array[:] = np.where(phi_1.x.array < 0, 1, 0)

    marker_exact = Function(V2)
    marker_exact.x.array[:] = np.where(phi_1_exact[timeframe] < 0, 1, 0)

    v_exact = Function(V2)
    v_exact.x.array[:] = v_data[timeframe]

    p1 = multiprocessing.Process(
        target=plot_f_on_domain,
        kwargs={
            'domain': subdomain_ventricle,
            'f': marker_result,
            'title': 'ischemia_result',
        },
    )
    p2 = multiprocessing.Process(
        target=plot_f_on_domain,
        kwargs={
            'domain': subdomain_ventricle,
            'f': marker_exact,
            'title': 'ischemia_exact',
        },
    )
    p3 = multiprocessing.Process(
        target=plot_f_on_domain,
        kwargs={
            'domain': subdomain_ventricle,
            'f': v_exact,
            'title': 'v_exact',
        },
    )
    p4 = multiprocessing.Process(
        target=plot_f_on_domain,
        kwargs={
            'domain': subdomain_ventricle,
            'f': v,
            'title': 'v_result',
        },
    )
    p5 = multiprocessing.Process(
        target=plot_val_on_surface,
        kwargs={
            'domain': domain,
            'val': d.x.array,
            'function_space': V1,
            'title': 'd_exact',
        },
    )
    p6 = multiprocessing.Process(
        target=plot_val_on_surface,
        kwargs={
            'domain': domain,
            'val': u.x.array,
            'function_space': V1,
            'title': 'd_result',
        },
    )
    p7 = multiprocessing.Process(
        target=plot_loss_and_cm, args=(loss_per_iter, cm_cmp_per_iter)
    )
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()
    p7.start()
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
    p7.join()

    return phi_1, assemble_scalar(form_loss), assemble_scalar(form_reg)
