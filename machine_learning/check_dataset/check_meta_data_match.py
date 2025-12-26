import h5py
import numpy as np
from dolfinx.io import gmshio
from dolfinx.mesh import create_submesh
from mpi4py import MPI
from forward_inverse_3d.reaction_diffusion.simulate_reaction_diffusion import (
    compute_v_based_on_reaction_diffusion,
)
from utils.ventricular_segmentation_tools import (
    distinguish_epi_endo,
    lv_17_segmentation_from_mesh,
)
from utils.visualize_tools import plot_val_on_mesh
from utils.simulate_tools import get_activation_dict


if __name__ == "__main__":
    case_name_list = ['normal_male', 'normal_male2', 'normal_young_male']
    case_name = case_name_list[0]

    mesh_file = f'forward_inverse_3d/data/mesh/mesh_{case_name}.msh'
    severity = 'mild'
    gdim = 3

    v_data_file = f"machine_learning/data/Ischemia_Dataset/{case_name}/{severity}/v_dataset/v_part_000.h5"

    domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
    tdim = domain.topology.dim
    subdomain_ventricle, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))

    segment_ids = lv_17_segmentation_from_mesh(mesh_file, gdim=gdim)
    epi_endo_marker = distinguish_epi_endo(mesh_file, gdim=gdim)

    sample_idx = 100

    with h5py.File(v_data_file, "r") as data:
        v_data = data["X"][sample_idx]
        label = data["y"][sample_idx]
        center = data["center"][sample_idx]
        radius = data["radius"][sample_idx]
        rest = data["rest"][sample_idx]
        epi_endo = data["epi_endo"][sample_idx]

    label_idx = np.where(label == 1)[0]
    seg_ids_labeled = np.where(np.isin(segment_ids, label_idx), segment_ids, -1)

    if isinstance(epi_endo, bytes):
        epi_endo = epi_endo.decode("utf-8")
    else:
        epi_endo = str(epi_endo)

    epi_endo_ischemia = []
    if epi_endo[0] == '1':
        epi_endo_ischemia.append(1)
    if epi_endo[1] == '1':
        epi_endo_ischemia.append(0)
    if epi_endo[2] == '1':
        epi_endo_ischemia.append(-1)

    _, marker, _ = compute_v_based_on_reaction_diffusion(
        mesh_file,
        gdim=gdim,
        ischemia_flag=True,
        ischemia_epi_endo=epi_endo_ischemia,
        center_ischemia=center,
        radius_ischemia=radius,
        T=1,
        activation_dict_origin=get_activation_dict(case_name, mode='FREEWALL'),
    )

    import multiprocessing

    p1 = multiprocessing.Process(
        target=plot_val_on_mesh,
        args=(mesh_file, marker, gdim, 2, 'marker', 'Ischemia Marker', True),
    )
    p2 = multiprocessing.Process(
        target=plot_val_on_mesh,
        args=(mesh_file, seg_ids_labeled, gdim, 2, 'segemnt', 'Ischemia Label'),
    )
    p3 = multiprocessing.Process(
        target=plot_val_on_mesh,
        args=(mesh_file, v_data[0], gdim, 2, 'v', 'V Data', True),
    )
    p1.start()
    p2.start()
    p3.start()
    p1.join()
    p2.join()
    p3.join()
