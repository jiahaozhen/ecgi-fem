from dolfinx.mesh import locate_entities_boundary, create_submesh
from dolfinx.io import gmshio
from mpi4py import MPI
import numpy as np
from scipy.spatial import cKDTree

from .mesh_tools import submesh_node_index, get_boundary_vertex_connectivity


def distinguish_epi_endo(mesh_file: str, gdim: int) -> np.ndarray:
    """
    Distinguish epi and endo based on the mesh file.

    Parameters:
    - mesh_file: Path to the mesh file.

    Returns:
    - epi_endo_marker: Array with 1 for epi and -1 for endo.
    """
    marker = distinguish_left_right_endo_epi(mesh_file, gdim)
    epi_endo_marker = np.where(marker == -2, -1, marker)
    return epi_endo_marker.astype(np.int32)


def distinguish_left_right_endo_epi(mesh_file: str, gdim: int) -> np.ndarray:
    # mesh of Body
    domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
    tdim = domain.topology.dim
    # mesh of Heart
    subdomain_ventricle, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))
    subdomain_left_cavity, _, _, _ = create_submesh(domain, tdim, cell_markers.find(5))
    subdomain_right_cavity, _, _, _ = create_submesh(domain, tdim, cell_markers.find(6))

    # 1 epi 0 mid -1 left_endo -2 right_endo
    marker = np.zeros(subdomain_ventricle.geometry.x.shape[0], dtype=np.int32)

    ventricle_sub2parent = submesh_node_index(domain, cell_markers, 2)
    ventricle_parent2sub = np.zeros(domain.geometry.x.shape[0], dtype=np.int32) - 1
    ventricle_parent2sub[ventricle_sub2parent] = np.arange(len(ventricle_sub2parent))

    left_cavity_sub2parent = submesh_node_index(domain, cell_markers, 5)
    right_cavity_sub2parent = submesh_node_index(domain, cell_markers, 6)
    ventricle_boundary = locate_entities_boundary(
        subdomain_ventricle, tdim - 3, lambda x: np.full(x.shape[1], True, dtype=bool)
    )
    left_cavity_boundary = locate_entities_boundary(
        subdomain_left_cavity, tdim - 3, lambda x: np.full(x.shape[1], True, dtype=bool)
    )
    right_cavity_boundary = locate_entities_boundary(
        subdomain_right_cavity,
        tdim - 3,
        lambda x: np.full(x.shape[1], True, dtype=bool),
    )

    marker[ventricle_boundary] = 1
    for i in range(len(left_cavity_boundary)):
        node_index_in_ventricle = ventricle_parent2sub[
            left_cavity_sub2parent[left_cavity_boundary[i]]
        ]
        if node_index_in_ventricle != -1:
            marker[node_index_in_ventricle] = -1
    for i in range(len(right_cavity_boundary)):
        node_index_in_ventricle = ventricle_parent2sub[
            right_cavity_sub2parent[right_cavity_boundary[i]]
        ]
        if node_index_in_ventricle != -1:
            marker[node_index_in_ventricle] = -2

    return marker.astype(np.int32)


def get_ring_pts(mesh_file: str, gdim: int) -> tuple[np.ndarray, np.ndarray]:
    domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
    tdim = domain.topology.dim
    subdomain_ventricle, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))

    points = subdomain_ventricle.geometry.x

    marker = distinguish_left_right_endo_epi(mesh_file, gdim=gdim)
    boundary_vertices, adjacency = get_boundary_vertex_connectivity(subdomain_ventricle)

    left_point_index = []
    ring_point_index = []

    for v in boundary_vertices:
        adjacency_list = adjacency[v]
        marker_val = marker[v]
        if marker_val != 1:
            continue
        for v_n in adjacency_list:
            if marker[v_n] == -1:
                left_point_index.append(v_n)
            if marker[v_n] == -2:
                ring_point_index.append(v_n)

    left_points = points[left_point_index]
    ring_points = points[ring_point_index]

    return left_point_index, ring_point_index, left_points, ring_points


def separate_lv_rv(mesh_file, gdim=3):

    _, _, left_ring_pts, right_ring_pts = get_ring_pts(mesh_file, gdim=gdim)

    domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
    tdim = domain.topology.dim

    subdomain_ventricle, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))
    ventricle_pts = subdomain_ventricle.geometry.x

    apex = get_apex_from_annulus_pts(ventricle_pts, left_ring_pts)

    # 5. 分别计算瓣环中心
    left_c = left_ring_pts.mean(axis=0)
    right_c = right_ring_pts.mean(axis=0)

    # 6. 构造分割方向向量：apex -> 左瓣环方向 和 apex -> 右瓣环方向
    vL = left_c - apex
    vR = right_c - apex

    vL /= np.linalg.norm(vL)
    vR /= np.linalg.norm(vR)

    # 7. 对每个 ventricle 点计算 apex->point 的方向向量
    vecs = ventricle_pts - apex
    vecs_norm = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / (vecs_norm + 1e-8)

    # 8. 判断点更靠近左侧还是右侧（根据角度余弦）
    cosL = np.sum(vecs * vL, axis=1)
    cosR = np.sum(vecs * vR, axis=1)

    # 9. 分配标签：cosL > cosR 则为 LV，否则为 RV
    lv_mask = cosL > cosR

    marker = distinguish_left_right_endo_epi(mesh_file, gdim=gdim)

    lv_mask = np.where(marker == -1, True, lv_mask)
    rv_mask = ~lv_mask

    return ventricle_pts[lv_mask], ventricle_pts[rv_mask], lv_mask, rv_mask


def get_apex_from_annulus_pts(vertices, annulus_pts):
    V = np.asarray(vertices)
    annulus = np.asarray(annulus_pts)
    ann_centroid = annulus.mean(axis=0)
    U, S, VT = np.linalg.svd(annulus - ann_centroid, full_matrices=False)
    normal = VT[-1] / np.linalg.norm(VT[-1])
    centroid = V.mean(axis=0)
    if np.dot(normal, centroid - ann_centroid) < 0:
        normal = -normal
    proj = np.dot(V - ann_centroid, normal)
    idx = np.argmax(proj)
    apex = V[idx]

    return apex


def compute_v1_v2(left_ring_pts: np.ndarray, right_ring_pts: np.ndarray, axis):

    v1 = left_ring_pts.mean(axis=0) - right_ring_pts.mean(axis=0)
    # 投影到短轴平面：去掉 axis 分量
    v1 -= np.dot(v1, axis) * axis
    v1 /= np.linalg.norm(v1)

    # ---- v2: 与 v1 正交，通过叉乘得到 ----
    v2 = np.cross(axis, v1)
    v2 /= np.linalg.norm(v2)

    return v1, v2


def compute_h_and_theta(points, center, axis, v1, v2):
    """
    points: Nx3 点列
    center: annulus_center
    axis:   LV 主轴(unit vector)
    v1,v2:  短轴平面坐标系基向量（来自原方法）
    """
    vecs = points - center  # N×3

    # ---- 高度 h ----
    # 投影到主轴 axis
    h = np.dot(vecs, axis)  # N,
    h_norm = (h - h.min()) / (h.max() - h.min())

    # ---- 角度 θ ----
    x = np.dot(vecs, v1)  # v 在 v1 方向的投影
    y = np.dot(vecs, v2)  # v 在 v2 方向的投影
    theta = np.degrees(np.arctan2(y, x))

    return h_norm, theta


def compute_lv_axis(annulus_points, apex_point):
    annulus_center = annulus_points.mean(axis=0)
    axis = apex_point - annulus_center
    axis /= np.linalg.norm(axis)
    return annulus_center, axis


def assign_segment(hi, theta_deg):
    if hi < 0.33:
        ring = 'basal'
        n_seg, offset = 6, 0
    elif hi < 0.66:
        ring = 'mid'
        n_seg, offset = 6, 6
    elif hi < 0.9:
        ring = 'apical'
        n_seg, offset = 4, 12
    else:
        return 16  # segment 17 (apex cap)

    theta_offset = 60 if hi < 0.66 else 45

    theta_deg = (theta_deg + 360 - theta_offset) % 360
    seg = int(theta_deg / (360 / n_seg)) + offset
    return seg


def lv_17_segmentation(lv_points, left_ring_pts, right_ring_pts, apex_point):
    annulus_center, axis = compute_lv_axis(left_ring_pts, apex_point)

    v1, v2 = compute_v1_v2(left_ring_pts, right_ring_pts, axis)

    h_norm, theta_list = compute_h_and_theta(lv_points, annulus_center, axis, v1, v2)

    seg_ids = []

    for h, theta in zip(h_norm, theta_list):
        seg_ids.append(assign_segment(h, theta))
    return np.array(seg_ids)


def lv_17_segmentation_from_mesh(mesh_file: str, gdim: int = 3) -> np.ndarray:
    """
    Perform 17-segment segmentation of the left ventricle based on the mesh file.

    Parameters:
    - mesh_file: Path to the mesh file.
    - gdim: Geometric dimension of the mesh (default is 3).

    Returns:
    - segment_ids: Array of segment IDs for each vertex in the left ventricle.
    """
    # Load mesh and extract ventricle submesh
    domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
    tdim = domain.topology.dim
    subdomain_ventricle, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))

    points = subdomain_ventricle.geometry.x
    _, _, left_ring_pts, right_right_pts = get_ring_pts(mesh_file, gdim=gdim)

    lv_points, _, lv_mask, rv_mask = separate_lv_rv(mesh_file, gdim=gdim)

    apex_point = get_apex_from_annulus_pts(lv_points, left_ring_pts)

    segment_ids_lv = lv_17_segmentation(
        lv_points, left_ring_pts, right_right_pts, apex_point
    )

    segment_ids = np.zeros(points.shape[0], dtype=np.int32)
    segment_ids[lv_mask] = segment_ids_lv
    segment_ids[rv_mask] = -1  # RV points marked as -1

    return segment_ids


def get_IVS_region(mesh_file, gdim=3, threshold=2.0):
    """
    基于 LV/RV 内膜距离差提取室间隔
    threshold：决定多接近才属于室间隔（mm）
    """
    domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
    tdim = domain.topology.dim

    subdomain, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))
    ventricle_pts = subdomain.geometry.x

    marker = distinguish_left_right_endo_epi(mesh_file, gdim=gdim)
    left_endo_pts = ventricle_pts[marker == -1]
    right_endo_pts = ventricle_pts[marker == -2]

    tree_LV = cKDTree(left_endo_pts)
    tree_RV = cKDTree(right_endo_pts)

    d_LV, _ = tree_LV.query(ventricle_pts)
    d_RV, _ = tree_RV.query(ventricle_pts)

    # 同时距离不应太大（避免后壁被分类进去）
    ivs_mask = (d_LV < threshold) & (d_RV < threshold)

    ivs_mask = np.where(marker == 1, False, ivs_mask)

    return ivs_mask, ventricle_pts[ivs_mask], d_LV, d_RV


def get_free_wall_region(
    mesh_file, gdim, theta_min=-60, theta_max=60, h_min=0, h_max=1
):
    domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
    tdim = domain.topology.dim
    subdomain_ventricle, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))

    vertices_pts = subdomain_ventricle.geometry.x

    _, _, left_ring_points, right_ring_points = get_ring_pts(mesh_file, gdim=gdim)

    lv_points, rv_points, lv_mask, rv_mask = separate_lv_rv(mesh_file, gdim=gdim)

    left_ring_points = np.asarray(left_ring_points)
    right_ring_points = np.asarray(right_ring_points)

    left_center = left_ring_points.mean(axis=0)
    right_center = right_ring_points.mean(axis=0)

    U, S, VT = np.linalg.svd(left_ring_points - left_center, full_matrices=False)
    axis = VT[-1] / np.linalg.norm(VT[-1])

    ventricle_center = vertices_pts.mean(axis=0)
    if np.dot(axis, ventricle_center - left_center) < 0:
        axis = -axis

    v1, v2 = compute_v1_v2(left_ring_points, right_ring_points, axis)

    left_h, left_theta = compute_h_and_theta(lv_points, left_center, axis, -v1, v2)

    right_h, right_theta = compute_h_and_theta(rv_points, right_center, axis, v1, v2)

    epi_endo_marker = distinguish_epi_endo(mesh_file, gdim)

    # LV 自由壁区域
    lv_region_mask = (
        (left_theta >= theta_min)
        & (left_theta <= theta_max)
        & (left_h >= h_min)
        & (left_h <= h_max)
    )

    # 去掉 LV 外膜点
    lv_region_mask &= epi_endo_marker[lv_mask] != 1  # 若 0=endo / 1=epi

    lv_freewall_pts = lv_points[lv_region_mask]

    # ---------- RV 自由壁筛选 ----------
    rv_region_mask = (
        (right_theta >= theta_min)
        & (right_theta <= theta_max)
        & (right_h >= h_min)
        & (right_h <= h_max)
    )

    # 去掉 RV 外膜点
    rv_region_mask &= epi_endo_marker[rv_mask] != 1

    rv_freewall_pts = rv_points[rv_region_mask]

    # 合并
    free_wall_pts = np.vstack([lv_freewall_pts, rv_freewall_pts])

    return free_wall_pts


def get_ischemia_segment(
    ventricle_pts,
    seg_ids,
    epi_endo_marker,
    center_ischemia,
    radius_ischemia,
    epi_endo_ischemia,
    ratio_threshold=0.1,
):
    allowed_layers = np.asarray(epi_endo_ischemia)
    layer_mask = np.isin(epi_endo_marker, allowed_layers)

    valid_mask = layer_mask & (seg_ids != -1)

    lv_pts_layer = ventricle_pts[valid_mask]
    seg_ids_layer = seg_ids[valid_mask]

    n_seg = 17
    label = np.zeros(n_seg, dtype=np.int64)

    if len(lv_pts_layer) == 0:
        return label

    # ---------- 2. 空间缺血区域 ----------
    tree = cKDTree(lv_pts_layer)
    indices = tree.query_ball_point(center_ischemia, radius_ischemia)

    if len(indices) == 0:
        return label

    seg_ids_in_region = seg_ids_layer[indices]

    # ---------- 3. segment 内比例判断 ----------
    for seg in np.unique(seg_ids_in_region):
        total_pts = np.sum(seg_ids_layer == seg)
        ischemia_pts = np.sum(seg_ids_in_region == seg)

        if ischemia_pts / total_pts > ratio_threshold:
            label[int(seg)] = 1

    return label
