import h5py
import numpy as np

from utils.visualize_tools import plot_triangle_mesh


def main():
    case_name = "normal_male"
    geom_file = f'forward_inverse_3d/data/raw_data/geom_{case_name}.mat'
    geom_data = h5py.File(geom_file, 'r')
    geom_thorax = geom_data['geom_thorax']

    geom_thorax_fac = np.array(geom_thorax['fac'], dtype=np.int32) - 1
    geom_thorax_pts = np.array(geom_thorax['pts'])

    leadelec64_file = f'thesis_drawing/data/geom_{case_name}_leadelec64.mat'
    leadelec_data = h5py.File(leadelec64_file, 'r')
    leadelec = np.array(leadelec_data['leadelec']).T

    p = plot_triangle_mesh(
        geom_thorax_pts,
        geom_thorax_fac,
        title='Nijmegen 64',
        extra_points=leadelec,
        opacity=1.0,
        color="navajowhite",
        point_color="dodgerblue",
        show=False,
        off_screen=True,
        show_edges=False,
    )

    # Front view (Anterior)
    p.view_yz()
    p.render()
    p.screenshot("thesis_drawing/figs/thorax_front.png")

    # Back view (Posterior)
    # Get current camera parameters
    position, focal_point, view_up = p.camera_position

    # Calculate new position: reflect across the focal plane perpendicular to view direction
    # Since view_yz looks along the X axis, we flip the X coordinate relative to focal point
    new_position = (
        focal_point[0] - (position[0] - focal_point[0]),
        position[1],
        position[2],
    )

    # Set new camera position
    p.camera_position = [new_position, focal_point, view_up]
    p.render()
    p.screenshot("thesis_drawing/figs/thorax_back.png")


if __name__ == "__main__":
    main()
