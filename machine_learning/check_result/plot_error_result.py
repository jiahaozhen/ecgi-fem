import h5py
from pathlib import Path
import numpy as np
from utils.visualize_tools import plot_val_on_mesh, visualize_bullseye_segment
from utils.ventricular_segmentation_tools import lv_17_segmentation_from_mesh


def load_wrong_samples(h5_path):
    """
    对应 save_wrong_sample
    return: list[dict]
    """

    samples = []

    with h5py.File(h5_path, "r") as f:
        y_true = f["y_true"][:]
        y_pred = f["y_pred"][:]
        sample_idx = f["sample_idx"][:]
        file = f['file'][:]

        n = len(y_true)

        for i in range(n):
            samples.append(
                {
                    "y_true": y_true[i],
                    "y_pred": y_pred[i],
                    "sample_idx": int(sample_idx[i]),
                    "file": file[i],
                }
            )

    return samples


if __name__ == '__main__':

    dataset_type = "cnn_features"
    # dataset_type = "features"
    # dataset_type = "processed"
    # dataset_type = "cnn128"

    sample_path = f"machine_learning/data/error_samples/{dataset_type}/ml/multilabel_xgb_classifier.h5"
    samples = load_wrong_samples(sample_path)
    sample = samples[0]

    raw_file = sample["file"]
    if isinstance(raw_file, bytes):
        file_path = raw_file.decode("utf-8")
    else:
        file_path = str(raw_file)

    p = Path(file_path)
    parts = list(p.parts)
    s1 = parts[-1]
    s1_parts = s1.split("_", 1)
    new_s1 = "v_" + s1_parts[1]
    parts[-1] = new_s1
    parts[-2] = "v_dataset"
    new_path = Path(*parts)

    file_path = str(new_path)
    sample_idx = sample["sample_idx"]

    with h5py.File(file_path, "r") as f:
        v_data = f["X"][sample_idx]
        center = f["center"][sample_idx]
        radius = f["radius"][sample_idx]
        y = f["y"][sample_idx]

    y_true = sample["y_true"]
    y_pred = sample["y_pred"]

    assert (y_true == y).all()

    case_name = parts[-4]
    mesh_file = f'forward_inverse_3d/data/mesh/mesh_{case_name}.msh'

    seg_ids = lv_17_segmentation_from_mesh(mesh_file, gdim=3)

    idx_true = np.where(y_true == 1)[0]
    seg_ids_true = np.where(np.isin(seg_ids, idx_true), seg_ids, -1)

    idx_pred = np.where(y_pred == 1)[0]
    seg_ids_pred = np.where(np.isin(seg_ids, idx_pred), seg_ids, -1)

    print(mesh_file)
    print(f"y true {y_true}")
    print(f"y pred {y_pred}")
    print(f"center {center}")
    print(f"radius {radius}")

    import multiprocessing

    p1 = multiprocessing.Process(
        target=plot_val_on_mesh, args=(mesh_file, v_data[0], 3, 2, "v", "V Data", True)
    )

    p2 = multiprocessing.Process(
        target=plot_val_on_mesh,
        args=(mesh_file, seg_ids_true, 3, 2, 'segment', 'True Ischemia'),
    )

    p3 = multiprocessing.Process(
        target=plot_val_on_mesh,
        args=(mesh_file, seg_ids_pred, 3, 2, 'segment', 'Predicted Ischemia'),
    )

    p4 = multiprocessing.Process(
        target=visualize_bullseye_segment,
        args=(y_true, 'True Ischemia Bullseye'),
    )

    p5 = multiprocessing.Process(
        target=visualize_bullseye_segment,
        args=(y_pred, 'Predicted Ischemia Bullseye'),
    )

    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
