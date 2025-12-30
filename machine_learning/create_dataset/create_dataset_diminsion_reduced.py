import h5py
import numpy as np
import os
from glob import glob

from utils.ECGDimReducer_tools import ECGReducerFactory


def collect_h5_from_dirs(dirs):
    """
    Collect all h5 files from multiple directories.
    """
    h5_files = []
    for d in dirs:
        files = sorted(glob(os.path.join(d, "*.h5")))
        h5_files.extend(files)

    assert len(h5_files) > 0, "No h5 files found in given dirs."
    return h5_files


def read_all_h5(h5_files, x_key="X", y_key="y"):
    """
    Read and concatenate all h5 data.
    Also record source file id and index within file for each sample.
    """
    X_list = []
    y_list = []

    src_file_ids = []
    src_indices = []

    for file_id, path in enumerate(h5_files):
        with h5py.File(path, "r") as f:
            X = f[x_key][()]  # (B, T, D)
            y = f[y_key][()]

        B = X.shape[0]

        X_list.append(X)
        y_list.append(y)

        # 关键：记录来源
        src_file_ids.append(np.full(B, file_id, dtype=np.int32))
        src_indices.append(np.arange(B, dtype=np.int32))

        print(f"Loaded {path}: {X.shape}")

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    src_file_ids = np.concatenate(src_file_ids, axis=0)
    src_indices = np.concatenate(src_indices, axis=0)

    print(f"Total X shape: {X_all.shape}")

    return X_all, y_all, src_file_ids, src_indices


def compress_from_multiple_dirs(
    input_dirs,
    output_h5,
    reducer,
    x_key="X",
    y_key="y",
):
    h5_files = collect_h5_from_dirs(input_dirs)

    print(f"Found {len(h5_files)} h5 files")

    X_all, y_all, src_file_ids, src_indices = read_all_h5(
        h5_files,
        x_key=x_key,
        y_key=y_key,
    )

    print("Compressing ECG data...")
    X_reduced = reducer.fit_transform(X_all)

    print(f"Reduced X shape: {X_reduced.shape}")

    with h5py.File(output_h5, "w") as f:
        f.create_dataset(
            x_key,
            data=X_reduced,
            compression="gzip",
            compression_opts=4,
        )

        f.create_dataset(y_key, data=y_all)

        f.create_dataset(
            "src_file_id",
            data=src_file_ids,
        )

        f.create_dataset(
            "src_index",
            data=src_indices,
        )

        f.create_dataset(
            "file_names",
            data=np.array(
                [os.path.abspath(p) for p in h5_files],
                dtype="S",
            ),
        )

    print(f"Saved compressed dataset → {output_h5}")


if __name__ == "__main__":

    method_name_list = [
        "flat",
        "flat_pca",
        "lead_pca",
        "temporal_pca",
        "temporal_pooling",
        "temporal_downsample",
        "temporal_lead_pca",
    ]
    method_name = method_name_list[-1]
    reducer = ECGReducerFactory.create(method_name)

    out_put_h5_dir = f"machine_learning/data/Ischemia_Dataset_{method_name}/"

    os.makedirs(out_put_h5_dir, exist_ok=True)

    out_put_h5 = os.path.join(out_put_h5_dir, "data.h5")

    compress_from_multiple_dirs(
        input_dirs=[
            "machine_learning/data/Ischemia_Dataset/normal_male/mild/d64_processed_dataset/",
            "machine_learning/data/Ischemia_Dataset/normal_male/severe/d64_processed_dataset/",
            "machine_learning/data/Ischemia_Dataset/normal_male/healthy/d64_processed_dataset/",
            "machine_learning/data/Ischemia_Dataset/normal_male2/mild/d64_processed_dataset/",
            "machine_learning/data/Ischemia_Dataset/normal_male2/severe/d64_processed_dataset/",
            "machine_learning/data/Ischemia_Dataset/normal_male2/healthy/d64_processed_dataset/",
        ],
        output_h5=out_put_h5,
        reducer=reducer,
        x_key="X",
        y_key="y",
    )
