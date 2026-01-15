import os
import h5py
import numpy as np


def get_ischemia_ratio(
    case_name,
    severity,
):
    data_dir = [
        f"machine_learning/data/Ischemia_Dataset/{case_name}/{severity}/v_dataset/",
    ]

    markers = []

    # ---------- 收集 ----------
    for d in data_dir:
        assert os.path.isdir(d), f"{d} not found"

        for f in os.listdir(d):
            if f.endswith(".h5"):
                with h5py.File(os.path.join(d, f), "r") as data:
                    v_data = data["X"][:]  # (B, T, D)
                    v_data_0 = v_data[:, 0, :]  # (B, D)
                    marker_ = np.where(v_data_0 == -90, 0, 1)  # (B, D)
                    markers.append(marker_)  # (B, D)

    markers = np.concatenate(markers, axis=0)  # (N, D)
    ischemia_ratios = np.sum(markers, axis=1) / markers.shape[1]  # (N, )
    return np.max(ischemia_ratios), np.min(ischemia_ratios), np.mean(ischemia_ratios)


if __name__ == "__main__":
    case_name_list = ['normal_male', 'normal_male2']
    severity_list = ['mild', 'severe']

    dict_ischemia_ratio = {}

    for case_name in case_name_list:
        for severity in severity_list:
            ischemia_ratio = get_ischemia_ratio(
                case_name,
                severity,
            )
            dict_ischemia_ratio[(case_name, severity)] = ischemia_ratio

    print(dict_ischemia_ratio)
