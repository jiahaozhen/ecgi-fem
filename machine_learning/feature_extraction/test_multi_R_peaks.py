import neurokit2 as nk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import h5py

file_path = 'machine_learning/data/Ischemia_Dataset/normal_male/healthy/d64_noisy_dataset/d_part_000.h5'

with h5py.File(file_path, 'r') as f:
    batch_data = f['X'][:]

print(f"Loaded data shape: {batch_data.shape}")

sampling_rate = 1000
signal = batch_data[0, :, 2]

signal = np.tile(signal, 10)

# 2. 信号预处理 (去噪)
cleaned_ecg = nk.ecg_clean(signal, sampling_rate=sampling_rate, method="neurokit")

# Debug: Check signal range
print(f"Cleaned ECG range: {np.min(cleaned_ecg):.4f} to {np.max(cleaned_ecg):.4f}")

# Find peaks
_, rpeaks = nk.ecg_peaks(cleaned_ecg, sampling_rate=sampling_rate)
rpeaks_list = rpeaks['ECG_R_Peaks']
rpeaks_list = np.insert(rpeaks_list, 0, rpeaks_list[0] - len(batch_data[0, :, 2]))
print(f"Adjusted R-peaks: {rpeaks_list}")
_, waves_peaks = nk.ecg_delineate(cleaned_ecg, rpeaks, sampling_rate=sampling_rate)
results_dict = {"R_Peak": rpeaks_list}
for key in ["ECG_T_Peaks", "ECG_T_Onsets", "ECG_T_Offsets"]:
    results_dict[key] = waves_peaks[key]
results = pd.DataFrame(results_dict)
print("--- 提取的前 5 个周期的特征点索引 ---")
print(results.head())
