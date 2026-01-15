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
_, rpeaks = nk.ecg_peaks(cleaned_ecg, sampling_rate=sampling_rate, method="elgendi2010")
print(f"R-peaks found: {len(rpeaks['ECG_R_Peaks'])}")


_, waves_peaks = nk.ecg_delineate(
    cleaned_ecg, rpeaks, sampling_rate=sampling_rate, method="cwt"
)

# 4. 提取 ST 段和 T 波的信息
# 我们将结果整理成 DataFrame 以便观察
results_dict = {"R_Peak": rpeaks["ECG_R_Peaks"]}

# Add other waves if they exist and align in length
for key in ["ECG_T_Peaks", "ECG_T_Onsets", "ECG_T_Offsets"]:
    if key in waves_peaks:
        # Check if list length matches R-peaks or pad/truncate (usually delineate aligns them or uses NaNs)
        feat_len = len(waves_peaks[key])
        target_len = len(results_dict['R_Peak'])

        if feat_len == target_len:
            results_dict[key] = waves_peaks[key]
        else:
            # If shorter, pad with NaNs
            if feat_len < target_len:
                padded = list(waves_peaks[key]) + [np.nan] * (target_len - feat_len)
                results_dict[key] = padded
            # If longer, truncate
            else:
                results_dict[key] = waves_peaks[key][:target_len]

results = pd.DataFrame(results_dict)

print("--- 提取的前 5 个周期的特征点索引 ---")
print(results.head())

# 5. 可视化
plt.figure(figsize=(12, 6))

# 绘制一段心电图 (取前2秒展示或全部，视数据长度而定)
# Limit plot to avoid overcrowding
plot_len = min(len(cleaned_ecg), 1000)
plot_range = range(0, plot_len)

plt.plot(
    np.array(cleaned_ecg)[plot_range], color='black', alpha=0.7, label='Cleaned ECG'
)


# Helper to plot points
def plot_points(indices, color, marker, label):
    valid_points = [p for p in indices if not np.isnan(p) and p < plot_len]
    if valid_points:
        plt.scatter(
            valid_points,
            np.array(cleaned_ecg)[np.array(valid_points, dtype=int)],
            color=color,
            marker=marker,
            label=label,
            zorder=5,
        )


# Plot R peaks
plot_points(rpeaks["ECG_R_Peaks"], 'red', 'o', 'R Peak')

# Plot T peaks
if "ECG_T_Peaks" in waves_peaks:
    plot_points(waves_peaks["ECG_T_Peaks"], 'blue', 'v', 'T Peak')

# Plot T Waves intervals
if "ECG_T_Onsets" in waves_peaks and "ECG_T_Offsets" in waves_peaks:
    onsets = waves_peaks["ECG_T_Onsets"]
    offsets = waves_peaks["ECG_T_Offsets"]
    for on, off in zip(onsets, offsets):
        if np.isnan(on) or np.isnan(off):
            continue
        if on >= plot_len:
            continue
        off_vis = min(off, plot_len - 1)
        plt.axvspan(on, off_vis, color='orange', alpha=0.3)

# 避免图例重复显示
handles, labels_plot = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels_plot, handles))
plt.legend(by_label.values(), by_label.keys())

plt.title("ECG Feature Extraction")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.show()
