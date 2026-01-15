import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

import h5py


def extract_ecg_features(signal, fs=1000):
    """
    针对单次心跳片段提取形态学特征 (考虑信号绝对值进行波形定位)
    :param signal: 单次心拍的幅值数组
    :param fs: 采样率 (Hz)
    :return: 包含特征的字典
    """
    features = {}
    n = len(signal)

    # --- 1. 定位 R 波 (片段内的绝对最大值) ---
    r_idx = np.argmax(np.abs(signal))
    features['R_amplitude'] = signal[r_idx]
    # Return time instead of index
    features['R_time'] = r_idx / fs * 1000  # ms

    # 判断 R 波的主方向 (正向或负向)
    # 如果 R 波为正，则 Q/S 为波谷 (min)；如果 R 波为负，则 Q/S 为波峰 (max)
    if features['R_amplitude'] >= 0:
        find_extremum = np.argmin
    else:
        find_extremum = np.argmax

    # --- 2. 定位 Q 波与 S 波 ---
    # 搜索 R 前 50ms 寻找 Q
    q_window = signal[max(0, r_idx - int(0.05 * fs)) : r_idx]
    if len(q_window) > 0:
        q_local_idx = find_extremum(q_window)
        q_idx = r_idx - len(q_window) + q_local_idx
        features['Q_amplitude'] = signal[q_idx]
        features['Q_time'] = q_idx / fs * 1000  # ms

    # 搜索 R 后 50ms 寻找 S
    s_window = signal[r_idx : min(n, r_idx + int(0.05 * fs))]
    if len(s_window) > 0:
        s_local_idx = find_extremum(s_window)
        s_idx = r_idx + s_local_idx
        features['S_amplitude'] = signal[s_idx]
        features['S_time'] = s_idx / fs * 1000  # ms

    # --- 3. 定位 T 波 (R波之后 100ms 到 450ms 之间的最大值) ---
    # T 波可能与 R 波同向或反向，这里使用绝对值最大来定位
    dt = 1000.0 / fs  # ms per sample (for feature calculation)
    t_start = r_idx + int(0.2 * fs)
    t_end = min(n, r_idx + int(0.45 * fs))

    t_idx = t_amp = t_latency = t_width = t_area = t_sign = np.nan

    if t_start < n:
        t_window = signal[t_start:t_end]
        if len(t_window) > 0:
            t_local_idx = np.argmax(np.abs(t_window))
            t_idx = t_start + t_local_idx

            t_amp = signal[t_idx]
            t_latency = t_idx * dt
            t_sign = np.sign(t_amp)

            features['T_amplitude'] = t_amp
            features['T_time'] = t_idx / fs * 1000  # ms

            # T wave boundaries (10% peak threshold)
            thresh = 0.1 * abs(t_amp)

            # Search within the T window
            # Convert local index to window-relative index
            local_peak = t_local_idx

            # Find left boundary
            left_mask = np.abs(t_window[:local_peak]) <= thresh
            # Find the last index (closest to peak) that is below threshold
            left_indices = np.where(left_mask)[0]
            if len(left_indices) > 0:
                left = (
                    left_indices[-1] + 1
                )  # Use the point just after it goes above threshold
            else:
                left = 0  # No point below threshold found to the left

            # Find right boundary
            right_mask = np.abs(t_window[local_peak + 1 :]) <= thresh
            right_indices = np.where(right_mask)[0]
            if len(right_indices) > 0:
                right = (
                    local_peak + 1 + right_indices[0] - 1
                )  # Use point just before it drops
            else:
                right = len(t_window) - 1

            t_width = (right - left + 1) * dt
            t_area = np.sum(t_window[left : right + 1]) * dt

    features["T_peak_time"] = t_idx / fs * 1000 if not np.isnan(t_idx) else np.nan  # ms
    features["T_peak_amplitude"] = t_amp
    features["T_peak_latency"] = t_latency
    features["T_width"] = t_width
    features["T_area"] = t_area
    features["T_sign"] = t_sign

    # --- 4. Extract ST Segment Information ---
    st_level_60 = st_level_80 = st_slope = st_area = st_min = st_mean = np.nan

    if 'S_time' in features:
        # Backward compatibility for calculation logic requiring indices
        # We can reconstruct index from time (ms): idx = int(time * fs / 1000)
        s_idx = int(features['S_time'] * fs / 1000)
        j_idx = s_idx + int(0.04 * fs)  # J point assumed 40ms after S

        def mean_at_ms(start_idx, ms, window_ms=10):
            center = start_idx + int(ms * fs / 1000)
            half = int(window_ms * fs / 1000 / 2)
            a = max(0, center - half)
            b = min(n, center + half + 1)
            return np.mean(signal[a:b]) if a < b else np.nan

        st_level_60 = mean_at_ms(j_idx, 60)
        st_level_80 = mean_at_ms(j_idx, 80)

        st_start = j_idx
        st_end = min(n, j_idx + int(0.08 * fs))  # 80ms window

        # Avoid T wave overlap
        if not np.isnan(t_idx):
            st_end = min(st_end, int(t_idx))

        if st_end > st_start:
            st_seg = signal[st_start:st_end]
            if len(st_seg) > 1:
                t_seg_ms = np.arange(len(st_seg)) * dt
                try:
                    st_slope = np.polyfit(t_seg_ms, st_seg, 1)[0]
                except np.linalg.LinAlgError:
                    st_slope = 0
                st_area = np.sum(st_seg) * dt
                st_min = np.min(st_seg)
                st_mean = np.mean(st_seg)

        # Save indices directly for easier plotting/verification
        features["ST_start_idx"] = st_start
        features["ST_end_idx"] = st_end

    features["ST_level_60"] = st_level_60
    features["ST_level_80"] = st_level_80
    features["ST_slope"] = st_slope
    features["ST_area"] = st_area
    features["ST_min"] = st_min
    features["ST_mean"] = st_mean

    # Add ST measurement point time for visualization
    if 'S_time' in features:
        features['ST_time_60ms'] = j_idx + 60
        features['ST_amplitude_60ms'] = st_level_60

    return features


# --- 使用示例 ---
file_path = 'machine_learning/data/Ischemia_Dataset/normal_male/healthy/d12_noisy_dataset/d_part_000.h5'

with h5py.File(file_path, 'r') as f:
    batch_data = f['X'][:]

print(f"Loaded data shape: {batch_data.shape}")
sampling_rate = 1000
single_beat = batch_data[0, :, 6]
# Denoise the signal
single_beat = gaussian_filter1d(single_beat, sigma=2.0)
features = extract_ecg_features(single_beat, fs=sampling_rate)
print("提取的特征:")
for key, value in features.items():
    print(f"{key}: {value}")


# --- Visualization ---
plt.figure(figsize=(10, 6))
time_axis = np.arange(len(single_beat)) / sampling_rate * 1000  # ms
plt.plot(time_axis, single_beat, label='ECG Signal', color='black')

if 'ST_start_idx' in features and 'ST_end_idx' in features:
    st_start = features['ST_start_idx']
    st_end = features['ST_end_idx']
    if st_end > st_start:
        plt.plot(
            time_axis[st_start:st_end],
            single_beat[st_start:st_end],
            color='orange',
            linewidth=3,
            label='ST Segment',
        )

# Plot detected points
points_to_plot = {
    'Q': ('Q_time', 'Q_amplitude', 'red', 'v'),
    'R': ('R_time', 'R_amplitude', 'red', '^'),
    'S': ('S_time', 'S_amplitude', 'red', 'v'),
    'T': ('T_time', 'T_amplitude', 'blue', 'o'),
    'ST': ('ST_time_60ms', 'ST_amplitude_60ms', 'green', 'x'),
}

for wave, (time_key, amp_key, color, marker) in points_to_plot.items():
    if time_key in features and amp_key in features:
        t_val = features[time_key]
        amp = features[amp_key]
        plt.scatter(
            t_val,
            amp,
            color=color,
            marker=marker,
            label=f'{wave} wave',
            s=100,
            zorder=5,
        )
        plt.text(
            t_val,
            amp,
            f' {wave}',
            fontsize=12,
            color=color,
            verticalalignment='bottom',
        )

plt.title('Single Heartbeat ECG with Detected Features (Denoised)')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()
