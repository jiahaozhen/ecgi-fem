import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
import h5py
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


def transfer_bsp_to_standard12lead(
    bsp_data: np.ndarray,
    case_name: str = 'normal_male',
):
    geom_file = f'forward_inverse_3d/data/raw_data/geom_{case_name}.mat'
    geom = h5py.File(geom_file, 'r')
    lead_index = np.array(geom['leadelec']).astype(int) - 1
    lead_index = lead_index.reshape(-1)
    standard12Lead = np.zeros((bsp_data.shape[0], 12))
    # I = VL - VR
    standard12Lead[:, 0] = bsp_data[:, lead_index[7]] - bsp_data[:, lead_index[6]]
    # II = VF - VR
    standard12Lead[:, 1] = bsp_data[:, lead_index[8]] - bsp_data[:, lead_index[6]]
    # III = VF - VL
    standard12Lead[:, 2] = bsp_data[:, lead_index[8]] - bsp_data[:, lead_index[7]]
    # Vi = Vi - (VR + VL + VF) / 3
    standard12Lead[:, 3:9] = bsp_data[:, lead_index[0:6]] - np.mean(
        bsp_data[:, lead_index[6:9]], axis=1, keepdims=True
    )
    # aVR = VR - (VL + VF) / 2
    standard12Lead[:, 9] = bsp_data[:, lead_index[6]] - np.mean(
        bsp_data[:, lead_index[7:9]], axis=1
    )
    # aVL = VL - (VR + VF) / 2
    standard12Lead[:, 10] = bsp_data[:, lead_index[7]] - np.mean(
        bsp_data[:, [lead_index[6], lead_index[8]]], axis=1
    )
    # aVF = VF - (VR + VL) / 2
    standard12Lead[:, 11] = bsp_data[:, lead_index[8]] - np.mean(
        bsp_data[:, lead_index[6:8]], axis=1
    )

    return standard12Lead


def transfer_bsp_to_standard300lead(
    bsp_data: np.ndarray, case_name: str = 'normal_male'
):
    geom_file = f'forward_inverse_3d/data/raw_data/geom_{case_name}.mat'
    geom = h5py.File(geom_file, 'r')
    lead_index = np.array(geom['leadelec']).astype(int) - 1
    lead_index = lead_index.reshape(-1)
    lead_index = lead_index[-3:]
    bsp_data = np.asarray(bsp_data, dtype=float)
    bsp_data = bsp_data - np.mean(bsp_data[:, lead_index], axis=1, keepdims=True)
    return bsp_data


def transfer_bsp_to_standard64lead(
    bsp_data: np.ndarray, case_name: str = 'normal_male'
):
    geom_file = f'forward_inverse_3d/data/raw_data/geom_{case_name}.mat'
    geom = h5py.File(geom_file, 'r')
    lead_index = np.array(geom['leadelec']).astype(int) - 1
    lead_index = lead_index.reshape(-1)
    lead_index = lead_index[-3:]
    bsp_data = np.asarray(bsp_data, dtype=float)
    if lead_index is not None:
        bsp_data = bsp_data[:, 0:64] - np.mean(
            bsp_data[:, lead_index], axis=1, keepdims=True
        )
    return bsp_data


def smooth_ecg_gaussian(ecg_matrix, sigma=2.0):
    """
    对 N×12 ECG 信号矩阵进行逐导联高斯平滑。
    sigma: 平滑宽度（越大越平滑）
    """
    ecg_matrix = np.asarray(ecg_matrix, dtype=float)
    if ecg_matrix.ndim != 2:
        raise ValueError(f"输入应为二维矩阵 (N, 12)，当前形状 {ecg_matrix.shape}")

    N, leads = ecg_matrix.shape
    smoothed = np.zeros_like(ecg_matrix)

    for i in range(leads):
        sig = np.nan_to_num(ecg_matrix[:, i])
        smoothed[:, i] = gaussian_filter1d(sig, sigma=sigma)

    return smoothed


def smooth_ecg_mean(ecg_matrix, window_size=50):
    """
    对 N×12 ECG 信号矩阵进行滑动平均平滑。
    window_size: 滑动窗口长度（奇数）
    """
    ecg_matrix = np.asarray(ecg_matrix, dtype=float)
    if ecg_matrix.ndim != 2:
        raise ValueError(f"输入应为二维矩阵 (N, 12)，当前形状 {ecg_matrix.shape}")

    N, leads = ecg_matrix.shape
    smoothed = np.zeros_like(ecg_matrix)

    kernel = np.ones(window_size) / window_size

    for i in range(leads):
        sig = np.nan_to_num(ecg_matrix[:, i])
        smoothed[:, i] = np.convolve(sig, kernel, mode='same')

    return smoothed


def smooth_ecg_savgol(ecg_matrix, window_length=11, polyorder=3):
    """
    对 N×12 ECG 信号矩阵进行 Savitzky–Golay 平滑。
    window_length: 窗口长度（必须为奇数）
    polyorder: 局部多项式阶数
    """
    ecg_matrix = np.asarray(ecg_matrix, dtype=float)
    if ecg_matrix.ndim != 2:
        raise ValueError(f"输入应为二维矩阵 (N, 12)，当前形状 {ecg_matrix.shape}")

    N, leads = ecg_matrix.shape
    smoothed = np.zeros_like(ecg_matrix)

    # 确保窗口长度有效
    if window_length % 2 == 0:
        window_length += 1
    if window_length >= N:
        window_length = N - 1 if N % 2 == 0 else N

    for i in range(leads):
        sig = np.nan_to_num(ecg_matrix[:, i])
        smoothed[:, i] = savgol_filter(
            sig, window_length=window_length, polyorder=polyorder
        )

    return smoothed


def add_noise_based_on_snr(data: np.ndarray, snr: float) -> np.ndarray:
    """
    Add noise to the data based on the specified SNR (Signal-to-Noise Ratio).

    Parameters:
    - data: The original data to which noise will be added.
    - snr: The desired SNR in decibels (dB).

    Returns:
    - Noisy data with the specified SNR.
    """
    signal_power = np.mean(data**2)
    noise_power = signal_power / (10 ** (snr / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), data.shape)
    noisy_data = data + noise
    return noisy_data


def normalize_ecg_zscore(ecg: np.ndarray) -> np.ndarray:
    ecg = ecg.T

    means = np.mean(ecg, axis=1, keepdims=True)
    stds = np.std(ecg, axis=1, keepdims=True)
    stds[stds == 0] = 1e-8  # avoid division by zero

    ecg = (ecg - means) / stds

    return ecg.T


def check_noise_level_snr(data: np.ndarray, noise: np.ndarray) -> float:
    """
    Check the SNR (Signal-to-Noise Ratio) of the data.

    Parameters:
    - data: The original data.
    - noise: The noise added to the data.

    Returns:
    - SNR in decibels (dB).
    """
    signal_power = np.mean(data**2)
    noise_power = np.mean(noise**2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def load_bsp_pts(case_name: str = 'normal_male'):
    path = f'forward_inverse_3d/data/raw_data/geom_{case_name}.mat'
    geom = h5py.File(path, 'r')
    points = np.array(geom['geom_thorax']['pts'])
    return points


def unwrap_surface(original_pts):
    """
    将环绕身体的 3D 电极点展开到 2D 平面
    返回 2D 坐标 (num_leads, 2)
    """
    X, Y, Z = original_pts[:, 0], original_pts[:, 1], original_pts[:, 2]
    theta = np.arctan2(Y, X)  # 围绕中心旋转角度
    return np.column_stack([theta, Z])  # (num_leads, 2)


def project_bsp_on_surface(bsp_data, original_pts=load_bsp_pts(), length=50, width=50):
    num_timepoints, num_leads = bsp_data.shape

    xy_pts = unwrap_surface(original_pts)  # (num_leads, 2)

    # 构建规则网格
    x_min, x_max = xy_pts[:, 0].min(), xy_pts[:, 0].max()
    y_min, y_max = xy_pts[:, 1].min(), xy_pts[:, 1].max()
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, length), np.linspace(y_min, y_max, width)
    )
    grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    # 一次性构建两个插值器
    lin_interp = LinearNDInterpolator(xy_pts, np.zeros(num_leads))
    near_interp = NearestNDInterpolator(xy_pts, np.zeros(num_leads))

    surface_bsp = np.empty((num_timepoints, width, length), dtype=np.float64)

    for t in range(num_timepoints):
        voltages = bsp_data[t, :]

        # 更新插值器数值
        lin_interp.values[:] = voltages.reshape(-1, 1)
        near_interp.values[:] = voltages
        # 线性插值
        grid_z = lin_interp(grid_pts).reshape(width, length)

        # 最近邻插值（完整）
        grid_z_near = near_interp(grid_pts).reshape(width, length)

        # 补洞，不会再 shape mismatch
        nan_mask = np.isnan(grid_z)
        if np.any(nan_mask):
            grid_z[nan_mask] = grid_z_near[nan_mask]

        surface_bsp[t] = grid_z


def _extract_single_lead_features(signal, fs=1000):
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
            features['T_peak_time'] = t_idx / fs * 1000  # ms

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
        features['J_time'] = j_idx * dt

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

    features["ST_level_60"] = st_level_60
    features["ST_level_80"] = st_level_80
    features["ST_slope"] = st_slope
    features["ST_area"] = st_area
    features["ST_min"] = st_min
    # Filter features to return only ST and T wave features
    final_features = {
        "ST_level_60": st_level_60,
        "ST_level_80": st_level_80,
        "ST_slope": st_slope,
        "ST_area": st_area,
        "ST_min": st_min,
        "ST_mean": st_mean,
        "T_peak_amplitude": t_amp,
        "T_peak_latency": t_latency,
        "T_width": t_width,
        "T_area": t_area,
        "T_sign": t_sign,
    }

    return final_features


def extract_features(data, fs=1000):
    """
    Extract ECG features from (T, D) data.

    Parameters:
    - data: (T, D) numpy array.
            T: Time steps (n_samples)
            D: Dimensions/Leads (n_leads)
    - fs: sampling frequency in Hz (default 1000).

    Returns:
    - features_df: DataFrame with features for each lead.
    """
    n_samples, n_leads = data.shape

    features = []

    for lead_idx in range(n_leads):
        signal = data[:, lead_idx]

        # Denoise the signal
        signal = gaussian_filter1d(signal, sigma=2.0)

        # Extract features using the improved single lead extractor
        lead_features = _extract_single_lead_features(signal, fs=fs)
        features.append(lead_features)

    return pd.DataFrame(features)


def batch_extract_features(data_batch, fs=1000):
    """
    Batch process ECG data to extract features.

    Parameters:
    - data_batch: (B, T, D) numpy array.
        B: Batch size (number of records)
        T: Time steps (number of samples)
        D: Dimensions (number of leads)
    - fs: sampling frequency in Hz.

    Returns:
    - features_array: (B, D, F) numpy array.
    - feature_names: list of feature names.
    """
    if data_batch.ndim != 3:
        raise ValueError("data_batch must be 3D: (B, T, D)")

    B, T, D = data_batch.shape

    features_list = []
    feature_names = None

    print(f"Starting batch processing for {B} records...")

    for i in range(B):
        # Extract features for single record
        record_data = data_batch[i]

        try:
            df = extract_features(record_data, fs=fs)

            # Remove lead_id as it corresponds to the row index
            if 'lead_id' in df.columns:
                df = df.drop(columns=['lead_id'])

            if feature_names is None:
                feature_names = df.columns.tolist()

            features_list.append(df.values)

        except Exception as e:
            print(f"Error processing record {i}: {e}")
            features_list.append(None)

    # Determine feature dimension F
    F = 0
    if feature_names:
        F = len(feature_names)

    if F == 0:
        return np.array([]), []

    # Create the result array (B, D, F)
    features_array = np.full((B, D, F), np.nan)

    for i, feats in enumerate(features_list):
        if feats is not None:
            if feats.shape == (D, F):
                features_array[i] = feats
            else:
                print(f"Shape mismatch for record {i}: {feats.shape} vs {(D, F)}")

    return features_array, feature_names


# 写一个新的batch_extract_features，提取一些所有导联的统计信息（这些信息依旧是之前函数里提到的，包括ST段信息和T波信息）
def batch_extract_statistical_features(data_batch, fs=1000):
    """
    Batch process ECG data to extract statistical features across all leads.

    Parameters:
    - data_batch: (B, T, D) numpy array.
        B: Batch size (number of records)
        T: Time steps (number of samples)
        D: Dimensions (number of leads)
    - fs: sampling frequency in Hz.

    Returns:
    - features_array: (B, F) numpy array.
    - feature_names: list of feature names.
    """
    if data_batch.ndim != 3:
        raise ValueError("data_batch must be 3D: (B, T, D)")

    B, T, D = data_batch.shape

    features_list = []
    feature_names = []

    print(f"Starting batch processing for {B} records...")

    for i in range(B):
        record_data = data_batch[i]

        try:
            df = extract_features(record_data, fs=fs)

            # Initialize feature names on first success
            if len(feature_names) == 0:
                for col in df.columns:
                    feature_names.append(f"{col}_mean")
                    feature_names.append(f"{col}_std")

            # Calculate statistical features across all leads
            record_stats = []
            for col in df.columns:
                record_stats.append(df[col].mean())
                record_stats.append(df[col].std())

            features_list.append(record_stats)

        except Exception as e:
            print(f"Error processing record {i}: {e}")
            features_list.append(None)

    # Handle case where no records were successfully processed
    if len(feature_names) == 0:
        return np.array([]), []

    F = len(feature_names)
    features_array = np.full((B, F), np.nan)

    for i, feats in enumerate(features_list):
        if feats is not None:
            if len(feats) == F:
                features_array[i] = feats
            else:
                print(f"Shape mismatch for record {i}")

    return features_array, feature_names
