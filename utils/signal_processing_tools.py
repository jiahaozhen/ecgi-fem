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


def _get_fiducial_indices(signal, fs=1000):
    n = len(signal)
    indices = {'r_idx': 0, 's_idx': 0, 't_idx': -1, 't_onset': 0, 't_offset': 0}

    if n == 0:
        return indices

    signal = np.nan_to_num(signal)

    # --- 1. R wave ---
    r_idx = np.argmax(np.abs(signal))
    r_amp = signal[r_idx]
    indices['r_idx'] = r_idx

    # --- 2. S wave ---
    if r_amp >= 0:
        find_extremum = np.argmin
    else:
        find_extremum = np.argmax

    s_idx = r_idx
    s_end_search = min(n, r_idx + int(0.05 * fs))
    if s_end_search > r_idx:
        s_window = signal[r_idx:s_end_search]
        if len(s_window) > 0:
            s_local_idx = find_extremum(s_window)
            s_idx = r_idx + s_local_idx
    indices['s_idx'] = s_idx

    # --- 3. T wave ---
    t_start = r_idx + int(0.2 * fs)
    t_end = min(n, r_idx + int(0.45 * fs))

    if t_start < n and t_end > t_start:
        t_window = signal[t_start:t_end]
        if len(t_window) > 0:
            t_local_idx = np.argmax(np.abs(t_window))
            t_idx = t_start + t_local_idx

            t_amp_val = signal[t_idx]
            indices['t_idx'] = t_idx

            thresh = 0.1 * abs(t_amp_val)

            # Left
            left_search = t_window[:t_local_idx]
            left_indices = np.where(np.abs(left_search) <= thresh)[0]
            if len(left_indices) > 0:
                left_local = left_indices[-1] + 1
            else:
                left_local = 0
            indices['t_onset'] = t_start + left_local

            # Right
            right_search = t_window[t_local_idx + 1 :]
            right_indices = np.where(np.abs(right_search) <= thresh)[0]
            if len(right_indices) > 0:
                right_local = t_local_idx + 1 + right_indices[0] - 1
            else:
                right_local = len(t_window) - 1
            indices['t_offset'] = t_start + right_local

    return indices


def _extract_single_lead_features(signal, fs=1000, fiducials=None):
    """
    针对单次心跳片段提取形态学特征 (考虑信号绝对值进行波形定位)
    :param signal: 单次心拍的幅值数组
    :param fs: 采样率 (Hz)
    :param fiducials: Dict containing global fiducial indices (r_idx, s_idx, t_idx, t_onset, t_offset)
    :return: 包含特征的字典
    """
    # Initialize all simple features with 0.0 to avoid NaNs
    st_level_60 = 0.0
    st_level_80 = 0.0
    st_slope = 0.0
    st_area = 0.0
    st_min = 0.0
    st_mean = 0.0
    t_amp = 0.0
    t_latency = 0.0
    t_width = 0.0
    t_area = 0.0
    t_sign = 0.0

    n = len(signal)
    if n == 0:
        return {
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

    # Ensure no NaNs in signal
    signal = np.nan_to_num(signal)
    dt = 1000.0 / fs  # ms per sample

    if fiducials is None:
        fiducials = _get_fiducial_indices(signal, fs)

    r_idx = fiducials['r_idx']
    s_idx = fiducials['s_idx']
    t_idx = fiducials['t_idx']
    t_onset = fiducials.get('t_onset', 0)
    t_offset = fiducials.get('t_offset', 0)

    # --- T Wave ---
    if t_idx != -1 and t_idx < n:
        t_amp = signal[t_idx]
        t_latency = t_idx * dt
        t_sign = np.sign(t_amp)

        safe_onset = max(0, min(t_onset, n - 1))
        safe_offset = max(0, min(t_offset, n - 1))

        if safe_offset >= safe_onset:
            t_width = (safe_offset - safe_onset + 1) * dt
            t_area = np.sum(signal[safe_onset : safe_offset + 1]) * dt

    # --- ST Segment Information ---
    j_idx = s_idx + int(0.04 * fs)

    # Helper for safe window mean
    def safe_window_mean(center_idx, half_win=int(0.005 * fs)):
        if center_idx >= n:
            return 0.0
        a = max(0, center_idx - half_win)
        b = min(n, center_idx + half_win + 1)
        if b > a:
            return np.mean(signal[a:b])
        return 0.0

    if j_idx < n:
        st_level_60 = safe_window_mean(j_idx + int(0.06 * fs))
        st_level_80 = safe_window_mean(j_idx + int(0.08 * fs))

        st_seg_end = min(n, j_idx + int(0.08 * fs))

        if t_idx != -1 and t_idx > j_idx:
            st_seg_end = min(st_seg_end, t_idx)

        if st_seg_end > j_idx:
            st_seg = signal[j_idx:st_seg_end]
            if len(st_seg) > 1:
                t_seg_ms = np.arange(len(st_seg)) * dt
                try:
                    slope, _ = np.polyfit(t_seg_ms, st_seg, 1)
                    st_slope = slope
                except:
                    st_slope = 0.0

                st_area = np.sum(st_seg) * dt
                st_min = np.min(st_seg)
                st_mean = np.mean(st_seg)

    return {
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

    # Calculate global fiducials from RMS of denoised data
    # Filter all leads first to get clean RMS
    data_clean = gaussian_filter1d(data.astype(float), sigma=2.0, axis=0)
    rms_signal = np.sqrt(np.mean(data_clean**2, axis=1))

    global_fiducials = _get_fiducial_indices(rms_signal, fs=fs)

    for lead_idx in range(n_leads):
        signal = data_clean[:, lead_idx]

        # Extract features using the global fiducials
        lead_features = _extract_single_lead_features(
            signal, fs=fs, fiducials=global_fiducials
        )
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


def get_feature_names():
    """
    Return the list of base feature names extracted from ECG signals.
    """
    return [
        "ST_level_60",
        "ST_level_80",
        "ST_slope",
        "ST_area",
        "ST_min",
        "ST_mean",
        "T_peak_amplitude",
        "T_peak_latency",
        "T_width",
        "T_area",
        "T_sign",
    ]


def get_statistical_feature_names():
    """
    Return the list of statistical feature names (mean and std for each base feature).
    """
    base_features = get_feature_names()
    stats_features = []
    for feat in base_features:
        stats_features.append(f"{feat}_mean")
        stats_features.append(f"{feat}_std")
    return stats_features
