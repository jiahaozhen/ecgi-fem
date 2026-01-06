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
    dt = 1000.0 / fs  # ms per sample

    features = []

    for lead_idx in range(n_leads):
        signal = data[:, lead_idx]

        # 1. Baseline correction (using first 20ms)
        baseline_window = int(20 / dt)
        if baseline_window < 1:
            baseline_window = 1
        baseline = np.mean(signal[:baseline_window])
        signal_corrected = signal - baseline

        # 2. Find R peak
        # We assume the R peak is the maximum absolute value in the first half of the signal
        # (to avoid confusing it with a large T wave in the second half)
        search_len = min(n_samples, int(300 / dt))
        r_idx = np.argmax(np.abs(signal_corrected[:search_len]))

        # 3. Find J point (QRS end)
        # Heuristic: J point is approximately 40ms after R peak.
        # In a real application, this should be detected by slope changes.
        j_idx = r_idx + int(40 / dt)
        j_idx = min(n_samples - 1, j_idx)

        # 4. ST Segment Features
        # ST window: J+20ms to J+100ms (or J+80ms for calculation)

        def get_val_at_ms(start_idx, ms_offset):
            idx = start_idx + int(ms_offset / dt)
            if 0 <= idx < n_samples:
                return signal_corrected[idx]
            return np.nan

        st_level_60 = get_val_at_ms(j_idx, 60)
        st_level_80 = get_val_at_ms(j_idx, 80)

        # Slope: Linear fit between J+20 and J+80
        st_start_idx = j_idx + int(20 / dt)
        st_end_idx = j_idx + int(80 / dt)

        st_slope = np.nan
        st_area = np.nan
        st_min = np.nan
        st_mean = np.nan

        if (
            st_start_idx < n_samples
            and st_end_idx < n_samples
            and st_end_idx > st_start_idx
        ):
            st_segment = signal_corrected[st_start_idx : st_end_idx + 1]
            time_axis = np.arange(len(st_segment)) * dt

            if len(st_segment) > 1:
                # Slope (mV/ms)
                poly = np.polyfit(time_axis, st_segment, 1)
                st_slope = poly[0]

                # Area (mV * ms)
                st_area = np.sum(st_segment) * dt

                # Min / Mean
                st_min = np.min(st_segment)
                st_mean = np.mean(st_segment)

        # 5. T wave Features
        # Search window: J + 100ms to end
        t_search_start = j_idx + int(100 / dt)
        t_search_end = n_samples

        t_amp = np.nan
        t_latency = np.nan
        t_width = np.nan
        t_area = np.nan
        t_sign = np.nan
        t_idx = np.nan

        if t_search_start < t_search_end:
            t_window = signal_corrected[t_search_start:t_search_end]
            if len(t_window) > 0:
                # Find T peak (max abs)
                t_local_idx = np.argmax(np.abs(t_window))
                t_idx = t_search_start + t_local_idx

                t_amp = signal_corrected[t_idx]
                t_latency = t_idx * dt
                t_sign = np.sign(t_amp)

                # T Width: FWHM (Full Width at Half Maximum)
                half_max = t_amp * 0.5

                # Search left from peak
                left_idx = t_local_idx
                while left_idx > 0:
                    if np.sign(t_window[left_idx]) != np.sign(t_amp) or abs(
                        t_window[left_idx]
                    ) < abs(half_max):
                        break
                    left_idx -= 1

                # Search right from peak
                right_idx = t_local_idx
                while right_idx < len(t_window) - 1:
                    if np.sign(t_window[right_idx]) != np.sign(t_amp) or abs(
                        t_window[right_idx]
                    ) < abs(half_max):
                        break
                    right_idx += 1

                t_width = (right_idx - left_idx) * dt

                # T Area: Integral of the T wave
                # We integrate the part of the wave around the peak that has the same sign
                mask = np.sign(t_window) == np.sign(t_amp)
                t_area = np.sum(t_window[mask]) * dt

        lead_features = {
            'R_time': r_idx * dt,
            'J_time': j_idx * dt,
            'T_peak_time': t_idx * dt,
            'ST_level_60': st_level_60,
            'ST_level_80': st_level_80,
            'ST_slope': st_slope,
            'ST_area': st_area,
            'ST_min': st_min,
            'ST_mean': st_mean,
            'T_peak_amplitude': t_amp,
            'T_peak_latency': t_latency,
            'T_width': t_width,
            'T_area': t_area,
            'T_sign': t_sign,
        }
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
    feature_names = [
        'ST_level_60_mean',
        'ST_level_60_std',
        'ST_level_80_mean',
        'ST_level_80_std',
        'ST_slope_mean',
        'ST_slope_std',
        'ST_area_mean',
        'ST_area_std',
        'ST_min_mean',
        'ST_min_std',
        'ST_mean_mean',
        'ST_mean_std',
        'T_peak_amplitude_mean',
        'T_peak_amplitude_std',
        'T_width_mean',
        'T_width_std',
        'T_area_mean',
        'T_area_std',
    ]

    print(f"Starting batch processing for {B} records...")

    for i in range(B):
        record_data = data_batch[i]

        try:
            df = extract_features(record_data, fs=fs)

            # Calculate statistical features across all leads
            record_features = [
                df['ST_level_60'].mean(),
                df['ST_level_60'].std(),
                df['ST_level_80'].mean(),
                df['ST_level_80'].std(),
                df['ST_slope'].mean(),
                df['ST_slope'].std(),
                df['ST_area'].mean(),
                df['ST_area'].std(),
                df['ST_min'].mean(),
                df['ST_min'].std(),
                df['ST_mean'].mean(),
                df['ST_mean'].std(),
                df['T_peak_amplitude'].mean(),
                df['T_peak_amplitude'].std(),
                df['T_width'].mean(),
                df['T_width'].std(),
                df['T_area'].mean(),
                df['T_area'].std(),
            ]

            features_list.append(record_features)

        except Exception as e:
            print(f"Error processing record {i}: {e}")
            features_list.append([np.nan] * len(feature_names))

    features_array = np.array(features_list)
    return features_array, feature_names
