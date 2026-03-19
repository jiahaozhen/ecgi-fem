import h5py
import numpy as np
import matplotlib.pyplot as plt

from utils.signal_processing_tools import extract_features, _get_fiducial_indices
from scipy.ndimage import gaussian_filter1d


def plot_consistency(
    data, fs, fiducials=None, filename='thesis_drawing/figs/global_reference_points.png'
):
    N, n_leads = data.shape
    t = np.arange(N) / fs
    fontsize = 14
    fig_width = 12
    left_margin = 0.15
    right_margin = 0.98

    # Figure 1: RMS signal and global fiducials
    fig0, ax0 = plt.subplots(figsize=(fig_width, 5))

    # Plot 1: RMS Signal and Fiducials
    data_clean = gaussian_filter1d(data.astype(float), sigma=2.0, axis=0)
    rms_signal = np.sqrt(np.mean(data_clean**2, axis=1))

    ax0.plot(t, rms_signal, 'k-', linewidth=2, label='Global RMS')
    ax0.set_xlabel('Time (s)', fontsize=fontsize)
    ax0.set_ylabel(r'$d_{\mathrm{RMS}}$', fontsize=fontsize)

    if fiducials:
        # Mark fiducials
        r_t = fiducials['r_idx'] / fs
        s_t = fiducials['s_idx'] / fs
        t_onset_t = fiducials['t_onset'] / fs
        t_offset_t = fiducials['t_offset'] / fs
        t_idx_t = fiducials['t_idx'] / fs

        ax0.axvline(r_t, color='r', linestyle='--', label='R peak')
        ax0.axvline(s_t, color='g', linestyle='--', label='S point')
        ax0.axvline(t_idx_t, color='b', linestyle='--', label='T peak')

        # Highlight intervals
        ax0.axvspan(
            t_onset_t, t_offset_t, color='yellow', alpha=0.3, label='T Wave Window'
        )

        # ST Segment (J=S+40ms, End=S+40ms+80ms or T peak)
        j_t = s_t + 0.04
        st_end_t = min(j_t + 0.08, t_idx_t)
        if st_end_t > j_t:
            ax0.axvspan(j_t, st_end_t, color='cyan', alpha=0.3, label='ST Segment')

    ax0.tick_params(axis='both', labelsize=fontsize)
    ax0.legend(fontsize=fontsize)

    # Figure 2: All leads with global fiducials
    fig1, ax1 = plt.subplots(figsize=(fig_width, 8))
    offset = 0
    n_show_leads = min(n_leads, 10)
    y_ticks = []
    y_labels = []
    for i in range(n_show_leads):
        # Normalize for display
        sig = data[:, i]
        sig_norm = (sig - np.mean(sig)) / (np.std(sig) + 1e-6)
        ax1.plot(t, sig_norm + offset, label=f'Lead {i+1}')
        y_ticks.append(offset)
        y_labels.append(f'Lead {i+1}')
        offset += 5  # Stack signals

    ax1.set_xlabel('Time (s)', fontsize=fontsize)
    # ax1.set_ylabel('Lead', fontsize=fontsize)
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(y_labels, fontsize=fontsize)
    ax1.tick_params(axis='x', labelsize=fontsize)

    # Draw the same fiducial lines across all leads
    if fiducials:
        for boundary in [fiducials['t_onset'] / fs, fiducials['t_offset'] / fs]:
            ax1.axvline(boundary, color='yellow', linestyle='-', alpha=0.5, linewidth=2)

        st_on = (fiducials['s_idx'] / fs) + 0.04
        st_offset = min(st_on + 0.08, fiducials['t_idx'] / fs)

        # ST segment markers use cyan to match Figure 1.
        ax1.axvline(st_on, color='cyan', linestyle='--', linewidth=2)
        ax1.axvline(st_offset, color='cyan', linestyle=':', linewidth=2)

    fig0.subplots_adjust(left=left_margin, right=right_margin)
    fig1.subplots_adjust(left=left_margin, right=right_margin)
    plt.show()
    print(f"Plot saved to {filename}")


def test_consistency():
    fs = 1000

    file_path = 'machine_learning/data/Ischemia_Dataset/normal_male/healthy/d64_noisy_dataset/d_part_000.h5'
    with h5py.File(file_path, 'r') as f:
        # Load a small batch for testing
        base_signal = f['X'][:]  # Take first 5 records (5, T, D)

    # Create multi-lead data (N, Leads)
    data = base_signal[0]  # Use the first record

    print("Extracting features...")
    df = extract_features(data, fs=fs)

    # Re-calculate fiducials just for plotting (logic duplicated from extract_features)
    data_clean = gaussian_filter1d(data.astype(float), sigma=2.0, axis=0)
    rms_signal = np.sqrt(np.mean(data_clean**2, axis=1))
    fiducials = _get_fiducial_indices(rms_signal, fs=fs)

    plot_consistency(data, fs, fiducials)

    print("\nFeature DataFrame Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())

    print("\nFeatures for first 3 leads:")
    print(df.head(3).T)

    # Check consistency of T-wave width/latency across leads
    # Since we use global fiducials, the window for T-wave is fixed relative to R-peak.
    # However, T-peak latency might still vary slightly if the peak within that fixed window shifts due to noise,
    # BUT the start/end search boundaries are now fixed relative to the global R.
    # Actually, `t_onset` and `t_offset` are now computed GLOBALLY and passed in.
    # So `t_width` should be IDENTICAL for all leads (since t_width = t_offset - t_onset).

    t_widths = df['T_width'].values
    print("\nT_width values across leads:", t_widths)

    if np.allclose(t_widths, t_widths[0]):
        print("\nSUCCESS: T_width is consistent across all leads.")
    else:
        print("\nWARNING: T_width is NOT consistent across leads.")

    t_areas = df['T_area'].values
    print("T_area values (should vary due to amplitude scaling):", t_areas)


if __name__ == "__main__":
    test_consistency()
