import numpy as np
import mne
import scipy.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False

BASE_DIR  = Path(__file__).parent
DATA_DIR  = BASE_DIR.parent
LOC_PATH  = BASE_DIR.parent / 'location_xyz.txt'
LABEL_CSV = BASE_DIR.parent / 'task6_labeled.csv'
MAT_FILES = [DATA_DIR / f'{i}.mat' for i in range(1, 7)]   # subjects 1-6 only
REPORT_DIR = BASE_DIR / 'reports'
REPORT_DIR.mkdir(exist_ok=True)


# ── 1. Load ──────────────────────────────────────────────────────────────────

def load_raw_mat(mat_path):
    try:
        mat = scipy.io.loadmat(str(mat_path))
        sr_key = next(k for k in mat if k.strip('\x00').lower() == 'sr')
        data = np.squeeze(mat['data'])   # (130, n_samples)
        sfreq = float(np.squeeze(mat[sr_key]))
    except NotImplementedError:          # فقط v7.3 HDF5 — نه هر Exception
        if not HDF5_AVAILABLE:
            raise ImportError("h5py required for v7.3 .mat files.")
        with h5py.File(str(mat_path), 'r') as f:
            sr_key = next(k for k in f if k.lower() == 'sr')
            data = np.array(f['data']).T
            sfreq = float(np.squeeze(np.array(f[sr_key])))

    eeg = data[:126, :] * 1e-6   # µV → V
    trigger = data[129, :]        # کانال 130ام MATLAB
    return eeg, trigger, sfreq

def load_channel_locations(loc_path):
    """Read channel names and XYZ positions from location_xyz.txt.
    File format: <index> <X> <Y> <Z> <channel_name>
    """
    ch_names, ch_pos = [], {}
    with open(loc_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            name = parts[4]          # ✅ اسم واقعی کانال
            ch_names.append(name)
            ch_pos[name] = np.array([float(parts[1]),
                                     float(parts[2]),
                                     float(parts[3])]) / 1000.0
    return ch_names, ch_pos


# ── 2. Epoching from task6_labeled.csv ───────────────────────────────────────

def epoching(raw, label_csv=LABEL_CSV):
    df = pd.read_csv(label_csv)  # columns: filename, value, id, label
    sfreq = raw.info['sfreq']
    n_times = raw.n_times

    events, valid_idx = [], []
    for i, row in df.iterrows():
        sample = int(round(row['value'] * sfreq))
        if 0 <= sample < n_times:
            events.append([sample, 0, int(row['label']) + 1])  # face=2, non-face=1
            valid_idx.append(i)

    events = np.array(events, dtype=int)
    epochs = mne.Epochs(
        raw, events,
        event_id={'non-face': 1, 'face': 2},
        tmin=-0.1, tmax=1.0,
        baseline=(-0.1, 0),
        preload=True, reject=None, verbose=False
    )

    valid_df = df.loc[valid_idx].reset_index(drop=True)

    # فقط ردیف‌هایی که MNE واقعاً نگه داشته (بعد از drop شدن epochهای خارج از محدوده)
    kept = epochs.selection
    epochs.metadata = valid_df.iloc[kept][['filename', 'id', 'label']].reset_index(drop=True)

    filenames = valid_df.iloc[kept]['filename'].tolist()
    labels = valid_df.iloc[kept]['label'].values

    return epochs, filenames, labels


# ── 3. Downsample ─────────────────────────────────────────────────────────────

def downsample_raw(raw, sfreq=250.0):
    """Resample raw to target sfreq."""
    raw.resample(sfreq, npad='auto')
    return raw


# ── 4. Filter ─────────────────────────────────────────────────────────────────

def apply_filters(raw):
    """Band-pass 0.5-100 Hz + notch at 50 Hz (powerline Iran/Europe)."""
    raw.filter(0.5, 100.0, fir_window='hamming', verbose=False)
    raw.notch_filter(50.0, verbose=False)
    return raw



# ── 5. Re-reference (CAR) ─────────────────────────────────────────────────────

def apply_car_reference(raw):
    """Common Average Reference."""
    raw.set_eeg_reference('average', projection=False, verbose=False)
    return raw


# ── 6. Baseline normalisation ─────────────────────────────────────────────────

def apply_baseline(epochs, baseline=(-0.1, 0)):
    """Subtract mean of baseline window from each epoch (applied in epoching too)."""
    epochs.apply_baseline(baseline, verbose=False)
    return epochs


# ── 7. Interpolate bad channels ───────────────────────────────────────────────

def interpolate_bad_channels(raw, ch_names, ch_pos):
    """
    Mark bad channels via STD-based MAD criterion, then interpolate with spherical spline.
    Must be called BEFORE epoching so that all epochs inherit the cleaned raw data.
    """
    # 1. Create montage from real coordinates
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
    raw.set_montage(montage, on_missing='warn', verbose=False)

    # 2. Detect bad channels
    data = raw.get_data()  # shape: (n_channels, n_samples)
    stds = data.std(axis=1)
    med = np.median(stds)
    mad = np.median(np.abs(stds - med))
    
    # Stricter threshold (3.5 instead of 5.0)
    threshold = 3.5
    bad_mask = (stds < med - threshold * mad) | (stds > med + threshold * mad)
    
    detected_bads = [ch_names[i] for i in np.where(bad_mask)[0]]
    
    # 3. Merge with manually marked bad channels (if any)
    if raw.info['bads']:
        detected_bads = list(set(detected_bads + raw.info['bads']))
    
    raw.info['bads'] = detected_bads
    
    # 4. Interpolate (only if we have bad channels)
    if raw.info['bads']:
        print(f"    Interpolating {len(raw.info['bads'])} bad channels: {raw.info['bads']}")
        raw.interpolate_bads(reset_bads=True, verbose=False)
    
    return raw


# ── 8. ICA ────────────────────────────────────────────────────────────────────

def run_ica(epochs, subject_id, report_dir=REPORT_DIR):
    """Fit ICA (24 components), save diagnostic plots. No auto-exclusion."""
    ica = mne.preprocessing.ICA(n_components=24, random_state=42, max_iter='auto')
    ica.fit(epochs, verbose=False)

    # Source time-series plot
    fig_src = ica.plot_sources(epochs, show=False)
    fig_src.savefig(report_dir / f'sub{subject_id}_ica_sources.png', dpi=100)
    plt.close(fig_src)

    # Component topographies
    fig_topo = ica.plot_components(show=False)
    for j, fig in enumerate(fig_topo if isinstance(fig_topo, list) else [fig_topo]):
        fig.savefig(report_dir / f'sub{subject_id}_ica_topo_{j}.png', dpi=100)
        plt.close(fig)

    return ica


def reject_channels_based_on_ica(epochs, ica, exclude_components=None):
    """Apply ICA cleaning. Components selected after visual inspection are removed."""

    if exclude_components is not None:
        ica.exclude = exclude_components

    else:
        # Manual rejection based on ICA inspection
        ica.exclude = [8, 12, 17, 19]

    print("Excluded ICA components:", ica.exclude)

    ica.apply(epochs, verbose=False)

    return epochs


# ── 9. Save ───────────────────────────────────────────────────────────────────

def save_results(epochs, subject_id, tag, report_dir=REPORT_DIR):
    """
    Save epochs as FIF and MAT, plus clean ERP comparison figure with CI.
    """
    fif_path = report_dir / f'sub{subject_id}_{tag}-epo.fif'
    epochs.save(str(fif_path), overwrite=True, verbose=False)

    # Save MAT (including metadata)
    scipy.io.savemat(
        str(report_dir / f'sub{subject_id}_{tag}.mat'),
        {
            'data':     epochs.get_data(),
            'times':    epochs.times,
            'ch_names': epochs.ch_names,
            'sfreq':    epochs.info['sfreq'],
            'events':   epochs.events,
            'filename': epochs.metadata['filename'].values if epochs.metadata is not None else [],
            'id':       epochs.metadata['id'].values if epochs.metadata is not None else [],
            'label':    epochs.metadata['label'].values if epochs.metadata is not None else [],
        }
    )

    # Plot ERP Face vs Non-face with CI in ROI (P7, P8, PO7, PO8)
    try:
        roi_channels = ['P7', 'P8', 'PO7', 'PO8']
        # Check which channels are available
        available = [ch for ch in roi_channels if ch in epochs.ch_names]
        if not available:
            print(f"    Warning: No ROI channels found; skipping ERP plot.")
            return
        
        epochs_roi = epochs.copy().pick_channels(available)
        
        # Build Evoked for each condition
        evokeds = {}
        for cond in ['face', 'non-face']:
            if cond in epochs_roi.event_id:
                evokeds[cond] = epochs_roi[cond].average()
        
        if len(evokeds) < 2:
            print(f"    Warning: Need both 'face' and 'non-face' conditions; skipping.")
            return
        
        # Plot with CI (like your first image: red/blue with shadow)
        fig, ax = plt.subplots(figsize=(10, 6))
        mne.viz.plot_compare_evokeds(
            evokeds,
            axes=ax,
            show=False,
            legend='upper right',
            ci=0.95,  # 95% confidence interval
            colors={'face': 'blue', 'non-face': 'red'},
            linestyles={'face': '-', 'non-face': '-'},
            title=f'Subject {subject_id} — {tag.upper()} (ROI: {", ".join(available)})'
        )
        ax.axvline(0, color='black', linestyle='--', linewidth=1, label='Stimulus Onset')
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Amplitude (µV)', fontsize=12)
        ax.legend(loc='upper right')
        
        fig.tight_layout()
        fig.savefig(report_dir / f'sub{subject_id}_{tag}_erp_roi.png', dpi=150)
        plt.close(fig)
        
        print(f"    Saved ERP plot: sub{subject_id}_{tag}_erp_roi.png")
    
    except Exception as e:
        print(f"    ERP plot failed: {e}")

    print(f'  Saved {fif_path.name}')


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(mat_path, subject_id, bad_channels=None, exclude_ica=None):
    print(f'\n=== Subject {subject_id} ===')

    # 1. Load
    eeg, trigger, orig_sfreq = load_raw_mat(mat_path)
    ch_names, ch_pos = load_channel_locations(LOC_PATH)

    info = mne.create_info(ch_names=ch_names, sfreq=orig_sfreq, ch_types='eeg')
    raw  = mne.io.RawArray(eeg, info, verbose=False)
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
    raw.set_montage(montage, on_missing='warn', verbose=False)
    # 2. Downsample (before epoching so onset times stay consistent)
    raw = downsample_raw(raw)

    # 3. Filter
    raw = apply_filters(raw)

    # 4. CAR
    raw = apply_car_reference(raw)

    # 5. Epoch using task6_labeled.csv onsets + labels
    epochs, filenames, labels = epoching(raw)
    #print(f'  Epochs: {epochs}')

    # 6. Baseline (already applied in epoching; explicit call for safety)
    epochs = apply_baseline(epochs)

    # 7. Interpolate bad channels (on raw before epoching is ideal,
    #    but applied here on the montage-set raw copy)
    if bad_channels:
        raw.info['bads'] = bad_channels
        raw.interpolate_bads(reset_bads=True, verbose=False)
    else:
        raw = interpolate_bad_channels(raw, ch_names, ch_pos)

    # 8. Save pre-ICA
    save_results(epochs, subject_id, 'pre_ica')

    # 9. ICA
    ica = run_ica(epochs, subject_id)
    epochs = reject_channels_based_on_ica(epochs, ica, exclude_ica)

    # 10. Save post-ICA
    save_results(epochs, subject_id, 'post_ica')

    print(f'  Final shape: {epochs.get_data().shape}')
    return epochs, ica


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    BAD_CHANNELS  = {i: None for i in range(1, 7)}
    EXCLUDE_ICA   = {i: None for i in range(1, 7)}

    for i, mat_path in enumerate(MAT_FILES, start=1):
        if not mat_path.exists():
            print(f'Subject {i}: file not found, skipping.')
            continue
        run_pipeline(mat_path, subject_id=i,
                     bad_channels=BAD_CHANNELS[i],
                     exclude_ica=EXCLUDE_ICA[i])
