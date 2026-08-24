"""
erp_analysis.py
----------------------------------------------------------------------
ERP analysis pipeline built on top of the ALREADY CLEANED post-ICA
epochs (sub{ID}_post_ica-epo.fif), NOT the raw pre-processing .mat files.

Rationale (Luck, "An Introduction to the Event-Related Potential
Technique"): each trial = signal + noise. Signal is (approximately)
identical across trials while noise fluctuates randomly around zero.
Averaging many trials aligned to time = 0 cancels the noise and leaves
the ERP. Since alignment to the trigger (time = 0) and baseline
correction were already done during preprocessing/epoching, we only
need to average across trials here.

This script answers three questions:
    Q1: ERP for all channels (grand average across all trials).
    Q2: Face vs. Non-face ERP comparison with confidence intervals.
    Q3: N170 component -> timing + amplitude comparison, t-test,
        and a "searchlight" time-course of the statistic.
----------------------------------------------------------------------
"""

import os
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from scipy import stats

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
REPORT_DIR = r"D:/8thSeme/neuroscience/EEG/EEG data/data/codes/reports"
LABEL_CSV = r"D:/8thSeme/neuroscience/EEG/EEG data/data/task6_labeled.csv"
RESULTS_DIR = r"D:/8thSeme/neuroscience/EEG/EEG data/data/codes/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

SUBJECT_IDS = [1, 2, 3, 4, 5, 6]

# Regions of interest for component analyses
N170_CHANNELS = ["P7", "P8", "PO7", "PO8"]   # temporo-occipital ROI
#؟؟
N170_WINDOW = (0.130, 0.200)                 # seconds
#؟؟N170_window?
ALPHA = 0.05
#alpha

# ----------------------------------------------------------------------
# 1. LOAD POST-ICA EPOCHS AND ATTACH FACE / NON-FACE LABELS
# ----------------------------------------------------------------------
def load_clean_epochs(subject_id, report_dir=REPORT_DIR):
    """
    Load the post-ICA cleaned epochs (already baseline-corrected and
    time-locked to stimulus onset during preprocessing).
    """
    fif_path = os.path.join(report_dir, f"sub{subject_id}_post_ica-epo.fif")
    epochs = mne.read_epochs(fif_path, preload=True, verbose=False)
    return epochs


def attach_labels(epochs, label_csv=LABEL_CSV):
    """
    Attach filename / id / label columns from task6_labeled.csv to
    epochs.metadata.

    IMPORTANT: the epochs were created from ALL rows of the label CSV,
    in chronological order (df sorted by 'value' = stimulus onset time).
    Some rows may have been dropped during epoch creation (e.g. onset
    too close to the recording edge). MNE keeps track of which original
    events survived through `epochs.selection`, so we index the label
    dataframe with that array to recover perfect row-to-epoch alignment
    without guessing.
    """
    df = pd.read_csv(label_csv)  # columns: filename, value, id, label
    df = df.reset_index(drop=True)

    # epochs.selection holds the indices (into the original events array,
    # i.e. into df rows, since events were built 1:1 from df rows in
    # chronological order) that survived epoch creation/rejection.
    valid_df = df.iloc[epochs.selection].reset_index(drop=True)

    assert len(valid_df) == len(epochs), (
        f"Mismatch: {len(valid_df)} label rows vs {len(epochs)} epochs. "
        "Check that epochs.selection indexes df rows correctly."
    )

    epochs.metadata = valid_df
    return epochs


def get_face_nonface(epochs):
    """
    Split epochs into face (label == 1) and non-face (label == 0)
    conditions using the attached metadata.
    """
    face = epochs["label == 1"]
    nonface = epochs["label == 0"]
    return face, nonface


# ----------------------------------------------------------------------
# Q1: ERP FOR ALL CHANNELS
# ----------------------------------------------------------------------
def plot_all_channels_erp(epochs, subject_id, out_dir=RESULTS_DIR):
    """
    Compute the grand-average evoked response (ERP) across ALL trials
    (face + non-face combined) and plot every channel.
    """
    evoked_all = epochs.average()

    fig = evoked_all.plot(spatial_colors=True, gfp=True, show=False)
    fig.savefig(os.path.join(out_dir, f"sub{subject_id}_erp_all_channels.png"),
                dpi=150)
    plt.close(fig)

    # Butterfly + topomap joint plot, useful overview figure
    fig2 = evoked_all.plot_joint(show=False)
    fig2.savefig(os.path.join(out_dir, f"sub{subject_id}_erp_joint.png"),
                 dpi=150)
    plt.close(fig2)

    return evoked_all


# ----------------------------------------------------------------------
# Q2: FACE vs NON-FACE ERP COMPARISON WITH CONFIDENCE INTERVALS
# ----------------------------------------------------------------------
def compare_face_nonface(epochs, subject_id, channels=None,
                          out_dir=RESULTS_DIR):
    """
    Plot face vs. non-face ERPs (mean +/- 95% CI, computed across
    trials) for the given channel(s). If channels=None, use the
    N170 ROI by default.
    """
    if channels is None:
        channels = N170_CHANNELS

    face, nonface = get_face_nonface(epochs)

    evokeds = {
        "Face": list(face.iter_evoked()),        # each trial as "evoked"
        "Non-face": list(nonface.iter_evoked()),
    }

    fig = mne.viz.plot_compare_evokeds(
        evokeds,
        picks=channels,
        combine="mean",
        ci=0.95,            # 95% confidence interval (bootstrapped by MNE)
        show=False,
        title=f"Subject {subject_id}: Face vs Non-face (ROI={channels})",
    )
    if isinstance(fig, list):
        fig = fig[0]
    fig.savefig(
        os.path.join(out_dir, f"sub{subject_id}_face_vs_nonface_CI.png"),
        dpi=150,
    )
    plt.close(fig)

    return face, nonface


# ----------------------------------------------------------------------
# Q3: N170 TIMING/AMPLITUDE + STATISTICAL TEST + SEARCHLIGHT
# ----------------------------------------------------------------------
def extract_n170_peaks(evoked_face_trials, evoked_nonface_trials,
                        channels=N170_CHANNELS, window=N170_WINDOW):
    """
    For each single-trial evoked object, find the negative peak
    (amplitude and latency) inside the N170 window, averaged over the
    ROI channels. Returns arrays of peak amplitudes/latencies for both
    conditions (one value per trial).
    """
    def _peak_stats(evoked_list):
        amps, lats = [], []
        for ev in evoked_list:
            roi_data = ev.copy().pick(channels).data.mean(axis=0)  # ROI avg
            times = ev.times
            mask = (times >= window[0]) & (times <= window[1])
            seg = roi_data[mask]
            seg_times = times[mask]
            # N170 is a negative deflection -> look for the minimum
            idx_min = np.argmin(seg)
            amps.append(seg[idx_min])
            lats.append(seg_times[idx_min])
        return np.array(amps), np.array(lats)

    face_amp, face_lat = _peak_stats(evoked_face_trials)
    nonface_amp, nonface_lat = _peak_stats(evoked_nonface_trials)
    return face_amp, face_lat, nonface_amp, nonface_lat


def searchlight_ttest(face_epochs, nonface_epochs, channels=N170_CHANNELS):
    """
    Compute an independent-samples t-statistic at EVERY time point,
    using the ROI-averaged single-trial data, to visualize when the
    face vs non-face difference is statistically significant
    ("searchlight" across time).
    """
    face_data = face_epochs.copy().pick(channels).get_data().mean(axis=1)     # (n_trials, n_times)
    nonface_data = nonface_epochs.copy().pick(channels).get_data().mean(axis=1)
    times = face_epochs.times

    t_vals, p_vals = stats.ttest_ind(face_data, nonface_data, axis=0)
    return times, t_vals, p_vals


def n170_analysis(epochs, subject_id, out_dir=RESULTS_DIR,
                   channels=N170_CHANNELS, window=N170_WINDOW):
    """
    Full N170 analysis for one subject:
      1. Peak amplitude/latency per trial (face & non-face).
      2. Independent t-test comparing amplitudes and latencies.
      3. Combined figure: ERP with CI (top) + searchlight t-stat (bottom).
      4. Save numeric results to a text file.
    """
    face_epochs, nonface_epochs = get_face_nonface(epochs)
    face_trials = list(face_epochs.iter_evoked())
    nonface_trials = list(nonface_epochs.iter_evoked())

    face_amp, face_lat, nonface_amp, nonface_lat = extract_n170_peaks(
        face_trials, nonface_trials, channels=channels, window=window
    )

    # --- Statistical tests on peak amplitude and latency ---
    t_amp, p_amp = stats.ttest_ind(face_amp, nonface_amp, equal_var=False)
    t_lat, p_lat = stats.ttest_ind(face_lat, nonface_lat, equal_var=False)

    # --- Searchlight across the whole epoch ---
    times, t_vals, p_vals = searchlight_ttest(face_epochs, nonface_epochs,
                                               channels=channels)
    sig_mask = p_vals < ALPHA

    # ---------------- PLOT ----------------
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    # Top: ERP comparison with CI (ROI-averaged)
    evokeds = {"Face": face_trials, "Non-face": nonface_trials}
    mne.viz.plot_compare_evokeds(
        evokeds, picks=channels, combine="mean", ci=0.95,
        axes=axes[0], show=False,
    )
    axes[0].axvspan(window[0], window[1], color="gray", alpha=0.2,
                     label="N170 window")
    axes[0].set_title(f"Subject {subject_id}: N170 ROI ({channels})")
    axes[0].legend()

    # Bottom: searchlight t-statistic across time
    axes[1].plot(times, t_vals, color="black", label="t-statistic")
    axes[1].fill_between(times, t_vals, 0, where=sig_mask,
                          color="red", alpha=0.4,
                          label=f"p < {ALPHA}")
    axes[1].axhline(0, color="gray", lw=0.5)
    axes[1].axvspan(window[0], window[1], color="gray", alpha=0.2)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("t-value")
    axes[1].set_title("Searchlight: Face vs Non-face (independent t-test per time point)")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"sub{subject_id}_n170_analysis.png"),
                dpi=150)
    plt.close(fig)

    # ---------------- SAVE NUMERIC RESULTS ----------------
    results_txt = os.path.join(out_dir, f"sub{subject_id}_n170_stats.txt")
    with open(results_txt, "w") as f:
        f.write(f"=== Subject {subject_id}: N170 statistics ===\n")
        f.write(f"ROI channels: {channels}\n")
        f.write(f"N170 window: {window[0]*1000:.0f}-{window[1]*1000:.0f} ms\n\n")

        f.write(f"Face   peak amplitude: mean={face_amp.mean():.3f} uV, "
                f"SD={face_amp.std():.3f}, n={len(face_amp)}\n")
        f.write(f"NonFace peak amplitude: mean={nonface_amp.mean():.3f} uV, "
                f"SD={nonface_amp.std():.3f}, n={len(nonface_amp)}\n")
        f.write(f"Amplitude t-test: t={t_amp:.3f}, p={p_amp:.5f} "
                f"({'significant' if p_amp < ALPHA else 'not significant'})\n\n")

        f.write(f"Face   peak latency: mean={face_lat.mean()*1000:.1f} ms, "
                f"SD={face_lat.std()*1000:.1f}\n")
        f.write(f"NonFace peak latency: mean={nonface_lat.mean()*1000:.1f} ms, "
                f"SD={nonface_lat.std()*1000:.1f}\n")
        f.write(f"Latency t-test: t={t_lat:.3f}, p={p_lat:.5f} "
                f"({'significant' if p_lat < ALPHA else 'not significant'})\n\n")

        sig_times = times[sig_mask]
        if sig_times.size > 0:
            f.write(f"Searchlight: significant time points (p<{ALPHA}): "
                    f"{sig_times.min()*1000:.0f}-{sig_times.max()*1000:.0f} ms "
                    f"({sig_times.size} of {len(times)} samples)\n")
        else:
            f.write(f"Searchlight: no time points reached p<{ALPHA}\n")

    return {
        "t_amp": t_amp, "p_amp": p_amp,
        "t_lat": t_lat, "p_lat": p_lat,
        "times": times, "t_vals": t_vals, "p_vals": p_vals,
    }


# ----------------------------------------------------------------------
# MAIN PIPELINE (runs for every subject)
# ----------------------------------------------------------------------
def run_erp_pipeline():
    for subject_id in SUBJECT_IDS:
        print(f"\n=== Processing subject {subject_id} ===")

        epochs = load_clean_epochs(subject_id)
        epochs = attach_labels(epochs)

        # Q1: ERP for all channels
        plot_all_channels_erp(epochs, subject_id)

        # Q2: Face vs non-face with CI
        compare_face_nonface(epochs, subject_id)

        # Q3: N170 timing/amplitude + stats + searchlight
        n170_analysis(epochs, subject_id)

        print(f"Subject {subject_id} done. Results saved in {RESULTS_DIR}")


if __name__ == "__main__":
    run_erp_pipeline()
