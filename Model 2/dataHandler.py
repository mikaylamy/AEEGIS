import re
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import mne
from typing import List, Tuple
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix, accuracy_score
import warnings
from sklearn.model_selection import train_test_split
import os
import csv
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from scipy.signal import iirnotch, butter, filtfilt
import torch
from torch.utils.data import DataLoader, TensorDataset
import time

np.set_printoptions(threshold=np.inf)

# ── ADS1299 hardware constant ──────────────────────────────────────────────
# Set this to your actual output sample rate after SPI decimation.
# ADS1299 at DR=110 (500 SPS) decimated by 2 → 250 SPS.
# ADS1299 at DR=101 (250 SPS native) → 250 SPS.
ACQUISITION_SFREQ: int = 256  # .ftr dataset was recorded at 256 Hz
# ── Dataset paths ──────────────────────────────────────────────────────────
# Update these to match your Codespace paths
FTR_DATASET_PATH   = '/Users/mikayla/Downloads/archive/seizure_256Hz_dataset'
SEIZURE_EVENTS_CSV = '/Users/mikayla/Downloads/archive/seizure_256Hz_dataset/seizure_events.csv'


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — SEIZURE ANNOTATION LOADING
# Replaces parse_summary_file() entirely.
# Reads seizure_events.csv instead of chbXX-summary.txt files.
# ══════════════════════════════════════════════════════════════════════════════

def parse_seizure_events_csv(csv_path: str = SEIZURE_EVENTS_CSV) -> dict:
    """
    Replaces parse_summary_file().

    Reads seizure_events.csv (columns: series_id, onset, offset) and returns
    a dict structured identically to what the original summary parser returned,
    so all downstream functions work without any changes.

    Returns:
        dict: {
            'chb01_03.edf': [{'start_seconds': 2996, 'end_seconds': 3036}],
            'chb01_04.edf': [{'start_seconds': 1467, 'end_seconds': 1494}],
            ...
        }
    """
    df = pd.read_csv(csv_path)
    seizure_dict = {}
    for _, row in df.iterrows():
        fname = str(row['series_id'])        # e.g. 'chb01_03.edf'
        onset  = int(row['onset'])
        offset = int(row['offset'])
        if fname not in seizure_dict:
            seizure_dict[fname] = []
        seizure_dict[fname].append({
            'start_seconds': onset,
            'end_seconds':   offset
        })
    return seizure_dict


def build_file_records_from_ftr(subject_id: str,
                                 seizure_dict: dict,
                                 ftr_path: str = FTR_DATASET_PATH) -> list:
    """
    Replaces the EDF-based file record builder.

    For a given subject (e.g. 'chb01'), scans the .ftr dataset folder for
    all files belonging to that subject, reads their shape to determine
    duration, and attaches seizure annotations from seizure_dict.

    Returns a list of dicts matching the structure expected by all downstream
    functions:
        [
          {
            'file_name':           'chb01_03.edf',   # kept as .edf for compatibility
            'ftr_file_name':       'chb01_03.ftr',   # actual file on disk
            'start_time_of_file':  datetime(...),
            'end_time_of_file':    datetime(...),
            'seizures': [{'start_seconds': int, 'end_seconds': int}, ...]
          },
          ...
        ]

    NOTE: Because .ftr files have no embedded timestamp, we assign synthetic
    absolute times by ordering files alphabetically and placing them
    consecutively with no gap between them. This is consistent with how
    CHB-MIT files were originally recorded (continuous within a subject session).
    """
    import glob

    pattern = os.path.join(ftr_path, f'{subject_id}_*.ftr')
    ftr_files = sorted(glob.glob(pattern))

    if not ftr_files:
        raise FileNotFoundError(
            f"No .ftr files found for subject '{subject_id}' in {ftr_path}.\n"
            f"Expected pattern: {pattern}"
        )

    records = []
    # Synthetic base time — actual date does not matter, only relative ordering
    current_time = datetime(2000, 1, 1, 0, 0, 0)

    for ftr_file in ftr_files:
        ftr_name = os.path.basename(ftr_file)                    # chb01_03.ftr
        edf_name = ftr_name.replace('.ftr', '.edf')              # chb01_03.edf

        # Read only the index to get row count (avoids loading full data into RAM)
        df_meta = pd.read_feather(ftr_file, columns=['FP1-F7'])  # any single channel
        n_samples = len(df_meta)
        duration_sec = n_samples / ACQUISITION_SFREQ             # e.g. 921600/256 = 3600s

        start_time = current_time
        end_time   = current_time + timedelta(seconds=duration_sec)

        # Attach seizures for this file (if any)
        seizures = seizure_dict.get(edf_name, [])

        records.append({
            'file_name':          edf_name,    # kept as .edf key for downstream compat
            'ftr_file_name':      ftr_name,    # actual file on disk
            'start_time_of_file': start_time,
            'end_time_of_file':   end_time,
            'seizures':           seizures
        })

        current_time = end_time  # next file starts immediately after

    return records


def get_all_subjects(ftr_path: str = FTR_DATASET_PATH) -> list:
    """
    Scan the .ftr dataset folder and return a sorted list of subject IDs.
    e.g. ['chb01', 'chb02', 'chb03', ...]
    """
    import glob
    files = glob.glob(os.path.join(ftr_path, 'chb*.ftr'))
    subjects = sorted(set(
        os.path.basename(f).split('_')[0] for f in files   # 'chb01_03.ftr' -> 'chb01'
    ))
    return subjects


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — ORIGINAL parse_summary_file (KEPT FOR REFERENCE, NOT USED)
# ══════════════════════════════════════════════════════════════════════════════

def parse_summary_file(summary_path):
    """
    ORIGINAL function — kept for reference only.
    NOT called anywhere in the pipeline when using .ftr files.
    Use parse_seizure_events_csv() + build_file_records_from_ftr() instead.
    """
    records = []
    current_record = {}
    last_ts = None

    def _parse_hms(tstr, base_day=1):
        parts = tstr.split(":")
        h = int(parts[0])
        m = int(parts[1])
        s = int(parts[2]) if len(parts) == 3 else 0
        dt = datetime(2000, 1, base_day, 0, 0, 0) \
             + timedelta(hours=h, minutes=m, seconds=s)
        return dt

    with open(summary_path, "r") as f:
        day = 1
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("File Name:"):
                if current_record:
                    records.append(current_record)
                current_record = {
                    "file_name": line.split("File Name:")[1].strip(),
                    "start_time_of_file": None,
                    "end_time_of_file": None,
                    "seizures": []
                }
            elif line.startswith("File Start Time:"):
                tstr = line.split("File Start Time:")[1].strip()
                ts = _parse_hms(tstr, base_day=day)
                if last_ts is not None and ts <= last_ts:
                    day += 1
                    ts = _parse_hms(tstr, base_day=day)
                current_record["start_time_of_file"] = ts
                last_ts = ts
            elif line.startswith("File End Time:"):
                tstr = line.split("File End Time:")[1].strip()
                te = _parse_hms(tstr, base_day=day)
                if te < current_record["start_time_of_file"]:
                    day += 1
                    te = _parse_hms(tstr, base_day=day)
                current_record["end_time_of_file"] = te
                last_ts = te
            elif "Seizure" in line and "Start Time:" in line:
                m = re.search(r"(\d+)\s*seconds", line)
                if m:
                    current_record["seizures"].append({
                        "start_seconds": int(m.group(1)),
                        "end_seconds": None
                    })
            elif "Seizure" in line and "End Time:" in line:
                m = re.search(r"(\d+)\s*seconds", line)
                if m and current_record["seizures"]:
                    current_record["seizures"][-1]["end_seconds"] = int(m.group(1))
        if current_record:
            records.append(current_record)

    return records


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PREICTAL / INTERICTAL EXTRACTION (UNCHANGED)
# ══════════════════════════════════════════════════════════════════════════════

def extract_preictal_segments_distance(
    file_records,
    preictal_duration_min: float = 30,
    postictal_duration_min: float = 30,
    min_length_min: float = 20
):
    P = preictal_duration_min * 60
    Q = postictal_duration_min * 60
    min_len = min_length_min * 60

    file_records = sorted(file_records, key=lambda r: r["start_time_of_file"])
    for r in file_records:
        r["duration"] = (r["end_time_of_file"] - r["start_time_of_file"]).total_seconds()

    seizures = []
    for rec in file_records:
        base = rec["start_time_of_file"]
        for seiz in rec.get("seizures", []):
            abs_start = base + timedelta(seconds=seiz["start_seconds"])
            abs_end   = base + timedelta(seconds=seiz["end_seconds"])
            seizures.append({
                "rec": rec,
                "start_sec": seiz["start_seconds"],
                "end_sec":   seiz["end_seconds"],
                "abs_start": abs_start,
                "abs_end":   abs_end
            })
    seizures.sort(key=lambda s: s["abs_start"])

    kept = []
    last_post_end = datetime.min
    for s in seizures:
        pre_start_abs = s["abs_start"] - timedelta(seconds=P)
        if pre_start_abs < last_post_end + timedelta(seconds=Q):
            continue
        kept.append(s)
        last_post_end = s["abs_end"]

    out = []
    for s in kept:
        rec = s["rec"]
        t0  = s["start_sec"]
        if t0 >= P:
            seg = {"file": rec["file_name"], "start": t0 - P, "end": t0}
        else:
            idx = file_records.index(rec)
            if idx == 0:
                seg = {"file": rec["file_name"], "start": 0, "end": t0}
            else:
                prev = file_records[idx-1]
                gap  = (rec["start_time_of_file"] - prev["end_time_of_file"]).total_seconds()
                needed_prev = P - t0 - gap
                if needed_prev <= 0:
                    seg = {"file": rec["file_name"], "start": 0, "end": t0}
                else:
                    seg = (
                        {
                          "file": prev["file_name"],
                          "start": max(prev["duration"] - needed_prev, 0),
                          "end":   prev["duration"]
                        },
                        {
                          "file": rec["file_name"],
                          "start": 0,
                          "end":   t0
                        }
                    )
        length = (seg["end"] - seg["start"]) if isinstance(seg, dict) \
                 else sum(piece["end"] - piece["start"] for piece in seg)
        if length >= min_len:
            out.append(seg)

    return out


def helper_absolute_time(file_records):
    records_sorted = sorted(file_records, key=lambda rec: rec["start_time_of_file"])
    base_time = records_sorted[0]["start_time_of_file"]
    abs_times = {}
    for rec in records_sorted:
        dur = (rec["end_time_of_file"] - rec["start_time_of_file"]).total_seconds()
        abs_start = (rec["start_time_of_file"] - base_time).total_seconds()
        abs_end = abs_start + dur
        abs_times[rec["file_name"]] = {"abs_start": abs_start, "abs_end": abs_end}
    return abs_times


def chunk_absolute_times(records, tn_chunk):
    rec_map = {r["file_name"]: r for r in records}

    def _abs_times(seg):
        rec = rec_map[seg["file"]]
        base = rec["start_time_of_file"]
        start_s = round(seg["start"])
        end_s   = round(seg["end"])
        return (base + timedelta(seconds=start_s), base + timedelta(seconds=end_s))

    interictal_list, preictal_chunk = tn_chunk
    interictal_times = [_abs_times(seg) for seg in interictal_list]

    if isinstance(preictal_chunk, tuple):
        preictal_segs = list(preictal_chunk)
    else:
        preictal_segs = [preictal_chunk]
    preictal_times = [_abs_times(seg) for seg in preictal_segs]

    return {"interictal": interictal_times, "preictal": preictal_times}


def extract_interictal_data(file_records,
                            preictal_duration_sec=30*60,
                            interictal_buffer_sec=180*60,
                            postictal_duration_min=30):
    file_records = sorted(file_records, key=lambda rec: rec["start_time_of_file"])
    base_time = file_records[0]["start_time_of_file"]

    for rec in file_records:
        rec["duration"] = (rec["end_time_of_file"] - rec["start_time_of_file"]).total_seconds()
        rec["abs_start"] = (rec["start_time_of_file"] - base_time).total_seconds()
        rec["abs_end"] = rec["abs_start"] + rec["duration"]

    postictal_duration_sec = postictal_duration_min * 60

    unsafe_intervals = []
    for rec in file_records:
        file_abs_start = rec["abs_start"]
        for seizure in rec.get("seizures", []):
            seizure_abs_start = file_abs_start + seizure["start_seconds"]
            seizure_abs_end   = file_abs_start + seizure["end_seconds"]
            unsafe_start = seizure_abs_start - (preictal_duration_sec + interictal_buffer_sec)
            if unsafe_start < 0:
                unsafe_start = 0
            unsafe_end = seizure_abs_end + postictal_duration_sec
            unsafe_intervals.append((unsafe_start, unsafe_end))

    unsafe_intervals.sort(key=lambda interval: interval[0])
    merged_unsafe = []
    for interval in unsafe_intervals:
        if not merged_unsafe:
            merged_unsafe.append(interval)
        else:
            last = merged_unsafe[-1]
            if interval[0] <= last[1]:
                merged_unsafe[-1] = (last[0], max(last[1], interval[1]))
            else:
                merged_unsafe.append(interval)

    subject_total_duration = file_records[-1]["abs_end"]
    safe_intervals = []
    prev_end = 0
    for (us, ue) in merged_unsafe:
        if us > prev_end:
            safe_intervals.append((prev_end, us))
        prev_end = max(prev_end, ue)
    if prev_end < subject_total_duration:
        safe_intervals.append((prev_end, subject_total_duration))

    result = {rec["file_name"]: [] for rec in file_records}
    for safe_start, safe_end in safe_intervals:
        for rec in file_records:
            file_abs_start = rec["abs_start"]
            file_abs_end   = rec["abs_end"]
            if safe_end <= file_abs_start or safe_start >= file_abs_end:
                continue
            overlap_start_abs = max(file_abs_start, safe_start)
            overlap_end_abs   = min(file_abs_end,   safe_end)
            local_start = overlap_start_abs - file_abs_start
            local_end   = overlap_end_abs   - file_abs_start
            if local_start < local_end:
                result[rec["file_name"]].append({"start": local_start, "end": local_end})

    return result


def flatten_interictal(interictal_dict):
    files = sorted(interictal_dict.keys())
    flattened = []
    global_time = 0
    for f in files:
        for seg in interictal_dict[f]:
            seg_length = seg["end"] - seg["start"]
            flattened.append({
                "file":         f,
                "local_start":  seg["start"],
                "local_end":    seg["end"],
                "length":       seg_length,
                "global_start": global_time,
                "global_end":   global_time + seg_length
            })
            global_time += seg_length
    return flattened, global_time


def partition_interictal(flattened, total_length, n):
    target = total_length / n
    partitions = []
    current_partition = []
    current_pos = 0
    for seg in flattened:
        seg_remaining = seg["length"]
        local_offset  = seg["local_start"]
        while seg_remaining > 0:
            needed = target - current_pos
            if seg_remaining <= needed + 1e-6:
                current_partition.append({
                    "file":  seg["file"],
                    "start": local_offset,
                    "end":   seg["local_end"]
                })
                current_pos   += seg_remaining
                seg_remaining  = 0
            else:
                piece_end = local_offset + needed
                current_partition.append({
                    "file":  seg["file"],
                    "start": local_offset,
                    "end":   piece_end
                })
                local_offset  += needed
                seg_remaining -= needed
                current_pos   += needed
            if current_pos >= target - 1e-6:
                partitions.append(current_partition)
                current_partition = []
                current_pos = 0
                if len(partitions) == n - 1:
                    break
        if len(partitions) == n - 1:
            break

    remaining = []
    if seg_remaining > 0:
        remaining.append({"file": seg["file"], "start": local_offset, "end": seg["local_end"]})
    idx = flattened.index(seg) + 1
    for s in flattened[idx:]:
        remaining.append({"file": s["file"], "start": s["local_start"], "end": s["local_end"]})
    partitions.append(current_partition + remaining)
    return partitions


def combine_interictal_preictal(interictal_dict, preictal_list):
    flattened, total_time = flatten_interictal(interictal_dict)
    n = len(preictal_list)
    if n == 0:
        raise ValueError("No preictal segments provided.")
    partitions = partition_interictal(flattened, total_time, n)
    preictal_shuffled = preictal_list.copy()
    random.shuffle(preictal_shuffled)
    pairs = list(zip(partitions, preictal_shuffled))
    return pairs


def generate_loocv_splits(pairs):
    splits = []
    n = len(pairs)
    for i in range(n):
        test_pair  = pairs[i]
        train_pairs = pairs[:i] + pairs[i+1:]
        splits.append((train_pairs, test_pair))
    return splits


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — DATA EXTRACTION
# extract_segment_data() is the ONLY function that reads from disk.
# Replaced to read .ftr files instead of .edf files.
# Everything else in the pipeline is unchanged.
# ══════════════════════════════════════════════════════════════════════════════

# Cache loaded DataFrames to avoid re-reading the same file multiple times
# during a training run (each .ftr file is ~44 MB, fitting comfortably in RAM)
_ftr_cache: dict = {}

def _load_ftr(ftr_path: str) -> pd.DataFrame:
    """Load a .ftr file with caching. Returns DataFrame with EEG columns only."""
    if ftr_path not in _ftr_cache:
        df = pd.read_feather(ftr_path)
        # Drop metadata columns, keep only EEG channels (float16 columns)
        eeg_cols = [c for c in df.columns if c not in ('series_id', 'p_id')]
        _ftr_cache[ftr_path] = df[eeg_cols]
    return _ftr_cache[ftr_path]


def extract_segment_data(segment, base_path, present_channels):
    """
    Drop-in replacement for the original EDF-based extract_segment_data().

    Reads a .ftr (Feather) file instead of an .edf file.
    Returns np.ndarray of shape (n_channels, n_samples), float32, in µV.

    Parameters:
        segment:          dict with keys 'file' (e.g. 'chb01_03.edf'),
                          'start' (seconds), 'end' (seconds)
        base_path:        path to the .ftr dataset folder (FTR_DATASET_PATH)
        present_channels: list of EEG channel name strings to extract
    """
    # Convert .edf filename → .ftr filename
    edf_name = segment["file"]                          # e.g. 'chb01_03.edf'
    ftr_name = edf_name.replace('.edf', '.ftr')         # e.g. 'chb01_03.ftr'
    ftr_path = os.path.join(str(base_path), ftr_name)

    if not os.path.exists(ftr_path):
        raise FileNotFoundError(
            f"Expected .ftr file not found: {ftr_path}\n"
            f"Check that FTR_DATASET_PATH is correct: {base_path}"
        )

    df = _load_ftr(ftr_path)

    # Select only the channels present in all recordings for this subject
    available = [c for c in present_channels if c in df.columns]
    if not available:
        raise ValueError(
            f"None of the requested channels {present_channels} "
            f"found in {ftr_name}. Available: {df.columns.tolist()}"
        )

    # Slice the time window (start/end are in seconds)
    start_sample = int(segment["start"] * ACQUISITION_SFREQ)
    end_sample   = int(segment["end"]   * ACQUISITION_SFREQ)

    # Shape: (n_samples, n_channels) → transpose → (n_channels, n_samples)
    data = df[available].iloc[start_sample:end_sample].values.T

    # Cast float16 → float32 (required by PyTorch and scipy filters)
    data = data.astype(np.float32)

    # Apply the same preprocessing chain as the original pipeline
    data = preprocess_eeg(data, fs=float(ACQUISITION_SFREQ))

    return data


def get_subject_channels(records, base_path):
    """
    Returns the list of EEG channels present in ALL .ftr files for this subject.
    Replaces the original EDF-based version.

    Excludes metadata columns ('series_id', 'p_id').
    """
    METADATA_COLS = {'series_id', 'p_id'}
    all_sets = []

    for rec in records:
        ftr_name = rec.get('ftr_file_name') or rec['file_name'].replace('.edf', '.ftr')
        ftr_path = os.path.join(str(base_path), ftr_name)

        if not os.path.exists(ftr_path):
            continue

        # Read only column names — do not load data
        df_cols = pd.read_feather(ftr_path, columns=None).columns.tolist()
        valid = {c for c in df_cols if c not in METADATA_COLS}
        all_sets.append(valid)

    if not all_sets:
        return []

    common = set.intersection(*all_sets)
    return sorted(common)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — PREPROCESSING (UNCHANGED, with ADS1299 guard)
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_eeg(
    sig: np.ndarray,
    fs: float,
    notch_freqs=(60.0, 120.0),
    notch_Q=30.0,
    highpass_freq: float = 0.5,
    bandpass: bool = False,
    bp_low: float = 0.5,
    bp_high: float = 60.0,
    bp_order: int = 4,
) -> np.ndarray:
    """
    Clean multi-channel EEG (n_ch × n_samples):
      • Notch at each frequency in notch_freqs
      • Then either:
         – band-pass between bp_low and bp_high  (if bandpass=True)
         – or high-pass at highpass_freq          (if bandpass=False)
    """
    # ADS1299 guard: reject implausible sample rates
    assert 100 <= fs <= 1000, (
        f"preprocess_eeg received fs={fs}. "
        f"Expected 250 or 256 from your dataset. "
        f"Check ACQUISITION_SFREQ at the top of dataHandler.py."
    )

    filtered = sig.copy()

    # 1) Notch filters
    for f0 in notch_freqs:
        b_n, a_n = iirnotch(f0, notch_Q, fs)
        filtered = filtfilt(b_n, a_n, filtered, axis=-1)

    # 2) Band-pass or high-pass
    nyq = fs / 2
    if bandpass:
        low  = bp_low  / nyq
        high = bp_high / nyq
        b_bp, a_bp = butter(bp_order, [low, high], btype='bandpass')
        filtered = filtfilt(b_bp, a_bp, filtered, axis=-1)
    else:
        wn = highpass_freq / nyq
        b_hp, a_hp = butter(bp_order, wn, btype='highpass')
        filtered = filtfilt(b_hp, a_hp, filtered, axis=-1)

    return filtered


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — COMPOSITE EXTRACTION AND WINDOWING (UNCHANGED)
# ══════════════════════════════════════════════════════════════════════════════

def extract_composite_data(chunks, base_path, present_channels):
    data_list = []
    for seg in chunks:
        seg_data = extract_segment_data(seg, base_path, present_channels)
        data_list.append(seg_data)
    if data_list:
        return np.concatenate(data_list, axis=1)
    else:
        return None


def process_tn_chunks(tn_chunks, base_path, present_channels):
    interictal_data_list = []
    preictal_data_list   = []

    for pair in tn_chunks:
        interictal_chunks, preictal_chunk = pair
        inter_data = extract_composite_data(interictal_chunks, base_path, present_channels)
        interictal_data_list.append(inter_data)

        if isinstance(preictal_chunk, tuple):
            preictal_segments = list(preictal_chunk)
        else:
            preictal_segments = [preictal_chunk]

        pre_data = extract_composite_data(preictal_segments, base_path, present_channels)
        preictal_data_list.append(pre_data)

    return interictal_data_list, preictal_data_list


def sliding_windows(data: np.ndarray,
                    window_size: int,
                    stride: int) -> List[np.ndarray]:
    n_samples = data.shape[1]
    windows = []
    for start in range(0, n_samples - window_size + 1, stride):
        end = start + window_size
        windows.append(data[:, start:end])
    return windows


def split_nonoverlap(data: np.ndarray,
                     window_size: int) -> List[np.ndarray]:
    n_samples = data.shape[1]
    n_full = n_samples // window_size
    return [data[:, i*window_size:(i+1)*window_size] for i in range(n_full)]


def _build_window_times(segments, file_abs_starts, window_len):
    durs = [seg["end"] - seg["start"] for seg in segments]
    cum  = np.cumsum([0.0] + durs)
    total = cum[-1]
    n_win = int(total // window_len)
    times = np.empty(n_win, dtype=float)
    for i in range(n_win):
        pos = i * window_len
        k   = np.searchsorted(cum, pos, side="right") - 1
        offset = pos - cum[k]
        seg = segments[k]
        times[i] = file_abs_starts[seg["file"]] + seg["start"] + offset
    return times


def balance_one_pair(inter: np.ndarray,
                     pre: np.ndarray,
                     window_len: float,
                     sampling_rate: float = 0.25
                     ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    window_len = int(window_len * ACQUISITION_SFREQ)
    stride = int(window_len * sampling_rate)
    pre_windows   = sliding_windows(pre,   window_len, stride)
    inter_windows = split_nonoverlap(inter, window_len)
    M = len(pre_windows)
    if len(inter_windows) > M:
        inter_windows = random.sample(inter_windows, M)
    return inter_windows, pre_windows


def make_balance_dataset(train_pairs, window_len, sampling_rate):
    balanced = []
    for inter_data, pre_data in train_pairs:
        i_wins, p_wins = balance_one_pair(inter_data, pre_data,
                                          window_len=window_len,
                                          sampling_rate=sampling_rate)
        balanced.append((i_wins, p_wins))
    return balanced


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — DATASET AND DATALOADER CLASSES (UNCHANGED)
# ══════════════════════════════════════════════════════════════════════════════

class WindowDataset(Dataset):
    def __init__(self, windows: List[np.ndarray], labels: List[int]):
        self.windows = windows
        self.labels  = labels

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = torch.from_numpy(self.windows[idx]).float()
        y = self.labels[idx]
        return w, y


def make_train_val_loaders(dataset, window_len, val_frac, batch_size):
    window_len = int(window_len * ACQUISITION_SFREQ)
    inter, pre = [], []
    for inter_data, pre_data in dataset:
        i_wins = split_nonoverlap(inter_data, window_len)
        p_wins = split_nonoverlap(pre_data,   window_len)
        inter.extend(i_wins)
        pre.extend(p_wins)

    n_i, n_p = len(inter), len(pre)
    si = int(n_i * (1 - val_frac))
    sp = int(n_p * (1 - val_frac))

    i_train, i_val = inter[:si], inter[si:]
    p_train, p_val = pre[:sp],   pre[sp:]

    w_train = i_train + p_train
    y_train = [0]*len(i_train) + [1]*len(p_train)
    w_val   = i_val   + p_val
    y_val   = [0]*len(i_val)   + [1]*len(p_val)

    train_ds = WindowDataset(w_train, y_train)
    val_ds   = WindowDataset(w_val,   y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    return train_loader, val_loader


def make_train_val_loaders_last_balanced(balanced_dataset, val_frac, batch_size):
    inter, pre = [], []
    for i_wins, p_wins in balanced_dataset:
        inter.extend(i_wins)
        pre.extend(p_wins)

    n_i, n_p = len(inter), len(pre)
    si = int(n_i * (1 - val_frac))
    sp = int(n_p * (1 - val_frac))

    i_train, i_val = inter[:si], inter[si:]
    p_train, p_val = pre[:sp],   pre[sp:]

    w_train = i_train + p_train
    y_train = [0]*len(i_train) + [1]*len(p_train)
    w_val   = i_val   + p_val
    y_val   = [0]*len(i_val)   + [1]*len(p_val)

    train_ds = WindowDataset(w_train, y_train)
    val_ds   = WindowDataset(w_val,   y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    return train_loader, val_loader


def make_train_val_loaders_random_val(dataset, window_len, val_frac, batch_size, random_state=42):
    window_len = int(window_len * ACQUISITION_SFREQ)
    inter, pre = [], []
    for inter_data, pre_data in dataset:
        i_wins = split_nonoverlap(inter_data, window_len)
        p_wins = split_nonoverlap(pre_data,   window_len)
        inter.extend(i_wins)
        pre.extend(p_wins)

    X = inter + pre
    y = [0] * len(inter) + [1] * len(pre)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_frac, stratify=y, random_state=random_state
    )

    train_ds = WindowDataset(X_tr, y_tr)
    val_ds   = WindowDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    return train_loader, val_loader


def make_test_loader(test_pair, window_len, batch_size):
    window_len = int(window_len * ACQUISITION_SFREQ)

    inter_data, pre_data = test_pair
    i_wins = split_nonoverlap(inter_data, window_len)
    p_wins = split_nonoverlap(pre_data,   window_len)
    X_test = np.stack(i_wins + p_wins, axis=0)
    y_test = np.array([0]*len(i_wins) + [1]*len(p_wins), dtype=np.int64)

    test_ds     = TensorDataset(torch.from_numpy(X_test).float(),
                                torch.from_numpy(y_test))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)
    return test_loader