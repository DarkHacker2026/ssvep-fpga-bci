#!/usr/bin/env python3
"""
Generate replay-ready EEG text files for the Vivado live testbench.

The current simulation path replays a single preprocessed stream, one sample per
line, in 512-sample chunks. This utility:
1. loads the MATLAB SSVEP dataset,
2. applies the same software preprocessing used during offline training,
3. chooses a strong subject for a 15 Hz vs 20 Hz demo, and
4. writes three text files:
   - YES calibration
   - NO calibration
   - REAL/demo stream
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np

import bootstrap_network as bn


MAX_Q24 = (1 << 23) - 1
MIN_Q24 = -(1 << 23)


def parse_subject_id(path: Path) -> int:
    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else 0


def load_combined_trials(
    mat_path: Path,
    target_freq_hz: int,
    condition: str,
    channel_indices: Sequence[int],
    spatial_q16: Sequence[int],
) -> List[np.ndarray]:
    condition_indices = {"low": [0], "high": [1], "both": [0, 1]}[condition]
    chan_lo = min(channel_indices)
    chan_hi = max(channel_indices) + 1
    chan_offsets = [idx - chan_lo for idx in channel_indices]
    spatial = np.asarray(spatial_q16, dtype=np.float64) / float(bn.Q_SCALE)
    trials: List[np.ndarray] = []

    with h5py.File(str(mat_path), "r") as handle:
        tensor = handle["datas"] if "datas" in handle else handle["data"]
        n_trials = int(tensor.shape[0])
        freq_idx = int(target_freq_hz) - 1
        for trial_idx in range(n_trials):
            for cond_idx in condition_indices:
                epoch = np.asarray(
                    tensor[trial_idx, freq_idx, :, chan_lo:chan_hi, cond_idx],
                    dtype=np.float64,
                )[:, chan_offsets]
                epoch_250 = bn.downsample_epoch(epoch)
                car_epoch = epoch_250 - np.mean(epoch_250, axis=1, keepdims=True)
                filt_epoch = bn.apply_frontend_filters(car_epoch)
                combined = np.dot(filt_epoch, spatial)
                trials.append(combined.astype(np.float64))

    return trials


def trial_to_windows(trial: np.ndarray) -> List[np.ndarray]:
    windows = []
    for start in range(0, trial.shape[0] - bn.WINDOW_LEN + 1, bn.HOP_LEN):
        windows.append(np.array(trial[start:start + bn.WINDOW_LEN], dtype=np.float64, copy=True))
    return windows


def compute_class_scores(windows: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    if not windows:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)

    detector_freqs = [15, 30, 45, 20, 40, 60]
    coeff_sin, coeff_cos = bn.build_detector_taps(detector_freqs)
    stack = np.stack(windows, axis=0).astype(np.float64)
    sin_proj = (stack @ coeff_sin.T) / float(bn.COEFF_SCALE)
    cos_proj = (stack @ coeff_cos.T) / float(bn.COEFF_SCALE)
    mags = np.sqrt(sin_proj * sin_proj + cos_proj * cos_proj)
    yes_score = np.sum(mags[:, :3], axis=1)
    no_score = np.sum(mags[:, 3:], axis=1)
    return yes_score, no_score


def evaluate_subject(
    mat_path: Path,
    yes_freq: int,
    no_freq: int,
    condition: str,
    channel_indices: Sequence[int],
    spatial_q16: Sequence[int],
) -> Dict[str, float]:
    yes_trials = load_combined_trials(mat_path, yes_freq, condition, channel_indices, spatial_q16)
    no_trials = load_combined_trials(mat_path, no_freq, condition, channel_indices, spatial_q16)
    yes_trial_windows = [trial_to_windows(trial) for trial in yes_trials]
    no_trial_windows = [trial_to_windows(trial) for trial in no_trials]
    yes_windows = [window for trial in yes_trial_windows for window in trial]
    no_windows = [window for trial in no_trial_windows for window in trial]

    yes_yes_score, yes_no_score = compute_class_scores(yes_windows)
    no_yes_score, no_no_score = compute_class_scores(no_windows)
    yes_preds = yes_no_score > yes_yes_score
    no_preds = no_no_score > no_yes_score
    window_total = len(yes_preds) + len(no_preds)
    window_correct = int(np.sum(~yes_preds) + np.sum(no_preds))
    window_accuracy = float(window_correct / window_total) if window_total else 0.0

    def trial_majority(trial_windows: Sequence[np.ndarray], expected_class: int) -> int:
        yes_score, no_score = compute_class_scores(trial_windows)
        preds = (no_score > yes_score).astype(np.int32)
        return int(round(np.mean(preds))) == expected_class

    trial_correct = 0
    trial_total = 0
    for trial_windows in yes_trial_windows:
        if trial_windows:
            trial_correct += int(trial_majority(trial_windows, 0))
            trial_total += 1
    for trial_windows in no_trial_windows:
        if trial_windows:
            trial_correct += int(trial_majority(trial_windows, 1))
            trial_total += 1

    all_yes_score = np.concatenate([yes_yes_score, no_yes_score]) if window_total else np.empty((0,))
    all_no_score = np.concatenate([yes_no_score, no_no_score]) if window_total else np.empty((0,))
    margin = float(np.mean(np.abs(all_yes_score - all_no_score) / (all_yes_score + all_no_score + 1e-9))) if window_total else 0.0

    return {
        "subject_id": parse_subject_id(mat_path),
        "window_accuracy": window_accuracy,
        "trial_accuracy": float(trial_correct / trial_total) if trial_total else 0.0,
        "margin": margin,
        "yes_windows": len(yes_windows),
        "no_windows": len(no_windows),
    }


def choose_subject(
    mat_paths: Sequence[Path],
    yes_freq: int,
    no_freq: int,
    condition: str,
    channel_indices: Sequence[int],
    spatial_q16: Sequence[int],
    required_yes_windows: int,
    required_no_windows: int,
) -> Tuple[Path, List[Dict[str, float]]]:
    scored: List[Tuple[Path, Dict[str, float]]] = []
    for mat_path in sorted(mat_paths):
        metrics = evaluate_subject(mat_path, yes_freq, no_freq, condition, channel_indices, spatial_q16)
        if metrics["yes_windows"] >= required_yes_windows and metrics["no_windows"] >= required_no_windows:
            scored.append((mat_path, metrics))

    if not scored:
        raise RuntimeError("no subject has enough windows for the requested calibration/demo export")

    scored.sort(
        key=lambda item: (
            item[1]["trial_accuracy"],
            item[1]["window_accuracy"],
            item[1]["margin"],
            -item[1]["subject_id"],
        ),
        reverse=True,
    )
    return scored[0][0], [metrics for _, metrics in scored]


def quantize_windows(windows: Sequence[np.ndarray]) -> List[np.ndarray]:
    quantized = []
    for window in windows:
        ints = np.clip(np.round(window), MIN_Q24, MAX_Q24).astype(np.int32)
        quantized.append(ints)
    return quantized


def write_window_file(path: Path, windows: Sequence[np.ndarray]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for window in windows:
            for sample in window:
                handle.write(f"{int(sample)}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate replay EEG text files from the 1-60 Hz SSVEP dataset.")
    parser.add_argument("--mat-dir", type=Path, required=True, help="Directory containing data_sX_64.mat files")
    parser.add_argument("--subject", type=Path, help="Optional specific subject MAT file to export")
    parser.add_argument("--outdir", type=Path, default=Path("demo_replay_exports"))
    parser.add_argument("--condition", choices=["low", "high", "both"], default="high")
    parser.add_argument("--yes-freq", type=int, default=15)
    parser.add_argument("--no-freq", type=int, default=20)
    parser.add_argument("--channel-indices", default="56,58,60")
    parser.add_argument("--spatial-q16", default="-32768,65536,-32768")
    parser.add_argument("--cal-windows", type=int, default=16)
    parser.add_argument("--block-windows", type=int, default=4, help="Windows per class block in the real/demo stream")
    parser.add_argument(
        "--demo-pattern",
        default="NO,YES",
        help="Comma-separated class block order for the real/demo file, e.g. NO,YES or NO,YES,NO,YES",
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    channel_indices = bn.parse_int_list(args.channel_indices, 3, "channel-indices")
    spatial_q16 = bn.parse_int_list(args.spatial_q16, 3, "spatial-q16")
    mat_paths = sorted(args.mat_dir.glob("*.mat"))
    if not mat_paths:
        raise FileNotFoundError(f"no .mat files found in {args.mat_dir}")

    pattern = [token.strip().upper() for token in args.demo_pattern.split(",") if token.strip()]
    if any(token not in {"YES", "NO"} for token in pattern):
        raise ValueError("demo-pattern entries must be YES or NO")

    required_yes = args.cal_windows + pattern.count("YES") * args.block_windows
    required_no = args.cal_windows + pattern.count("NO") * args.block_windows

    chosen_subject = args.subject
    rankings: List[Dict[str, float]] = []
    if chosen_subject is None:
        chosen_subject, rankings = choose_subject(
            mat_paths=mat_paths,
            yes_freq=args.yes_freq,
            no_freq=args.no_freq,
            condition=args.condition,
            channel_indices=channel_indices,
            spatial_q16=spatial_q16,
            required_yes_windows=required_yes,
            required_no_windows=required_no,
        )
    else:
        chosen_subject = chosen_subject.resolve()

    yes_trials = load_combined_trials(chosen_subject, args.yes_freq, args.condition, channel_indices, spatial_q16)
    no_trials = load_combined_trials(chosen_subject, args.no_freq, args.condition, channel_indices, spatial_q16)
    yes_windows = quantize_windows([window for trial in yes_trials for window in trial_to_windows(trial)])
    no_windows = quantize_windows([window for trial in no_trials for window in trial_to_windows(trial)])

    if len(yes_windows) < required_yes or len(no_windows) < required_no:
        raise RuntimeError(
            f"subject {chosen_subject.name} does not have enough windows "
            f"(YES={len(yes_windows)}, NO={len(no_windows)} required YES={required_yes}, NO={required_no})"
        )

    yes_cal_windows = yes_windows[: args.cal_windows]
    no_cal_windows = no_windows[: args.cal_windows]
    yes_demo_pool = yes_windows[args.cal_windows:]
    no_demo_pool = no_windows[args.cal_windows:]

    demo_windows: List[np.ndarray] = []
    yes_idx = 0
    no_idx = 0
    for token in pattern:
        if token == "YES":
            demo_windows.extend(yes_demo_pool[yes_idx: yes_idx + args.block_windows])
            yes_idx += args.block_windows
        else:
            demo_windows.extend(no_demo_pool[no_idx: no_idx + args.block_windows])
            no_idx += args.block_windows

    yes_path = args.outdir / "yes_15hz_calibration.txt"
    no_path = args.outdir / "no_20hz_calibration.txt"
    demo_path = args.outdir / "demo_no_then_yes.txt"
    meta_path = args.outdir / "demo_replay_metadata.json"

    write_window_file(yes_path, yes_cal_windows)
    write_window_file(no_path, no_cal_windows)
    write_window_file(demo_path, demo_windows)

    yes_abs = np.concatenate([window.astype(np.float64) for window in yes_cal_windows]) if yes_cal_windows else np.empty((0,))
    no_abs = np.concatenate([window.astype(np.float64) for window in no_cal_windows]) if no_cal_windows else np.empty((0,))
    demo_abs = np.concatenate([window.astype(np.float64) for window in demo_windows]) if demo_windows else np.empty((0,))
    metadata = {
        "chosen_subject": chosen_subject.name,
        "condition": args.condition,
        "yes_frequency_hz": args.yes_freq,
        "no_frequency_hz": args.no_freq,
        "cal_windows_per_class": args.cal_windows,
        "demo_pattern": pattern,
        "block_windows": args.block_windows,
        "yes_file": str(yes_path.resolve()),
        "no_file": str(no_path.resolve()),
        "demo_file": str(demo_path.resolve()),
        "yes_sample_range": [int(np.min(yes_abs)) if yes_abs.size else 0, int(np.max(yes_abs)) if yes_abs.size else 0],
        "no_sample_range": [int(np.min(no_abs)) if no_abs.size else 0, int(np.max(no_abs)) if no_abs.size else 0],
        "demo_sample_range": [int(np.min(demo_abs)) if demo_abs.size else 0, int(np.max(demo_abs)) if demo_abs.size else 0],
        "subject_rankings": rankings[:10],
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[ok] chose subject: {chosen_subject.name}")
    print(f"[ok] wrote YES calibration: {yes_path}")
    print(f"[ok] wrote NO calibration:  {no_path}")
    print(f"[ok] wrote demo stream:    {demo_path}")
    print(f"[ok] wrote metadata:       {meta_path}")


if __name__ == "__main__":
    main()
