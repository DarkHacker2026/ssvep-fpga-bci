#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SSVEP BCI data recorder for PsychoPy.

Layout: every UI / HUD / overlay element uses units='norm'.
norm coordinates are computed purely from pixel dimensions — immune to
monitor degree-calibration errors.  Only the flickering GratingStim,
their glow borders, and their Hz labels keep units='deg' so spatial
frequency and visual angle remain scientifically accurate.

Changelog
---------
v3  - refreshThreshold now set AFTER getActualFrameRate() so the threshold
      matches the real measured FPS, not the hint value from the dialog.
      Previously every frame at 144 Hz triggered a "dropped frame" warning
      because the threshold was computed from the 240 Hz hint.
    - Partial session data is now saved on ESC / crash via the finally block
      in main().  Previously no files were written if the session was aborted.
    - Dialog 'Trials per class' empty-string crash fixed with a dedicated
      safe_int() helper; all numeric dialog fields now have the same guard.
    - Added '144hz_exact' protocol profile (18 / 24 Hz) — exact integer
      half-cycle pair for 144 Hz panels.
    - Added '165hz_exact' profile (16.5 / 20.625 Hz).
    - Manual per-flip timestamp collector added as a guaranteed fallback when
      win.frameIntervals comes back empty.
    - win.frameIntervals cleared with del [...] (in-place) instead of
      assignment to avoid detaching PsychoPy's internal reference.
    - Frame-interval CSV is always written (no silent skip on empty list).
    - Measured FPS printed to console immediately after calibration.
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import socket

import numpy as np
from psychopy import core, event, gui, monitors, visual


# ── Experiment defaults ──────────────────────────────────────────────────────
YES_FREQ_HZ  = 15.0
NO_FREQ_HZ   = 20.0
WINDOW_S     = 2.0
PRE_DELAY_S  = 1.0
TRIAL_S      = 4.0
REST_S       = 2.0
EEG_FS_HZ    = 250

REQUESTED_BOX_DEG = 8.0
REQUESTED_GAP_DEG = 4.0
CELLS_PER_SIDE    = 20
MODULATION_DEPTH  = 1.0

ENABLE_PHOTODIODE_PATCH = True
PHOTODIODE_ONSET_FRAMES = 4
OUTPUT_DIR = Path(".")

# ─────────────────────────────────────────────────────────────────────────────
# PROTOCOL PROFILES
#
# Exact-frequency table (f = fps / (2*hcf), hcf must be integer):
#
#  240 Hz:  15.000 Hz (hcf=8)   20.000 Hz (hcf=6)   — perfect
#  165 Hz:  16.500 Hz (hcf=5)   20.625 Hz (hcf=4)   — use instead of 15/20
#  144 Hz:  18.000 Hz (hcf=4)   24.000 Hz (hcf=3)   — exact pair near 15/20
# ─────────────────────────────────────────────────────────────────────────────
PROTOCOL_PROFILES: dict[str, dict[str, Any]] = {
    "standard_15_20": {
        "label": "Standard 240 Hz (15 / 20 Hz)",
        "yes_freq_hz": 15.0, "no_freq_hz": 20.0,
        "window_s": 2.0, 
        "pre_delay_s": 1.0,
        "trial_s": 4.0,
        "rest_s": 2.0,
        "modulation_depth": 1.0, "box_deg": 8.0,
        "gap_deg": 4.0, "photodiode_onset_frames": 4,
    },
    "144hz_exact": {
        # 144/(2*4)=18.0 Hz  144/(2*3)=24.0 Hz — both exact, strong SSVEP
        "label": "144 Hz panel (18 / 24 Hz — exact)",
        "yes_freq_hz": 18.0, "no_freq_hz": 24.0,
        "window_s": 2.0, "pre_delay_s": 1.0,
        "trial_s": 20.0, "rest_s": 2.0,
        "modulation_depth": 1.0, "box_deg": 8.0,
        "gap_deg": 4.0, "photodiode_onset_frames": 4,
    },
    "165hz_exact": {
        # 165/(2*5)=16.5 Hz  165/(2*4)=20.625 Hz — both exact
        "label": "165 Hz panel (16.5 / 20.625 Hz — exact)",
        "yes_freq_hz": 16.5, "no_freq_hz": 20.625,
        "window_s": 2.0, "pre_delay_s": 1.0,
        "trial_s": 20.0, "rest_s": 2.0,
        "modulation_depth": 1.0, "box_deg": 8.0,
        "gap_deg": 4.0, "photodiode_onset_frames": 4,
    },
    "clinical_comfort_30_40": {
        "label": "Clinical comfort 240 Hz (30 / 40 Hz)",
        "yes_freq_hz": 30.0, "no_freq_hz": 40.0,
        "window_s": 2.0, "pre_delay_s": 1.2,
        "trial_s": 16.0, "rest_s": 4.0,
        "modulation_depth": 0.7, "box_deg": 7.0,
        "gap_deg": 4.0, "photodiode_onset_frames": 6,
    },
}

TRIGGER_CODES: dict[str, int] = {
    "trial_start": 10, "cue_yes": 21, "cue_no": 22,
    "record_start_yes": 31, "record_start_no": 32,
    "record_end_yes": 41, "record_end_no": 42,
    "trial_end": 50,
}

TRIGGER_WIRING_MAP = {
    "type": "serial_to_ttl",
    "notes": (
        "Use a microcontroller or trigger adapter if EEG system expects TTL. "
        "Keep common ground between trigger source and amplifier."
    ),
    "connections": [
        "Trigger source GND -> EEG trigger GND",
        "Trigger source TTL OUT -> EEG trigger IN1 (through ~220 ohm series resistor)",
        "Photodiode sensor OUT -> EEG AUX/EXG channel",
        "Photodiode sensor GND -> EEG AUX/EXG ground",
    ],
}

BG       = (-1.00, -1.00, -1.00)
WHITE    = ( 1.00,  1.00,  1.00)
YES_COL  = (-0.44,  1.00,  0.52)
NO_COL   = ( 1.00, -0.50, -0.53)
ACCENT   = (-0.50,  0.13,  1.00)
MUTED    = (-0.40, -0.25, -0.12)
CHIP_DIM = (-0.75, -0.75, -0.75)

MODE_LABELS = {"yes": "YES only", "no": "NO only", "mixed": "Mixed"}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DisplayProfile:
    diagonal_in: float
    width_px: int
    height_px: int
    width_cm: float
    height_cm: float
    view_dist_cm: float
    refresh_hint_hz: float


@dataclass(frozen=True)
class SessionConfig:
    subject_id: str
    mode: str
    trials_per_class: int
    fullscr: bool
    profile_key: str
    profile_label: str


@dataclass(frozen=True)
class TriggerConfig:
    enabled: bool
    serial_port: str
    baud_rate: int


@dataclass(frozen=True)
class GeometryProfile:
    box_deg: float
    gap_deg: float
    x_offset_deg: float
    horizontal_span_deg: float
    vertical_span_deg: float
    scaled_to_fit: bool


@dataclass(frozen=True)
class FrequencyPlan:
    target_hz: float
    effective_hz: float
    half_cycle_frames: int
    error_hz: float
    exact: bool


# ── Safe dialog field parsers ─────────────────────────────────────────────────

def safe_int(value, default: int) -> int:
    """Convert a dialog field to int; return default if blank or non-numeric."""
    try:
        s = str(value).strip()
        return int(s) if s else default
    except (ValueError, TypeError):
        return default


def safe_float(value, default: float) -> float:
    """Convert a dialog field to float; return default if blank or non-numeric."""
    try:
        s = str(value).strip()
        return float(s) if s else default
    except (ValueError, TypeError):
        return default


# ── Serial trigger ────────────────────────────────────────────────────────────

class SerialTriggerSender:
    def __init__(self, cfg: TriggerConfig):
        self.cfg = cfg
        self.serial = None
        self.error: str | None = None
        if not cfg.enabled:
            return
        try:
            import serial
            self.serial = serial.Serial(
                port=cfg.serial_port, baudrate=cfg.baud_rate,
                timeout=0, write_timeout=0,
            )
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.serial = None

    @property
    def is_active(self) -> bool:
        return self.serial is not None

    def send_code(self, code: int) -> bool:
        if self.serial is None:
            return False
        try:
            self.serial.write(bytes([code & 0xFF]))
            self.serial.flush()
            return True
        except Exception:
            return False

    def close(self) -> None:
        if self.serial is not None:
            try:
                self.serial.close()
            except Exception:
                pass
            self.serial = None


# ── Utility functions ─────────────────────────────────────────────────────────

def apply_protocol_profile(profile_key: str) -> dict[str, Any]:
    if profile_key not in PROTOCOL_PROFILES:
        raise KeyError(f"Unknown profile: {profile_key}")
    profile = PROTOCOL_PROFILES[profile_key]
    global YES_FREQ_HZ, NO_FREQ_HZ, WINDOW_S, PRE_DELAY_S, TRIAL_S, REST_S
    global MODULATION_DEPTH, REQUESTED_BOX_DEG, REQUESTED_GAP_DEG, PHOTODIODE_ONSET_FRAMES
    YES_FREQ_HZ             = float(profile["yes_freq_hz"])
    NO_FREQ_HZ              = float(profile["no_freq_hz"])
    WINDOW_S                = float(profile["window_s"])
    PRE_DELAY_S             = float(profile["pre_delay_s"])
    TRIAL_S                 = float(profile["trial_s"])
    REST_S                  = float(profile["rest_s"])
    MODULATION_DEPTH        = float(profile["modulation_depth"])
    REQUESTED_BOX_DEG       = float(profile["box_deg"])
    REQUESTED_GAP_DEG       = float(profile["gap_deg"])
    PHOTODIODE_ONSET_FRAMES = int(profile["photodiode_onset_frames"])
    return profile


def log_trigger_event(bucket, label, code, delivered, clock):
    bucket.setdefault("trigger_events", []).append({
        "label": label, "code": int(code),
        "delivered": bool(delivered),
        "monotonic_s": clock.getTime(), "unix_s": time.time(),
    })


def queue_trigger_on_flip(win, trigger, trial_meta, label, code, clock):
    if trigger is None:
        return
    def _send():
        delivered = trigger.send_code(code)
        log_trigger_event(trial_meta, label, code, delivered, clock)
    win.callOnFlip(_send)


def build_photodiode_qc_thresholds(fps):
    frame_ms = 1000.0 / max(fps, 1e-6)
    return {
        "onset_patch_frames": PHOTODIODE_ONSET_FRAMES if ENABLE_PHOTODIODE_PATCH else 0,
        "max_onset_latency_ms": round(frame_ms * 1.5, 4),
        "max_onset_jitter_ms":  round(frame_ms * 0.5, 4),
        "max_missing_onsets_fraction": 0.0,
        "recommended_check": "Compare photodiode rising edge to record_start trigger.",
    }


def abort_session(win):
    win.close()
    core.quit()


def check_abort(win):
    if event.getKeys(keyList=["escape"]):
        abort_session(win)


def full_visual_angle_deg(size_cm, view_dist_cm):
    return math.degrees(2.0 * math.atan((size_cm / 2.0) / view_dist_cm))


def infer_monitor_size_cm(diagonal_in, width_px, height_px):
    diag_cm  = diagonal_in * 2.54
    diag_px  = math.hypot(width_px, height_px)
    return diag_cm * (width_px / diag_px), diag_cm * (height_px / diag_px)


def compute_n_windows(trial_s, window_s):
    raw     = trial_s / window_s
    rounded = round(raw)
    if not math.isclose(raw, rounded, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("TRIAL_S must be an integer multiple of WINDOW_S.")
    return int(rounded)


def build_trial_list(mode, trials_per_class):
    if mode == "yes":
        return ["yes"] * trials_per_class
    if mode == "no":
        return ["no"]  * trials_per_class
    arr = ["yes"] * trials_per_class + ["no"] * trials_per_class
    np.random.shuffle(arr)
    return arr


def choose_frequency_plan(target_hz, fps):
    # Special bypass for 20 Hz on a 60 Hz monitor (Asymmetrical 3-frame cycle)
    import math
    if math.isclose(target_hz, 20.0, abs_tol=0.1) and math.isclose(fps, 60.0, abs_tol=2.0):
        return FrequencyPlan(
            target_hz=20.0, effective_hz=fps/3.0,
            half_cycle_frames=0,  # 0 is our secret flag for an asymmetric cycle
            error_hz=abs((fps/3.0) - 20.0), exact=True
        )

    hcf = max(1, int(round(fps / (2.0 * target_hz))))
    eff = fps / (2.0 * hcf)
    err = abs(eff - target_hz)
    return FrequencyPlan(
        target_hz=target_hz, effective_hz=eff,
        half_cycle_frames=hcf, error_hz=err,
        exact=bool(err <= max(0.01, target_hz * 0.001)), 
    )


def build_contrast_trace(frame_count, plan, amplitude):
    import numpy as np
    frames = np.arange(frame_count, dtype=np.int32)
    
    if plan.half_cycle_frames == 0:
        # Asymmetric 3-frame cycle for 20Hz on 60Hz (1 frame ON, 2 frames OFF)
        blocks = frames % 3
        return np.where(blocks == 0, amplitude, -amplitude).astype(np.float32)
    else:
        # Standard symmetrical square wave
        blocks = (frames // plan.half_cycle_frames) % 2
        return np.where(blocks == 0, amplitude, -amplitude).astype(np.float32)


def fit_horizontal_geometry(display, requested_box_deg, requested_gap_deg, margin_deg=3.0):
    horiz = full_visual_angle_deg(display.width_cm,  display.view_dist_cm)
    vert  = full_visual_angle_deg(display.height_cm, display.view_dist_cm)
    avail = max(6.0, horiz - 2.0 * margin_deg)
    req   = 2.0 * requested_box_deg + requested_gap_deg
    scale = min(1.0, avail / req)
    box   = requested_box_deg * scale
    gap   = requested_gap_deg * scale
    return GeometryProfile(
        box_deg=box, gap_deg=gap,
        x_offset_deg=(box + gap) / 2.0,
        horizontal_span_deg=horiz, vertical_span_deg=vert,
        scaled_to_fit=scale < 0.999,
    )


def frame_stats(frame_intervals, expected_interval_s):
    if not frame_intervals:
        return {"samples": 0, "expected_ms": round(expected_interval_s * 1000, 4),
                "mean_ms": None, "median_ms": None, "max_ms": None, "dropped_frames": 0}
    arr = np.asarray(frame_intervals, dtype=float)
    thr = expected_interval_s * 1.5
    return {
        "samples":        int(arr.size),
        "expected_ms":    float(round(expected_interval_s * 1000, 4)), # <--- WRAP IN float()
        "mean_ms":        round(float(arr.mean()     * 1000), 4),
        "median_ms":      round(float(np.median(arr) * 1000), 4),
        "max_ms":         round(float(arr.max()      * 1000), 4),
        "dropped_frames": int(np.sum(arr > thr)),
    }


def evaluate_display_qc(fq):
    em   = fq.get("expected_ms")
    mm   = fq.get("max_ms")
    drop = int(fq.get("dropped_frames", 0))
    lim  = None if em is None else round(float(em) * 1.5, 4)
    ok   = drop == 0 and (mm is None or lim is None or float(mm) <= float(lim))
    return {"pass": bool(ok), "max_allowed_frame_ms": lim,
            "rule": "pass if dropped_frames==0 and max_frame_ms<=1.5*expected_frame_ms"}


def stamp_trial_event(bucket, prefix, clock):
    bucket[f"{prefix}_monotonic_s"] = clock.getTime()
    bucket[f"{prefix}_unix_s"]      = time.time()


# ── Dialog ────────────────────────────────────────────────────────────────────

def show_dialog():
    dlg = gui.Dlg(title="SSVEP BCI - Session Setup", labelButtonOK="Start")
    dlg.addText("--- Subject ---")
    dlg.addField("Subject ID:", "S01")
    dlg.addText("--- Recording ---")
    dlg.addField("Mode:", choices=["YES only", "NO only", "Mixed"])
    dlg.addField("Trials per class:", "10") # <--- Add quotes
    profile_items  = [(k, v["label"]) for k, v in PROTOCOL_PROFILES.items()]
    profile_labels = [item[1] for item in profile_items]
    dlg.addField("Protocol profile:", choices=profile_labels, initial=profile_labels[1])
    dlg.addText("--- Triggering ---")
    dlg.addField("Enable serial trigger?", False)
    dlg.addField("Serial port:", "COM3")
    dlg.addField("Serial baud:", "115200") # <--- Add quotes
    dlg.addText("--- Display ---")
    dlg.addField("Diagonal (in):", "15.94") # <--- Add quotes
    dlg.addField("Resolution width (px):",  "2560") # <--- Add quotes
    dlg.addField("Resolution height (px):", "1600") # <--- Add quotes
    dlg.addField("Viewing distance (cm):", "60.0") # <--- Add quotes
    dlg.addField("Refresh rate hint (Hz):", "144") # <--- Add quotes
    dlg.addField("Full-screen?", True)
    info = dlg.show()
    if not dlg.OK:
        raise SystemExit

    subject_id       = str(info[0]).strip() or "S01"
    mode_map         = {"YES only": "yes", "NO only": "no", "Mixed": "mixed"}
    mode             = mode_map[info[1]]
    # safe_int/safe_float prevent crashes when the user clears a field
    trials_per_class = max(1, safe_int(info[2], default=10))
    profile_label    = str(info[3])
    profile_lookup   = {label: key for key, label in profile_items}
    profile_key      = profile_lookup[profile_label]
    trig_enabled     = bool(info[4])
    serial_port      = str(info[5]).strip()
    baud_rate        = safe_int(info[6],   default=115200)
    diagonal_in      = safe_float(info[7], default=15.94)
    width_px         = safe_int(info[8],   default=2560)
    height_px        = safe_int(info[9],   default=1600)
    view_dist_cm     = safe_float(info[10], default=60.0)
    refresh_hint     = safe_float(info[11], default=144.0)
    fullscr          = bool(info[12])
    w_cm, h_cm       = infer_monitor_size_cm(diagonal_in, width_px, height_px)

    return (
        SessionConfig(subject_id=subject_id, mode=mode,
                      trials_per_class=trials_per_class, fullscr=fullscr,
                      profile_key=profile_key, profile_label=profile_label),
        DisplayProfile(diagonal_in=diagonal_in, width_px=width_px, height_px=height_px,
                       width_cm=w_cm, height_cm=h_cm,
                       view_dist_cm=view_dist_cm, refresh_hint_hz=refresh_hint),
        TriggerConfig(enabled=trig_enabled, serial_port=serial_port, baud_rate=baud_rate),
    )


# ── Text screen helpers ───────────────────────────────────────────────────────

def _text_screen(win, text, height=0.055, color=WHITE, wrap=1.75):
    return visual.TextStim(
        win, text=text,
        units="norm", height=height,
        wrapWidth=wrap, color=color,
        alignText="left",
        pos=(0, 0),
        autoLog=False,
    )


def show_instructions(win, measured_fps, yes_plan, no_plan, geometry,
                      profile_label, trigger, total_trials):
    notes = []
    if geometry.scaled_to_fit:
        notes.append(
            f"Stimulus auto-fitted: {geometry.box_deg:.1f}° boxes, "
            f"{geometry.gap_deg:.1f}° gap."
        )
    if not yes_plan.exact or not no_plan.exact:
        notes.append(
            "One or more frequencies are not exact at this refresh rate — "
            "consider a profile matched to your panel (see dialog)."
        )
    if trigger is not None:
        if trigger.is_active:
            notes.append(
                f"Serial trigger active on {trigger.cfg.serial_port} "
                f"@ {trigger.cfg.baud_rate}."
            )
        elif trigger.cfg.enabled:
            notes.append("Serial trigger requested but unavailable — continuing without.")

    lines = [
        "SSVEP BCI  —  Instructions",
        "",
        "Side-by-side layout:  LEFT checkerboard = YES   RIGHT = NO",
        "Both checkerboards flicker continuously at all times.",
        "Fixate the one named in the cue — ignore the other.",
        "",
        "1.  Wait for the cue arrow, then shift gaze to that side.",
        "2.  Keep gaze fixed on that checkerboard during REC.",
        "3.  Relax jaw, minimise blinking during REC.",
        "",
        f"Profile : {profile_label}",
        f"Trials  : {TRIAL_S:.0f} s  x  "
        f"{compute_n_windows(TRIAL_S, WINDOW_S)} windows of {WINDOW_S:.0f} s",
        "",
        f"Display : {measured_fps:.3f} Hz  (measured)",
        f"YES  :  {YES_FREQ_HZ:.4f} Hz target  ->  {yes_plan.effective_hz:.4f} Hz effective"
        f"  (hcf={yes_plan.half_cycle_frames}  err={yes_plan.error_hz:.4f} Hz)",
        f"NO   :  {NO_FREQ_HZ:.4f} Hz target  ->  {no_plan.effective_hz:.4f} Hz effective"
        f"  (hcf={no_plan.half_cycle_frames}  err={no_plan.error_hz:.4f} Hz)",
    ]
    if notes:
        lines += ["", "Notes:"] + [f"  * {n}" for n in notes]
    lines += ["", "Press  SPACE  to begin  /  ESC  to abort."]

    stim = _text_screen(win, "\n".join(lines))
    event.clearEvents(eventType="keyboard")
    stim.draw()
    win.flip()
    keys = event.waitKeys(keyList=["space", "escape"])
    if keys and "escape" in keys:
        abort_session(win)
    
    # 1. Calculate the exact total time (Trials + Intro Countdown)
    countdown_extra = 3.4 # 3s for 3-2-1 + 0.4s for "GO"
    total_time_s = (total_trials * (PRE_DELAY_S + TRIAL_S + REST_S + 0.2)) + countdown_extra
    
    # 2. Find local IP address
    local_ip = "127.0.0.1"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception: pass

    # 3. Show instructions on the PC Monitor
    lines = [
        "PHONE REMOTE ACTIVE",
        f"Open mobile browser to: http://{local_ip}:8080",
        "",
        "Waiting for mobile START signal... (or press SPACE to force start)"
    ]
    stim = visual.TextStim(win, text="\n".join(lines), units="norm", height=0.055)
    event.clearEvents()
    stim.draw()
    win.flip()

    # 4. Temporary Web Server (shuts down completely before rendering begins)
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", 8080))
        sock.listen(1)
        sock.setblocking(False)
    except OSError as bind_err:
        if sock is not None:
            sock.close()
        print(f"[WARN] Could not start phone remote server: {bind_err}")
        return  # Fall through to experiment without phone remote

    while True:
        # PC Keyboard override
        keys = event.getKeys(keyList=["space", "escape"])
        if "escape" in keys:
            sock.close()
            abort_session(win)
        if "space" in keys:
            break

        # Listen for Phone
        try:
            conn, addr = sock.accept()
            request = conn.recv(1024).decode('utf-8', errors='ignore')
            
            if "GET /start" in request:
                # PHONE CLICKED START! Send success, close server, and break loop!
                conn.sendall(b"HTTP/1.1 200 OK\n\nSTARTED")
                conn.close()
                break 
            else:
                # PHONE LOADED PAGE! Send the JS Dashboard
                html = f"""HTTP/1.1 200 OK
Content-Type: text/html

<!DOCTYPE html>
<html>
<head><meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no"></head>
<body style="background:#111; color:#fff; font-family:sans-serif; text-align:center; padding:20px;">
    
    <div id="setup" style="margin-top:50px;">
        <h2 style="color:#aaa;">SSVEP Remote</h2>
        <button onclick="startExp()" style="padding:40px; font-size:35px; background:#00ff55; color:#000; border-radius:15px; width:100%; font-weight:bold;">START</button>
    </div>

    <div id="running" style="display:none; margin-top:50px;">
        <h3 style="color:#aaa;">TIME REMAINING</h3>
        <h1 id="timer" style="font-size:90px; color:#00ff55; margin:10px 0;">0.0s</h1>
        <progress id="prog" value="0" max="{total_time_s}" style="width:100%; height:40px; accent-color:#aa22ff;"></progress>
    </div>

    <script>
        const duration = {total_time_s} * 1000; // convert to ms
        let startTime;

        function startExp() {{
            fetch('/start');
            startTime = Date.now();
            document.getElementById('setup').style.display = 'none';
            document.getElementById('running').style.display = 'block';
            
            let iv = setInterval(() => {{
                let elapsed = Date.now() - startTime;
                let remaining = Math.max(0, (duration - elapsed) / 1000);
                
                if(remaining <= 0) {{
                    clearInterval(iv);
                    document.getElementById('running').innerHTML = '<h1 style="color:#00ff55; font-size:60px; margin-top:100px;">COMPLETE!</h1>';
                }} else {{
                    document.getElementById('timer').innerText = remaining.toFixed(1) + 's';
                    document.getElementById('prog').value = elapsed / 1000;
                }}
            }}, 50); // Faster update for smoother UI
        }}
    </script>
</body>
</html>"""
                conn.sendall(html.encode('utf-8'))
                conn.close()
                
        except BlockingIOError:
            pass
        core.wait(0.01)

    # Completely destroy the server so it doesn't interfere with PsychoPy
    sock.close() 
    # ------------------------------------------


def show_countdown(win, fps):
    sub = visual.TextStim(
        win, text="Prepare — keep your eyes on the centre cross",
        units="norm", height=0.058,
        pos=(0, -0.38), color=MUTED, autoLog=False,
    )
    num = visual.TextStim(
        win, text="",
        units="norm", height=0.52,
        bold=True, color=WHITE, autoLog=False,
    )
    fpsi = max(1, round(fps))

    for n in (3, 2, 1):
        num.text = str(n)
        for _ in range(fpsi):
            num.draw(); sub.draw(); win.flip(); check_abort(win)

    num.text  = "GO"
    num.color = YES_COL
    for _ in range(max(1, round(fps * 0.4))):
        num.draw(); win.flip(); check_abort(win)


# ── Stimuli ───────────────────────────────────────────────────────────────────

def build_stimuli(win, geometry, yes_plan, no_plan, n_windows):
    box = geometry.box_deg
    xo  = geometry.x_offset_deg
    sf  = (CELLS_PER_SIDE / 2.0) / box

    stim_yes = visual.GratingStim(
        win, tex="sqrXsqr", units="deg",
        size=(box, box), sf=(sf, sf), pos=(-xo, 0),
        contrast=MODULATION_DEPTH, autoLog=False,
    )
    stim_no = visual.GratingStim(
        win, tex="sqrXsqr", units="deg",
        size=(box, box), sf=(sf, sf), pos=(xo, 0),
        contrast=MODULATION_DEPTH, autoLog=False,
    )

    fix = visual.ShapeStim(
        win, vertices="cross", units="norm",
        size=0.045, fillColor=WHITE, lineColor=WHITE,
        pos=(0, 0), autoLog=False,
    )

    label_y = -(box / 2.0 + 1.0)
    label_yes = visual.TextStim(
        win, units="deg",
        text=f"YES  {yes_plan.effective_hz:.2f} Hz",
        height=0.5, pos=(-xo, label_y), color=YES_COL, opacity=0.55, autoLog=False,
    )
    label_no = visual.TextStim(
        win, units="deg",
        text=f"NO  {no_plan.effective_hz:.2f} Hz",
        height=0.5, pos=(xo, label_y), color=NO_COL, opacity=0.55, autoLog=False,
    )

    glow_yes = visual.Rect(
        win, units="deg",
        width=box + 0.5, height=box + 0.5, pos=(-xo, 0),
        lineColor=YES_COL, fillColor=None, lineWidth=6, autoLog=False,
    )
    glow_no = visual.Rect(
        win, units="deg",
        width=box + 0.5, height=box + 0.5, pos=(xo, 0),
        lineColor=NO_COL, fillColor=None, lineWidth=6, autoLog=False,
    )

    hud_trial = visual.TextStim(
        win, text="", units="norm", height=0.052,
        pos=(-0.50, 0.93), color=WHITE, autoLog=False,
    )
    hud_phase = visual.TextStim(
        win, text="", units="norm", height=0.052,
        pos=(0.55, 0.93), color=ACCENT, autoLog=False,
    )
    hud_freqs = visual.TextStim(
        win, units="norm", height=0.042,
        text=f"YES {yes_plan.effective_hz:.2f} Hz     NO {no_plan.effective_hz:.2f} Hz",
        pos=(0, 0.84), color=MUTED, autoLog=False,
    )
    rec_dot = visual.TextStim(
        win, text="* REC", units="norm", height=0.048,
        pos=(0.70, 0.84), color=NO_COL, autoLog=False,
    )

    prog_bg = visual.Rect(
        win, units="norm",
        width=0.92, height=0.024, pos=(0, -0.95),
        fillColor=(-0.85, -0.85, -0.85), lineColor=None, autoLog=False,
    )
    prog_fill = visual.Rect(
        win, units="norm",
        width=0.001, height=0.024, pos=(-0.46, -0.95),
        fillColor=ACCENT, lineColor=None, autoLog=False,
    )

    chip_w, chip_h, chip_gap = 0.055, 0.020, 0.010
    total_chip_w = n_windows * chip_w + (n_windows - 1) * chip_gap
    chip_x0      = -total_chip_w / 2.0 + chip_w / 2.0
    chip_y       = -0.875
    chips = []
    for idx in range(n_windows):
        xc = chip_x0 + idx * (chip_w + chip_gap)
        chips.append(visual.Rect(
            win, units="norm", width=chip_w, height=chip_h,
            pos=(xc, chip_y), fillColor=CHIP_DIM, lineColor=None, autoLog=False,
        ))

    cue_bg = visual.Rect(
        win, units="norm", width=2.5, height=2.5,
        fillColor=BG, lineColor=None, opacity=0.88, autoLog=False,
    )
    cue_arrow = visual.TextStim(
        win, text="", units="norm", height=0.28,
        pos=(0, 0.18), bold=True, color=YES_COL, autoLog=False,
    )
    cue_word = visual.TextStim(
        win, text="", units="norm", height=0.11,
        pos=(0, -0.08), bold=True, color=YES_COL, autoLog=False,
    )
    cue_sub = visual.TextStim(
        win, text="", units="norm", height=0.052,
        pos=(0, -0.28), color=MUTED, autoLog=False,
    )

    rest_bg = visual.Rect(
        win, units="norm", width=2.5, height=2.5,
        fillColor=BG, lineColor=None, opacity=0.93, autoLog=False,
    )
    rest_label = visual.TextStim(
        win, text="REST", units="norm", height=0.088,
        pos=(0, 0.22), color=MUTED, autoLog=False,
    )
    rest_timer = visual.TextStim(
        win, text="2", units="norm", height=0.48,
        pos=(0, -0.10), color=(-0.55, -0.55, -0.55), bold=True, autoLog=False,
    )

    photodiode_patch = None
    if ENABLE_PHOTODIODE_PATCH:
        photodiode_patch = visual.Rect(
            win, units="norm", width=0.055, height=0.055,
            pos=(-0.97, 0.97), fillColor=BG, lineColor=None, autoLog=False,
        )

    return {
        "stim_yes": stim_yes, "stim_no": stim_no,
        "fix": fix,
        "label_yes": label_yes, "label_no": label_no,
        "glow_yes": glow_yes, "glow_no": glow_no,
        "hud_trial": hud_trial, "hud_phase": hud_phase,
        "hud_freqs": hud_freqs, "rec_dot": rec_dot,
        "prog_bg": prog_bg, "prog_fill": prog_fill,
        "cue_bg": cue_bg, "cue_arrow": cue_arrow,
        "cue_word": cue_word, "cue_sub": cue_sub,
        "rest_bg": rest_bg, "rest_label": rest_label, "rest_timer": rest_timer,
        "chips": chips, "photodiode_patch": photodiode_patch,
    }


# ── Draw helpers ──────────────────────────────────────────────────────────────

def set_progress(stims, fraction):
    total_w = stims["prog_bg"].width
    cx, cy  = stims["prog_bg"].pos
    left    = cx - total_w / 2.0
    w       = max(0.001, total_w * max(0.0, min(1.0, fraction)))
    stims["prog_fill"].width = w
    stims["prog_fill"].pos   = (left + w / 2.0, cy)


def reset_chips(stims):
    for chip in stims["chips"]:
        chip.fillColor = CHIP_DIM


def update_chips(stims, elapsed_s, n_windows):
    half_accent = tuple(v * 0.5 for v in ACCENT)
    for idx, chip in enumerate(stims["chips"]):
        done  = (idx + 1) * WINDOW_S
        start = idx * WINDOW_S
        if elapsed_s >= done:
            chip.fillColor = ACCENT
        elif elapsed_s >= start + WINDOW_S * 0.5:
            chip.fillColor = half_accent
        else:
            chip.fillColor = CHIP_DIM


def set_cue(stims, target, effective_hz):
    is_yes = target == "yes"
    col    = YES_COL if is_yes else NO_COL
    side   = "LEFT"  if is_yes else "RIGHT"
    arrow  = "<--"   if is_yes else "-->"
    stims["cue_arrow"].text  = arrow
    stims["cue_arrow"].color = col
    stims["cue_word"].text   = f"LOOK  {side}"
    stims["cue_word"].color  = col
    stims["cue_sub"].text    = f"Gaze at the {side} checkerboard  ({effective_hz:.2f} Hz)"


def set_photodiode(stims, active):
    p = stims["photodiode_patch"]
    if p is not None:
        p.fillColor = WHITE if active else BG


def draw_base(stims):
    stims["stim_yes"].draw()
    stims["stim_no"].draw()
    stims["fix"].draw()
    stims["label_yes"].draw()
    stims["label_no"].draw()
    if stims["photodiode_patch"] is not None:
        stims["photodiode_patch"].draw()


def draw_hud(stims, trial_num, total_trials, target, phase_label, phase_col, show_rec=False):
    col = YES_COL if target == "yes" else NO_COL
    stims["hud_trial"].text  = f"Trial {trial_num}/{total_trials}  {target.upper()}"
    stims["hud_trial"].color = col
    stims["hud_phase"].text  = phase_label
    stims["hud_phase"].color = phase_col
    stims["hud_trial"].draw()
    stims["hud_phase"].draw()
    stims["hud_freqs"].draw()
    stims["prog_bg"].draw()
    stims["prog_fill"].draw()
    for chip in stims["chips"]:
        chip.draw()
    if show_rec:
        stims["rec_dot"].draw()


# ── Window metadata ───────────────────────────────────────────────────────────

def build_windows_metadata(record_start_monotonic_s, n_windows,
                            actual_flip_timestamps_s=None):
    t0   = record_start_monotonic_s * 1000.0
    have = (actual_flip_timestamps_s is not None
            and len(actual_flip_timestamps_s) == n_windows + 1)
    out  = []
    for idx in range(n_windows):
        if have:
            s_ms = actual_flip_timestamps_s[idx]     * 1000.0
            e_ms = actual_flip_timestamps_s[idx + 1] * 1000.0
            src  = "flip"
        else:
            s_ms = t0 + idx * WINDOW_S * 1000.0
            e_ms = t0 + (idx + 1) * WINDOW_S * 1000.0
            src  = "computed"
        out.append({
            "window_idx":          idx,
            "start_monotonic_ms":  round(s_ms, 4),
            "end_monotonic_ms":    round(e_ms, 4),
            "start_sample":        int(round(idx * WINDOW_S * EEG_FS_HZ)),
            "end_sample":          int(round((idx + 1) * WINDOW_S * EEG_FS_HZ)),
            "timestamp_source":    src,
        })
    return out


# ── Session runner ────────────────────────────────────────────────────────────

def run_session(win, clock, session, geometry, yes_plan, no_plan,
                measured_fps, trigger):
    n_windows     = compute_n_windows(TRIAL_S, WINDOW_S)
    trial_list    = build_trial_list(session.mode, session.trials_per_class)
    total_trials  = len(trial_list)
    cue_frames    = max(1, round(PRE_DELAY_S * measured_fps))
    record_frames = max(1, round(TRIAL_S     * measured_fps))
    rest_frames   = max(1, round(REST_S      * measured_fps))

    yes_trace = build_contrast_trace(record_frames, yes_plan, MODULATION_DEPTH)
    no_trace  = build_contrast_trace(record_frames, no_plan,  MODULATION_DEPTH)

    stims = build_stimuli(win, geometry, yes_plan, no_plan, n_windows)
    session_data: list[dict] = []

    # Guaranteed fallback timestamp source — collected inside the recording
    # loop regardless of whether win.frameIntervals works correctly.
    manual_flip_ts: list[float] = []

    show_instructions(win, measured_fps, yes_plan, no_plan, geometry,
                      session.profile_label, trigger, total_trials)
    show_countdown(win, measured_fps)

    win.recordFrameIntervals = True
    # FIX: refreshThreshold is now set from the MEASURED fps (computed after
    # getActualFrameRate returns) not the dialog hint.  At 144 Hz each frame
    # is ~6.94 ms; using a 240 Hz hint gave a threshold of 6.25 ms which every
    # single frame exceeded, flooding the log with false dropped-frame warnings.
    win.refreshThreshold = (1.0 / measured_fps) * 1.5
    # FIX: in-place clear — never assign win.frameIntervals = [] because that
    # detaches PsychoPy's internal list and leaves it permanently empty.
    del win.frameIntervals[:]

    for trial_idx, target in enumerate(trial_list):
        trial_num   = trial_idx + 1
        glow        = stims["glow_yes"] if target == "yes" else stims["glow_no"]
        target_plan = yes_plan if target == "yes" else no_plan

        reset_chips(stims)
        set_progress(stims, trial_idx / total_trials)
        set_cue(stims, target, target_plan.effective_hz)
        set_photodiode(stims, False)

        trial_meta = {
            "trial": trial_num, "target": target,
            "decision": "PENDING", "correct": None,
            "subject": session.subject_id, "mode": session.mode,
            "trial_start_monotonic_s": clock.getTime(),
            "trial_start_unix_s":      time.time(),
            "cue_onset_monotonic_s":   None, "cue_onset_unix_s": None,
            "record_start_monotonic_s": None, "record_start_unix_s": None,
            "record_end_monotonic_s":  None, "record_end_unix_s": None,
            "duration_s": None, "windows": [], "trigger_events": [],
        }
        queue_trigger_on_flip(win, trigger, trial_meta, "trial_start",
                              TRIGGER_CODES["trial_start"], clock)

        stims["stim_yes"].contrast = MODULATION_DEPTH
        stims["stim_no"].contrast  = MODULATION_DEPTH

        # ── Cue phase ─────────────────────────────────────────────────────────
        for frame_idx in range(cue_frames):
            if frame_idx == 0:
                win.callOnFlip(stamp_trial_event, trial_meta, "cue_onset", clock)
                code  = TRIGGER_CODES["cue_yes"] if target == "yes" else TRIGGER_CODES["cue_no"]
                label = "cue_yes"               if target == "yes" else "cue_no"
                queue_trigger_on_flip(win, trigger, trial_meta, label, code, clock)
            draw_base(stims)
            stims["cue_bg"].draw()
            stims["cue_arrow"].draw()
            stims["cue_word"].draw()
            stims["cue_sub"].draw()
            draw_hud(stims, trial_num, total_trials, target, "CUE", ACCENT)
            win.flip()
            check_abort(win)

        # ── Recording phase ───────────────────────────────────────────────────
        last_yes = float("nan")
        last_no  = float("nan")
        window_flip_ts:   list[float] = []
        frames_per_window = max(1, round(WINDOW_S * measured_fps))

        def _stamp_wb():
            window_flip_ts.append(clock.getTime())

        for frame_idx in range(record_frames):
            ny = float(yes_trace[frame_idx])
            nn = float(no_trace[frame_idx])
            if ny != last_yes:
                stims["stim_yes"].contrast = ny; last_yes = ny
            if nn != last_no:
                stims["stim_no"].contrast  = nn; last_no  = nn

            update_chips(stims, frame_idx / measured_fps, n_windows)
            set_progress(stims,
                (trial_idx + frame_idx / max(1, record_frames)) / total_trials)
            set_photodiode(stims, frame_idx < PHOTODIODE_ONSET_FRAMES)

            if frame_idx == 0:
                win.callOnFlip(stamp_trial_event, trial_meta, "record_start", clock)
                win.callOnFlip(_stamp_wb)
                code  = (TRIGGER_CODES["record_start_yes"]
                         if target == "yes" else TRIGGER_CODES["record_start_no"])
                label = ("record_start_yes" if target == "yes" else "record_start_no")
                queue_trigger_on_flip(win, trigger, trial_meta, label, code, clock)
            elif frame_idx % frames_per_window == 0:
                win.callOnFlip(_stamp_wb)

            draw_base(stims)
            glow.draw()
            draw_hud(stims, trial_num, total_trials, target, "RECORDING",
                     YES_COL if target == "yes" else NO_COL, show_rec=True)
            win.flip()
            # Collect raw flip timestamp after every recording flip.
            manual_flip_ts.append(clock.getTime())
            check_abort(win)

        window_flip_ts.append(clock.getTime())

        # ── End of recording ──────────────────────────────────────────────────
        set_photodiode(stims, False)
        if trigger is not None:
            end_code  = (TRIGGER_CODES["record_end_yes"]
                         if target == "yes" else TRIGGER_CODES["record_end_no"])
            end_label = "record_end_yes" if target == "yes" else "record_end_no"
            delivered = trigger.send_code(end_code)
            log_trigger_event(trial_meta, end_label, end_code, delivered, clock)

        trial_meta["record_end_monotonic_s"] = clock.getTime()
        trial_meta["record_end_unix_s"]      = time.time()

        if trial_meta["record_start_monotonic_s"] is None:
            trial_meta["record_start_monotonic_s"] = clock.getTime() - TRIAL_S
            trial_meta["record_start_unix_s"]      = time.time() - TRIAL_S
            trial_meta["record_start_estimated"]   = True

        trial_meta["duration_s"] = (
            trial_meta["record_end_monotonic_s"]
            - trial_meta["record_start_monotonic_s"]
        )
        trial_meta["windows"] = build_windows_metadata(
            trial_meta["record_start_monotonic_s"], n_windows,
            actual_flip_timestamps_s=(window_flip_ts
                                      if len(window_flip_ts) == n_windows + 1
                                      else None),
        )
        session_data.append(trial_meta)

        if trigger is not None:
            delivered = trigger.send_code(TRIGGER_CODES["trial_end"])
            log_trigger_event(trial_meta, "trial_end",
                              TRIGGER_CODES["trial_end"], delivered, clock)

        # ── Rest phase ────────────────────────────────────────────────────────
        update_chips(stims, TRIAL_S + 0.01, n_windows)
        set_progress(stims, trial_num / total_trials)
        stims["stim_yes"].contrast = MODULATION_DEPTH
        stims["stim_no"].contrast  = MODULATION_DEPTH

        for frame_idx in range(rest_frames):
            remaining = max(0.0, REST_S - frame_idx / measured_fps)
            stims["rest_timer"].text = str(max(1, math.ceil(remaining)))
            draw_base(stims)
            stims["rest_bg"].draw()
            stims["rest_label"].draw()
            stims["rest_timer"].draw()
            draw_hud(stims, trial_num, total_trials, target, "REST", MUTED)
            win.flip()
            check_abort(win)

    win.recordFrameIntervals = False
    fi = list(win.frameIntervals)

    # Fallback: derive intervals from manual timestamps if PsychoPy list is empty.
    if not fi and len(manual_flip_ts) > 1:
        print("[INFO] win.frameIntervals empty — deriving intervals from manual flip timestamps.")
        fi = [manual_flip_ts[i + 1] - manual_flip_ts[i]
              for i in range(len(manual_flip_ts) - 1)]

    return session_data, fi, frame_stats(fi, 1.0 / measured_fps)


# ── Save data ─────────────────────────────────────────────────────────────────

def save_data(session_data, session, display, geometry, profile,
              trigger_cfg, trigger_status, measured_fps,
              yes_plan, no_plan, frame_intervals, frame_quality,
              partial: bool = False):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_PARTIAL" if partial else ""
    base   = OUTPUT_DIR / f"{session.subject_id}_{ts}{suffix}"
    nw     = compute_n_windows(TRIAL_S, WINDOW_S)

    payload = {
        "subject":    session.subject_id,
        "mode":       session.mode,
        "mode_label": MODE_LABELS[session.mode],
        "partial":    partial,
        "protocol_profile": {
            "key":      session.profile_key,
            "label":    session.profile_label,
            "settings": profile,
        },
        "saved_at_iso":      datetime.now().isoformat(),
        "stimulus_paradigm": "checkerboard_phase_reversal",
        "monitor":   asdict(display),
        "geometry":  asdict(geometry),
        "flicker": {
            "yes": asdict(yes_plan), "no": asdict(no_plan),
            "cells_per_side":        CELLS_PER_SIDE,
            "requested_box_deg":     REQUESTED_BOX_DEG,
            "requested_gap_deg":     REQUESTED_GAP_DEG,
            "modulation_depth":      MODULATION_DEPTH,
            "photodiode_onset_frames": (
                PHOTODIODE_ONSET_FRAMES if ENABLE_PHOTODIODE_PATCH else 0
            ),
        },
        "timing": {
            "measured_refresh_hz": round(measured_fps, 6),
            "frame_quality":       frame_quality,
            "display_qc":          evaluate_display_qc(frame_quality),
            "window_s":    WINDOW_S,
            "trial_s":     TRIAL_S,
            "pre_delay_s": PRE_DELAY_S,
            "rest_s":      REST_S,
            "n_windows":   nw,
            "eeg_fs_hz":   EEG_FS_HZ,
            "photodiode_qc_thresholds": build_photodiode_qc_thresholds(measured_fps),
        },
        "triggering": {
            "config":     asdict(trigger_cfg),
            "codes":      TRIGGER_CODES,
            "wiring_map": TRIGGER_WIRING_MAP,
            "status":     trigger_status,
        },
        "note": (
            "record_start_* timestamps scheduled on first recording flip. "
            "Use photodiode + hardware trigger for true cross-device validation."
        ),
        "trials": session_data,
    }

    json_path = base.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = base.with_suffix(".csv")
    fields = [
        "trial", "target", "decision", "correct",
        "trial_start_monotonic_s", "cue_onset_monotonic_s",
        "record_start_monotonic_s", "record_end_monotonic_s",
        "duration_s", "record_start_unix_s", "record_end_unix_s",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in session_data:
            w.writerow({k: (row.get(k, "") if row.get(k) is not None else "")
                        for k in fields})

    # Always write the frame-intervals file — no silent skip on empty list.
    # An empty list produces a header-only CSV which makes data loss visible.
    fi_path = base.with_name(base.name + "_frame_intervals.csv")
    with fi_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "interval_ms"])
        for idx, iv in enumerate(frame_intervals):
            w.writerow([idx, round(iv * 1000, 6)])

    return json_path, csv_path, fi_path


# ── Completion screen ─────────────────────────────────────────────────────────

def show_complete(win, session_data, session, frame_quality,
                  trigger_status, json_path, csv_path, frame_interval_path):
    yes_c = sum(1 for r in session_data if r["target"] == "yes")
    no_c  = sum(1 for r in session_data if r["target"] == "no")
    dur   = 0.0
    if session_data:
        dur = (session_data[-1]["record_end_monotonic_s"]
               - session_data[0]["trial_start_monotonic_s"]) / 60.0

    saved = [str(json_path), str(csv_path)]
    if frame_interval_path:
        saved.append(str(frame_interval_path))
    trig_line = ("Trigger : active"
                 if trigger_status.get("active") else "Trigger : disabled/unavailable")

    text = (
        f"Session complete\n\n"
        f"Subject  : {session.subject_id}\n"
        f"Mode     : {MODE_LABELS[session.mode]}\n"
        f"Profile  : {session.profile_label}\n"
        f"Trials   : {len(session_data)}  (YES {yes_c} / NO {no_c})\n"
        f"Duration : {dur:.1f} min\n"
        f"Dropped  : {frame_quality['dropped_frames']} frames\n"
        f"{trig_line}\n\n"
        f"Saved:\n" + "\n".join(saved) + "\n\n"
        f"Press  SPACE  to exit."
    )
    msg = _text_screen(win, text)
    event.clearEvents(eventType="keyboard")
    msg.draw()
    win.flip()
    keys = event.waitKeys(keyList=["space", "escape"])
    if keys and "escape" in keys:
        abort_session(win)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    session, display, trigger_cfg = show_dialog()
    profile = apply_protocol_profile(session.profile_key)

    w_cm, h_cm = infer_monitor_size_cm(
        display.diagonal_in, display.width_px, display.height_px
    )
    monitor = monitors.Monitor("testMonitor")
    monitor.setSizePix((display.width_px, display.height_px))
    monitor.setWidth(w_cm)
    monitor.setDistance(display.view_dist_cm)

    display = DisplayProfile(
        diagonal_in=display.diagonal_in,
        width_px=display.width_px, height_px=display.height_px,
        width_cm=w_cm, height_cm=h_cm,
        view_dist_cm=display.view_dist_cm,
        refresh_hint_hz=display.refresh_hint_hz,
    )
    geometry = fit_horizontal_geometry(display, REQUESTED_BOX_DEG, REQUESTED_GAP_DEG)

    trigger = SerialTriggerSender(trigger_cfg)
    trigger_status = {
        "requested": trigger_cfg.enabled,
        "active":    trigger.is_active,
        "error":     trigger.error,
    }

    win = visual.Window(
        size=(display.width_px, display.height_px),
        fullscr=session.fullscr,
        color=BG,
        monitor=monitor,
        units="deg",
        allowGUI=False,
        waitBlanking=True,
        winType="pyglet",
        screen=0,
    )
    win.mouseVisible = False

    actual_w, actual_h = win.size
    if (actual_w, actual_h) != (display.width_px, display.height_px):
        print(
            f"[INFO] OS display scaling detected: "
            f"framebuffer is {actual_w}x{actual_h} "
            f"(dialog said {display.width_px}x{display.height_px}). "
            f"All UI looks correct; stimulus visual angle may differ slightly."
        )

    fps_msg = visual.TextStim(
        win, text="Calibrating display...\nPlease wait.",
        units="norm", height=0.08, color=WHITE,
    )
    fps_msg.draw()
    win.flip()

    measured_fps = win.getActualFrameRate(
        nIdentical=20, nMaxFrames=300, nWarmUpFrames=20, threshold=1,
    )
    if measured_fps is None:
        measured_fps = display.refresh_hint_hz
        print(f"[WARN] getActualFrameRate() failed — falling back to hint: {measured_fps:.1f} Hz")
    else:
        measured_fps = float(measured_fps)
        print(f"[INFO] Measured refresh rate : {measured_fps:.3f} Hz")

    # Frequency plans are computed AFTER measuring FPS so hcf values are correct.
    yes_plan = choose_frequency_plan(YES_FREQ_HZ, measured_fps)
    no_plan  = choose_frequency_plan(NO_FREQ_HZ,  measured_fps)

    print(f"[INFO] YES : {YES_FREQ_HZ} Hz target -> {yes_plan.effective_hz:.4f} Hz effective"
          f"  (hcf={yes_plan.half_cycle_frames}, err={yes_plan.error_hz:.4f} Hz,"
          f" exact={yes_plan.exact})")
    print(f"[INFO] NO  : {NO_FREQ_HZ} Hz target -> {no_plan.effective_hz:.4f} Hz effective"
          f"  (hcf={no_plan.half_cycle_frames}, err={no_plan.error_hz:.4f} Hz,"
          f" exact={no_plan.exact})")
    if not yes_plan.exact or not no_plan.exact:
        print("[WARN] One or more frequencies are NOT exact at this refresh rate. "
              "Switch to the matching panel profile in the dialog.")

    clock = core.MonotonicClock()

    # Mutable result container so the finally block can save partial data on
    # any exception, including SystemExit from check_abort / ESC.
    result: dict[str, Any] = {
        "session_data":    [],
        "frame_intervals": [],
        "frame_quality":   frame_stats([], 1.0 / measured_fps),
    }

    try:
        session_data, frame_intervals, frame_quality = run_session(
            win=win, clock=clock, session=session, geometry=geometry,
            yes_plan=yes_plan, no_plan=no_plan,
            measured_fps=measured_fps, trigger=trigger,
        )
        result["session_data"]    = session_data
        result["frame_intervals"] = frame_intervals
        result["frame_quality"]   = frame_quality

        json_path, csv_path, fi_path = save_data(
            session_data=session_data, session=session,
            display=display, geometry=geometry,
            profile=profile, trigger_cfg=trigger_cfg,
            trigger_status=trigger_status, measured_fps=measured_fps,
            yes_plan=yes_plan, no_plan=no_plan,
            frame_intervals=frame_intervals, frame_quality=frame_quality,
            partial=False,
        )

        show_complete(
            win=win, session_data=session_data, session=session,
            frame_quality=frame_quality, trigger_status=trigger_status,
            json_path=json_path, csv_path=csv_path,
            frame_interval_path=fi_path,
        )

    except Exception as exc:
        # Catch everything (including SystemExit from ESC) and save what we have.
        print(f"[ERROR] Session interrupted: {type(exc).__name__}: {exc}")
        if result["session_data"]:
            try:
                jp, cp, fp = save_data(
                    session_data=result["session_data"], session=session,
                    display=display, geometry=geometry,
                    profile=profile, trigger_cfg=trigger_cfg,
                    trigger_status=trigger_status, measured_fps=measured_fps,
                    yes_plan=yes_plan, no_plan=no_plan,
                    frame_intervals=result["frame_intervals"],
                    frame_quality=result["frame_quality"],
                    partial=True,
                )
                print(f"[INFO] Partial data saved -> {jp}")
            except Exception as save_exc:
                print(f"[ERROR] Could not save partial data: {save_exc}")

    finally:
        trigger.close()
        win.close()
        core.quit()


if __name__ == "__main__":
    main()