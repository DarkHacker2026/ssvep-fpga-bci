import argparse
import collections
import os
import subprocess
import threading
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIVADO_SIM_DIR = os.path.join(SCRIPT_DIR, "vivado_sim")
CMD_FILE = os.path.join(VIVADO_SIM_DIR, "cmd.txt")
EEG_FILE = os.path.join(VIVADO_SIM_DIR, "live_eeg.txt")

WINDOW_SIZE = 512
DEFAULT_CAL_WINDOWS = 16
DEFAULT_STREAM_DELAY_S = 0.10
DEFAULT_DEMO_WINDOWS = 8
COMMAND_TIMEOUT_S = 10.0
DEFAULT_UNCERTAIN_LIMIT = 4

DEFAULT_LABEL_NAMES = ("YES", "NO", "CLASS2", "CLASS3")

DECISION_LABELS = {idx: name for idx, name in enumerate(DEFAULT_LABEL_NAMES)}

# Decisions are parsed from XSim stdout lines like:
# DECISION|raw|stable_valid|stable_class|fault|p15|p20|p30|p40|cal|cal_yes|cal_no
decision_log = collections.deque(maxlen=400)
decision_lock = threading.Lock()
xsim_proc = None


def format_decision(value):
    return DECISION_LABELS.get(value, f"UNKNOWN({value})")


def parse_label_names(text):
    names = [part.strip() for part in text.split(",") if part.strip()]
    if not names:
        raise ValueError("at least one label name is required")
    if len(names) > 4:
        raise ValueError("at most four label names are supported")
    while len(names) < 4:
        names.append(f"CLASS{len(names)}")
    return names


def effective_decision(entry):
    if entry is None:
        return None
    if entry.get("fault"):
        return "FAULT"
    if not entry.get("stable_valid", 0):
        return None
    return entry.get("voted")


def format_entry_decision(entry):
    resolved = effective_decision(entry)
    if resolved == "FAULT":
        return "FAULT"
    if resolved is None:
        return "UNCERTAIN"
    return format_decision(resolved)


def count_recent_uncertains(limit):
    streak = 0
    with decision_lock:
        for entry in reversed(decision_log):
            if effective_decision(entry) is None:
                streak += 1
                if streak >= limit:
                    return streak
            else:
                break
    return streak


def set_xsim_proc(proc: subprocess.Popen):
    """Register a running XSim process so the reader thread can parse stdout."""
    global xsim_proc
    xsim_proc = proc
    t = threading.Thread(target=_xsim_reader, args=(proc,), daemon=True)
    t.start()


def _xsim_reader(proc: subprocess.Popen):
    """Background reader for DECISION / STATUS lines from XSim."""
    for raw_line in proc.stdout:
        line = raw_line.strip()
        if line.startswith("DECISION|"):
            parts = line.split("|")
            if len(parts) >= 12:
                entry = {
                    "raw": int(parts[1]),
                    "stable_valid": int(parts[2]),
                    "voted": int(parts[3]),
                    "fault": int(parts[4]),
                    "p15": int(parts[5]) if len(parts) > 5 else None,
                    "p20": int(parts[6]) if len(parts) > 6 else None,
                    "p30": int(parts[7]) if len(parts) > 7 else None,
                    "p40": int(parts[8]) if len(parts) > 8 else None,
                    "calibrated": int(parts[9]) if len(parts) > 9 else 0,
                    "cal_yes": int(parts[10]) if len(parts) > 10 else 0,
                    "cal_no": int(parts[11]) if len(parts) > 11 else 0,
                    "timestamp_s": time.time(),
                }
                with decision_lock:
                    decision_log.append(entry)
            elif len(parts) >= 3:
                entry = {
                    "raw": int(parts[1]),
                    "stable_valid": 1,
                    "voted": int(parts[2]),
                    "fault": 0,
                    "p15": int(parts[3]) if len(parts) > 3 else None,
                    "p20": int(parts[4]) if len(parts) > 4 else None,
                    "p30": int(parts[5]) if len(parts) > 5 else None,
                    "p40": int(parts[6]) if len(parts) > 6 else None,
                    "calibrated": int(parts[7]) if len(parts) > 7 else 0,
                    "cal_yes": int(parts[8]) if len(parts) > 8 else 0,
                    "cal_no": int(parts[9]) if len(parts) > 9 else 0,
                    "timestamp_s": time.time(),
                }
                with decision_lock:
                    decision_log.append(entry)
        elif line.startswith("STATUS|") or line.startswith("WINDOW_DONE|") or line == "READY":
            print(f"[XSim] {line}")


def get_last_decision():
    with decision_lock:
        return decision_log[-1] if decision_log else None


def get_decision_count():
    with decision_lock:
        return len(decision_log)


def wait_for_new_decision(previous_count, timeout_s=1.5):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with decision_lock:
            if len(decision_log) > previous_count:
                return decision_log[-1]
        time.sleep(0.02)
    return None


def get_accuracy(expected_class: int):
    with decision_lock:
        resolved = [entry for entry in decision_log if effective_decision(entry) not in (None, "FAULT")]
        if not resolved:
            return None
        correct = sum(1 for d in resolved if d["voted"] == expected_class)
        return correct / len(resolved)


def print_decision_summary():
    with decision_lock:
        total = len(decision_log)
        if total == 0:
            print("  [No decisions recorded yet]")
            return

        last = decision_log[-1]
        fault_v = sum(1 for d in decision_log if effective_decision(d) == "FAULT")
        unc_v = sum(1 for d in decision_log if effective_decision(d) is None)
        class_counts = {
            label: sum(1 for d in decision_log if effective_decision(d) == idx)
            for idx, label in DECISION_LABELS.items()
        }

    count_text = " | ".join(f"{label}={count}" for label, count in class_counts.items() if count)
    if not count_text:
        count_text = "no stable class decisions yet"
    print(
        f"  Decisions so far: {total} | {count_text} | "
        f"UNCERTAIN={unc_v} | FAULT={fault_v}"
    )
    print(
        "  Last window -> "
        f"raw={format_decision(last['raw'])} "
        f"effective={format_entry_decision(last)} "
        f"stable_valid={last.get('stable_valid', 1)} "
        f"fault={last.get('fault', 0)} "
        f"p15={last['p15']} p20={last['p20']} p30={last['p30']} p40={last['p40']} "
        f"calibrated={last['calibrated']} cal_yes={last['cal_yes']} cal_no={last['cal_no']}"
    )


def wait_for_vivado(timeout_s=COMMAND_TIMEOUT_S):
    deadline = None if timeout_s is None else (time.time() + timeout_s)
    while True:
        try:
            with open(CMD_FILE, "r", encoding="utf-8") as f:
                if f.read().strip() == "":
                    return
        except FileNotFoundError:
            return
        if deadline is not None and time.time() >= deadline:
            raise TimeoutError(
                "Timed out waiting for Vivado/XSim to consume the command. "
                "Start the live testbench and verify vivado_sim/cmd.txt is being cleared."
            )
        time.sleep(0.05)


def send_command(cmd_str, eeg_chunk=None, timeout_s=COMMAND_TIMEOUT_S):
    if eeg_chunk is not None:
        with open(EEG_FILE, "w", encoding="utf-8") as f:
            for val in eeg_chunk:
                f.write(f"{val}\n")

    with open(CMD_FILE, "w", encoding="utf-8") as f:
        f.write(cmd_str)

    wait_for_vivado(timeout_s=timeout_s)


def normalize_path(path_text):
    return os.path.abspath(os.path.expanduser(path_text.strip().strip('"')))


def prompt_for_file(label, provided_path=None, default_path=None):
    if provided_path:
        path = normalize_path(provided_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{label} file not found: {path}")
        return path

    while True:
        if default_path and os.path.isfile(default_path):
            prompt = f"{label} [{default_path}]: "
        else:
            prompt = f"{label}: "

        response = input(prompt).strip()
        candidate = default_path if (not response and default_path and os.path.isfile(default_path)) else response
        if not candidate:
            print("[-] Please provide a valid file path.")
            continue

        path = normalize_path(candidate)
        if os.path.isfile(path):
            return path
        print(f"[-] File not found: {path}")


def load_data(filepath):
    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                samples.append(int(stripped))
            except ValueError as exc:
                raise ValueError(f"Invalid integer in {filepath} on line {line_num}: {stripped}") from exc

    if not samples:
        raise ValueError(f"No EEG samples found in {filepath}")
    return samples


def count_windows(samples):
    return len(samples) // WINDOW_SIZE


def print_buffer_status(name, path, samples, current_idx):
    total_windows = count_windows(samples)
    used_windows = current_idx // WINDOW_SIZE
    remaining_windows = max(total_windows - used_windows, 0)
    print(f"  {name}: {path}")
    print(
        f"    samples={len(samples)} windows={total_windows} "
        f"used={used_windows} remaining={remaining_windows}"
    )


def print_last_decision_entry(entry, prefix="  Last decision"):
    if entry is None:
        print(f"{prefix}: [no DECISION line received yet]")
        return

    print(
        f"{prefix}: raw={format_decision(entry['raw'])} "
        f"effective={format_entry_decision(entry)} "
        f"stable_valid={entry.get('stable_valid', 1)} "
        f"fault={entry.get('fault', 0)} "
        f"p15={entry['p15']} p20={entry['p20']} p30={entry['p30']} p40={entry['p40']} "
        f"calibrated={entry['calibrated']}"
    )


def stream_windows(samples, start_idx, n_windows, label, delay_s=DEFAULT_STREAM_DELAY_S, uncertain_limit=DEFAULT_UNCERTAIN_LIMIT):
    current_idx = start_idx
    sent = 0

    for window_num in range(1, n_windows + 1):
        if current_idx + WINDOW_SIZE > len(samples):
            print(f"[-] Out of {label} data after {sent} windows.")
            break

        chunk = samples[current_idx: current_idx + WINDOW_SIZE]
        previous_decision_count = get_decision_count()
        send_command("RUN", chunk)
        current_idx += WINDOW_SIZE
        sent += 1
        time.sleep(delay_s)

        new_entry = wait_for_new_decision(previous_decision_count, timeout_s=0.75)
        if new_entry is not None:
            recent_uncertain_streak = count_recent_uncertains(uncertain_limit)
            print(
                f"  Window {window_num}/{n_windows} -> "
                f"{format_entry_decision(new_entry)} "
                f"(raw={format_decision(new_entry['raw'])}, "
                f"stable_valid={new_entry.get('stable_valid', 1)}, "
                f"fault={new_entry.get('fault', 0)})"
            )
            if new_entry.get("fault"):
                print("  [!] FPGA fault alert: too many uncertain windows in a row.")
            elif recent_uncertain_streak >= uncertain_limit:
                print("  [!] Warning: repeated uncertain windows suggest the input path or calibration is unstable.")
        else:
            print(f"  Window {window_num}/{n_windows} sent.")

    return current_idx, sent


def run_calibration(label, btn_command, samples, start_idx, cal_windows, uncertain_limit):
    print(f"[>] Starting {label} calibration")
    send_command(btn_command)
    print(f"[+] Sent {btn_command}")
    time.sleep(0.10)

    next_idx, sent = stream_windows(
        samples=samples,
        start_idx=start_idx,
        n_windows=cal_windows,
        label=f"{label} calibration",
        uncertain_limit=uncertain_limit,
    )
    print(f"[+] {label} calibration stream complete: {sent}/{cal_windows} windows sent.")
    print_decision_summary()
    return next_idx, sent


def run_real_stream(samples, start_idx, n_windows, uncertain_limit):
    print(f"[>] Streaming {n_windows} real-data window(s)")
    next_idx, sent = stream_windows(
        samples=samples,
        start_idx=start_idx,
        n_windows=n_windows,
        label="real/demo",
        uncertain_limit=uncertain_limit,
    )
    print(f"[+] Real-data stream complete: {sent}/{n_windows} windows sent.")
    print_decision_summary()
    return next_idx, sent


def print_loaded_files(yes_path, no_path, real_path, yes_samples, no_samples, real_samples, yes_idx, no_idx, real_idx):
    print("\nLoaded EEG files:")
    print_buffer_status("YES calibration", yes_path, yes_samples, yes_idx)
    print_buffer_status("NO calibration", no_path, no_samples, no_idx)
    print_buffer_status("REAL / demo", real_path, real_samples, real_idx)


def reset_indices():
    return 0, 0, 0


def parse_positive_int(text, default_value):
    stripped = text.strip()
    if not stripped:
        return default_value
    value = int(stripped)
    if value <= 0:
        raise ValueError("value must be > 0")
    return value


def main(
    yes_path=None,
    no_path=None,
    real_path=None,
    cal_windows=DEFAULT_CAL_WINDOWS,
    auto_demo=False,
    demo_windows=DEFAULT_DEMO_WINDOWS,
    label_names=",".join(DEFAULT_LABEL_NAMES),
    uncertain_limit=DEFAULT_UNCERTAIN_LIMIT,
):
    global DECISION_LABELS
    DECISION_LABELS = {
        idx: name for idx, name in enumerate(parse_label_names(label_names))
    }

    os.makedirs(VIVADO_SIM_DIR, exist_ok=True)
    with open(CMD_FILE, "w", encoding="utf-8") as f:
        f.write("")

    yes_default = os.path.join(SCRIPT_DIR, "yes_eeg.txt")
    no_default = os.path.join(SCRIPT_DIR, "no_eeg.txt")
    demo_default = os.path.join(SCRIPT_DIR, "demo_eeg.txt")

    yes_path = prompt_for_file("YES calibration EEG file", yes_path, yes_default)
    no_path = prompt_for_file("NO calibration EEG file", no_path, no_default)
    real_path = prompt_for_file("Real / demo EEG file", real_path, demo_default)

    print("\nLoading EEG datasets...")
    yes_samples = load_data(yes_path)
    no_samples = load_data(no_path)
    real_samples = load_data(real_path)

    yes_idx, no_idx, real_idx = reset_indices()

    print("\n" + "=" * 60)
    print(" ROUND-1 SSVEP BCI CONTROLLER")
    print("=" * 60)
    print(f"Vivado sim directory: {VIVADO_SIM_DIR}")
    print(f"Calibration windows per class: {cal_windows}")
    print(f"Decision labels: {', '.join(DECISION_LABELS[idx] for idx in sorted(DECISION_LABELS))}")
    print(f"Uncertain-window fault limit: {uncertain_limit}")
    print_loaded_files(yes_path, no_path, real_path, yes_samples, no_samples, real_samples, yes_idx, no_idx, real_idx)

    if count_windows(yes_samples) < cal_windows:
        print(f"[-] Warning: YES calibration file has only {count_windows(yes_samples)} full windows.")
    if count_windows(no_samples) < cal_windows:
        print(f"[-] Warning: NO calibration file has only {count_windows(no_samples)} full windows.")
    if count_windows(real_samples) == 0:
        print("[-] Warning: Real / demo file has no full 512-sample windows.")

    if auto_demo:
        try:
            yes_idx, _ = run_calibration("YES", "BTN_YES", yes_samples, yes_idx, cal_windows, uncertain_limit)
            no_idx, _ = run_calibration("NO", "BTN_NO", no_samples, no_idx, cal_windows, uncertain_limit)
            real_idx, _ = run_real_stream(real_samples, real_idx, demo_windows, uncertain_limit)
        except TimeoutError as exc:
            print(f"[-] {exc}")

    while True:
        print("\n" + "-" * 60)
        print(
            "Buffers | "
            f"YES {yes_idx // WINDOW_SIZE}/{count_windows(yes_samples)} | "
            f"NO {no_idx // WINDOW_SIZE}/{count_windows(no_samples)} | "
            f"REAL {real_idx // WINDOW_SIZE}/{count_windows(real_samples)}"
        )
        print("[1] Auto YES calibration (BTN_YES + stream calibration windows)")
        print("[2] Auto NO calibration  (BTN_NO  + stream calibration windows)")
        print("[3] Stream 1 real-data window")
        print("[4] Stream N real-data windows")
        print(f"[5] Run full demo from current pointers ({demo_windows} real-data windows)")
        print("[6] Show decision summary")
        print("[7] Show loaded file info")
        print("[8] Rewind all file pointers to the start")
        print("[q] Quit")

        choice = input("Select action: ").strip().lower()

        try:
            if choice == "1":
                yes_idx, _ = run_calibration("YES", "BTN_YES", yes_samples, yes_idx, cal_windows, uncertain_limit)
            elif choice == "2":
                no_idx, _ = run_calibration("NO", "BTN_NO", no_samples, no_idx, cal_windows, uncertain_limit)
            elif choice == "3":
                real_idx, _ = run_real_stream(real_samples, real_idx, 1, uncertain_limit)
            elif choice == "4":
                try:
                    custom_windows = parse_positive_int(
                        input(f"How many real-data windows? [{DEFAULT_DEMO_WINDOWS}]: "),
                        DEFAULT_DEMO_WINDOWS,
                    )
                    real_idx, _ = run_real_stream(real_samples, real_idx, custom_windows, uncertain_limit)
                except ValueError as exc:
                    print(f"[-] Invalid number of windows: {exc}")
            elif choice == "5":
                yes_idx, _ = run_calibration("YES", "BTN_YES", yes_samples, yes_idx, cal_windows, uncertain_limit)
                no_idx, _ = run_calibration("NO", "BTN_NO", no_samples, no_idx, cal_windows, uncertain_limit)
                real_idx, _ = run_real_stream(real_samples, real_idx, demo_windows, uncertain_limit)
            elif choice == "6":
                print_decision_summary()
            elif choice == "7":
                print_loaded_files(
                    yes_path, no_path, real_path,
                    yes_samples, no_samples, real_samples,
                    yes_idx, no_idx, real_idx,
                )
                print_last_decision_entry(get_last_decision())
            elif choice == "8":
                yes_idx, no_idx, real_idx = reset_indices()
                print("[+] Rewound YES, NO, and real-data file pointers.")
            elif choice == "q":
                break
            else:
                print("[-] Invalid choice.")
        except TimeoutError as exc:
            print(f"[-] {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay EEG text files into the Vivado SSVEP testbench.")
    parser.add_argument("--yes_calib", help="Path to the YES calibration EEG text file.")
    parser.add_argument("--no_calib", help="Path to the NO calibration EEG text file.")
    parser.add_argument("--real_eeg", help="Path to the real/demo EEG text file.")
    parser.add_argument(
        "--cal_windows",
        type=int,
        default=DEFAULT_CAL_WINDOWS,
        help="Number of 512-sample windows to stream for each calibration class.",
    )
    parser.add_argument(
        "--auto_demo",
        action="store_true",
        help="Run YES calibration, NO calibration, and a short real-data demo immediately.",
    )
    parser.add_argument(
        "--demo_windows",
        type=int,
        default=DEFAULT_DEMO_WINDOWS,
        help="Number of real/demo windows to stream during --auto_demo.",
    )
    parser.add_argument(
        "--labels",
        default=",".join(DEFAULT_LABEL_NAMES),
        help="Comma-separated class labels for outputs 0..3. Default matches the active 15 Hz / 20 Hz demo header.",
    )
    parser.add_argument(
        "--uncertain_limit",
        type=int,
        default=DEFAULT_UNCERTAIN_LIMIT,
        help="Consecutive uncertain windows before reporting a fault-style warning.",
    )
    args = parser.parse_args()

    main(
        yes_path=args.yes_calib,
        no_path=args.no_calib,
        real_path=args.real_eeg,
        cal_windows=args.cal_windows,
        auto_demo=args.auto_demo,
        demo_windows=args.demo_windows,
        label_names=args.labels,
        uncertain_limit=args.uncertain_limit,
    )
