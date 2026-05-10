"""Single-command launch for the ACT orchestrator.

Auto-discovers serial port and cameras, verifies calibration, saves camera
previews for visual confirmation, then starts the full orchestrator.

Usage:
    conda activate lerobot
    export ANTHROPIC_API_KEY=sk-ant-...
    python -m orchestrator.start

Override any discovered value:
    python -m orchestrator.start --port /dev/tty.usbserial-XXX
    python -m orchestrator.start --cameras front_cam:0 side_cam:2
    python -m orchestrator.start --dry-run   # mock robot, no hardware needed
"""

import argparse
import logging
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


# ── Logging setup ────────────────────────────────────────────────────────────

def setup_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"run_{timestamp}.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fmt_detailed = logging.Formatter(
        "%(asctime)s.%(msecs)03d  %(levelname)-8s  %(name)-30s  %(message)s",
        datefmt="%H:%M:%S",
    )
    fmt_console = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # File: everything at DEBUG
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt_detailed)
    root.addHandler(fh)

    # Console: INFO and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt_console)
    root.addHandler(ch)

    # Silence noisy libs in console (still go to file)
    for noisy in ("httpx", "httpcore", "anthropic._base_client", "strands.telemetry"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return log_file


# ── Hardware discovery ────────────────────────────────────────────────────────

def find_serial_port() -> str | None:
    """Return the first USB serial port that looks like an SO-101 arm."""
    try:
        import serial.tools.list_ports
    except ImportError:
        logging.getLogger(__name__).warning("pyserial not available — cannot auto-detect port")
        return None

    log = logging.getLogger(__name__)
    candidates = [
        p.device for p in serial.tools.list_ports.comports()
        if any(kw in (p.device + (p.description or "")).lower()
               for kw in ["usbserial", "usbmodem", "ttyusb", "ch340", "cp210", "ft232", "feetech"])
    ]

    if not candidates:
        # Broader fallback
        candidates = [
            p.device for p in serial.tools.list_ports.comports()
            if "usb" in p.device.lower() or "tty" in p.device.lower()
        ]

    log.info("Serial port candidates: %s", candidates)

    if len(candidates) == 1:
        log.info("Auto-selected port: %s", candidates[0])
        return candidates[0]

    if len(candidates) > 1:
        log.warning("Multiple serial ports found: %s — using first. Override with --port if wrong.", candidates)
        return candidates[0]

    log.error("No serial ports found. Is the robot plugged in and powered on?")
    return None


def discover_cameras(save_dir: Path) -> dict[int, Path]:
    """Probe OpenCV indices 0-6, save preview images, return {index: image_path}."""
    import cv2
    import numpy as np

    log = logging.getLogger(__name__)
    found: dict[int, Path] = {}
    save_dir.mkdir(parents=True, exist_ok=True)

    for i in range(7):
        cap = cv2.VideoCapture(i)
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None and frame.size > 0:
            path = save_dir / f"camera_{i}_preview.jpg"
            cv2.imwrite(str(path), frame)
            h, w = frame.shape[:2]
            log.info("Camera %d: FOUND (%dx%d) → preview saved to %s", i, w, h, path)
            found[i] = path
        else:
            log.debug("Camera %d: not found", i)

    return found


def assign_cameras(found: dict[int, Path], required: list[str]) -> dict[str, int] | None:
    """Try to assign camera names to indices. Returns None if cannot auto-assign."""
    log = logging.getLogger(__name__)

    if len(found) == 0:
        log.error("No cameras found. Check USB connections.")
        return None

    if len(found) == len(required):
        # Exactly the right number — assign in order
        assignment = {name: idx for name, idx in zip(required, sorted(found.keys()))}
        log.info("Camera assignment (by order): %s", assignment)
        log.info("Preview images saved to outputs/camera_previews/ — verify assignment is correct.")
        return assignment

    log.warning(
        "Found %d cameras but need %d (%s). "
        "Use --cameras to specify manually (e.g. --cameras front_cam:0 side_cam:2).",
        len(found), len(required), required,
    )
    return None


# ── Main ─────────────────────────────────────────────────────────────────────

TASK = "pick up bread, place it in the toaster, press the lever, wait for toast, remove bread"

# Each phase needs a separately trained ACT checkpoint.
# Phases marked TODO need training — comment them out until models are ready.
PHASES = [
    ("pick_and_drop",       "qualia-robotics/act-makermods-pick-and-drop-bread-4c417ee8"),
    ("press_lever",         "qualia-robotics/act-makermods-press-toast-button-4302c182"),
    ("wait_for_toast",      None),  # no ACT model — robot pauses, agent watches for lever pop
    ("pick_out_of_toaster", "qualia-robotics/act-makermods-put-back-the-toast-0d9730ec"),
]

# Per-phase instructions for the vision model — what to look for, what camera,
# and the exact trigger condition to advance to the next phase.
PHASE_PROMPTS = {
    "pick_and_drop": """
PHASE: pick_and_drop
ACT policy: picking up bread from counter and dropping into toaster slot.
Wait: 12 seconds for the policy to complete (it may need 2 action chunks).

END-OF-PHASE CHECK — use side_cam (PRIMARY success signal):
  Look for: bread end visible and protruding from the toaster slot (rectangular opening).
  The bread should be partially inside the slot, not lying on the counter.
  THIS IS YOUR PRIMARY SUCCESS CRITERION — trust the camera over the gripper sensor.

GRIPPER NOTE: For this robot's calibration, gripper values stay in the 0-10 range.
  Do NOT use check_gripper_closed. Use camera vision ONLY to assess success.

ADVANCE TO press_lever when: camera shows bread is in the toaster slot.
RETRY once if: bread is clearly still on the counter and not in the slot.
If retry also fails: call complete_task with a failure summary.
""",

    "press_lever": """
PHASE: press_lever
ACT policy: pressing the toaster lever down to start toasting.
Wait: 8 seconds for the policy to complete.

END-OF-PHASE CHECK — use front_cam (CAMERA IS THE ONLY SUCCESS SIGNAL):
  Look for: toaster lever handle visibly in the DOWN position, pushed toward the
  body of the toaster. The handle should be clearly lower than its resting height.
  Do NOT use check_joint_angle — joint values are not reliable for this check.

ADVANCE TO wait_for_toast when: camera shows lever is DOWN or has been pressed.
  If the lever was attempted but you cannot clearly confirm, advance anyway after 8s.
  Do not retry more than once — move forward to keep the task progressing.
""",

    "wait_for_toast": """
PHASE: wait_for_toast
No ACT policy — robot is paused while toast completes.
Call pause_robot() immediately. Do NOT call advance_phase for this phase.

MONITORING — use side_cam every 3 seconds:
  Look for: toaster lever has POPPED BACK UP to its resting (upright) position.
  This spring-loaded return means the toast cycle is complete.
  Do NOT advance until the lever is clearly back in the UP position.

Once lever is back UP: call advance_phase("pick_out_of_toaster").
If lever does not pop after 120 seconds: call complete_task with a timeout note.
""",

    "pick_out_of_toaster": """
PHASE: pick_out_of_toaster
ACT policy: reaching into toaster and removing the bread.
Wait: 6 seconds for the policy to complete.

END-OF-PHASE CHECK — use side_cam:
  Look for: bread is no longer in the toaster slot AND gripper is holding bread
  OR bread has been placed on the counter surface next to toaster.

SENSOR CHECK: check_gripper_closed(threshold=15)
  Expected: is_closed=True (gripper gripping bread) OR bread visible on surface.

ADVANCE TO complete_task when: bread is out of toaster.
RETRY if: bread still visible in slot AND gripper open.
""",
}

REQUIRED_CAMERAS = ["front_cam", "side_cam"]


def main() -> None:
    parser = argparse.ArgumentParser(description="ACT orchestrator — auto-discovery launcher")
    parser.add_argument("--port",    help="Serial port override (skip auto-detect)")
    parser.add_argument("--cameras", nargs="+", metavar="NAME:INDEX",
                        help="Camera assignments e.g. front_cam:0 side_cam:1")
    parser.add_argument("--device",  default="mps", help="Torch device (mps/cuda/cpu)")
    parser.add_argument("--claude-model", default="claude-sonnet-4-6")
    parser.add_argument("--dry-run", action="store_true", help="Use mock robot (no hardware)")
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    log_file  = setup_logging(repo_root / "outputs" / "logs")
    log       = logging.getLogger(__name__)

    log.info("=" * 60)
    log.info("ACT Orchestrator starting")
    log.info("Log file: %s", log_file)
    log.info("Task:  %s", TASK)
    log.info("=" * 60)

    # ── Load .env if present ─────────────────────────────────────────────────
    env_path = repo_root / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))
        log.debug("Loaded .env from %s", env_path)

    # ── API key check ────────────────────────────────────────────────────────
    if not os.environ.get("ANTHROPIC_API_KEY"):
        log.error("ANTHROPIC_API_KEY not set. Run: export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    # ── Calibration check ────────────────────────────────────────────────────
    cal_path = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower/auto_follower_left.json"
    if cal_path.exists():
        log.info("Calibration file found: %s", cal_path)
    else:
        log.error("Calibration file missing at %s", cal_path)
        log.error("Run: cp orchestrator/auto_follower_left.json %s", cal_path)
        sys.exit(1)

    # ── Serial port ──────────────────────────────────────────────────────────
    if args.dry_run:
        port = "mock"
        log.info("Dry-run mode — skipping hardware discovery")
    elif args.port:
        port = args.port
        log.info("Port (manual): %s", port)
    else:
        port = find_serial_port()
        if not port:
            log.error("Could not find serial port. Plug in the robot arm and retry, or use --port.")
            sys.exit(1)

    # ── Cameras ──────────────────────────────────────────────────────────────
    if args.cameras:
        cameras: dict[str, dict] = {}
        for entry in args.cameras:
            name, _, idx = entry.partition(":")
            cameras[name.strip()] = {"index_or_path": int(idx.strip()), "width": 640, "height": 480, "fps": 30}
        log.info("Cameras (manual): %s", {k: v["index_or_path"] for k, v in cameras.items()})
    elif args.dry_run:
        cameras = {name: {"index_or_path": i} for i, name in enumerate(REQUIRED_CAMERAS)}
        log.info("Cameras (dry-run): %s", {k: v["index_or_path"] for k, v in cameras.items()})
    else:
        preview_dir = repo_root / "outputs" / "camera_previews"
        found = discover_cameras(preview_dir)
        if not found:
            log.error("No cameras found. Check connections or use --cameras.")
            sys.exit(1)
        assignment = assign_cameras(found, REQUIRED_CAMERAS)
        if assignment is None:
            log.error("Cannot auto-assign cameras. Specify manually: --cameras front_cam:0 side_cam:1")
            log.info("Preview images saved to: %s", preview_dir)
            sys.exit(1)
        cameras = {
            name: {"index_or_path": idx, "width": 640, "height": 480, "fps": 30}
            for name, idx in assignment.items()
        }

    # ── Summary before launch ────────────────────────────────────────────────
    log.info("-" * 60)
    log.info("Hardware summary:")
    log.info("  Port:    %s", port)
    log.info("  Cameras: %s", {k: v["index_or_path"] for k, v in cameras.items()})
    log.info("  Device:  %s", args.device)
    log.info("Phases:")
    for name, model_id in PHASES:
        log.info("  [%s] %s", name, model_id)
    log.info("-" * 60)

    # ── Launch ───────────────────────────────────────────────────────────────
    from .shared_state import SharedState
    from .tools import make_tools
    from .prompts import SYSTEM_PROMPT, build_initial_prompt

    state = SharedState()
    phase_names = [name for name, _ in PHASES]

    if args.dry_run:
        from .mock_robot import run as run_loop
        loop_kwargs = {"state": state, "cameras": cameras}
    else:
        from .fast_loop import run as run_loop
        loop_kwargs = {
            "state": state,
            "robot_port": port,
            "cameras": cameras,
            "phase_model_map": {name: mid for name, mid in PHASES},  # None entries handled in fast_loop
            "device_str": args.device,
        }

    loop_thread = threading.Thread(
        target=run_loop,
        kwargs=loop_kwargs,
        daemon=True,
        name="act-loop",
    )
    loop_thread.start()

    # Wait for first observation
    log.info("Waiting for robot/mock to come online...")
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        if state.snapshot_obs():
            break
        time.sleep(0.2)
    else:
        log.error("Robot did not produce an observation within 60 s. Check connection.")
        sys.exit(1)

    log.info("Robot online. Starting Strands orchestrator.")

    from strands import Agent
    from strands.models.anthropic import AnthropicModel

    model = AnthropicModel(
        model_id=args.claude_model,
        max_tokens=4096,
        use_native_token_count=False,
    )
    tools = make_tools(state, phase_names)
    agent = Agent(model=model, tools=tools, system_prompt=SYSTEM_PROMPT)

    prompt = build_initial_prompt(TASK, PHASES, PHASE_PROMPTS)
    log.info("Orchestrator running. Watching for phase transitions...")
    log.info("(Full debug log at: %s)", log_file)

    try:
        response = agent(prompt)
        log.info("Orchestrator finished: %s", response)
    except KeyboardInterrupt:
        log.info("Interrupted — stopping robot")
        state.stop = True

    loop_thread.join(timeout=5.0)
    log.info("Shutdown complete. Full log: %s", log_file)


if __name__ == "__main__":
    main()
