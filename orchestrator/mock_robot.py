"""Simulated robot for dry-run testing — ACT version.

Replaces fast_loop.run() entirely. Fills SharedState.obs with realistic fake data
and advances joint/gripper state based on the active policy name, mimicking what
a trained ACT checkpoint would do for each phase.

Simulated behaviour per phase name:
  Contains "grasp"   → gripper closes 90→8 over ~4 s
  Contains "insert"  → elbow_flex moves toward -0.6 over ~5 s
  Contains "press"   → wrist_flex moves toward -0.9 over ~4 s
  Contains "release" → gripper opens 8→90 over ~2 s
  anything else      → joints hold current position

Camera frames are PIL images with a coloured background + policy name label +
a gripper progress bar, so you can visually verify what the agent is seeing.
"""

import logging
import time
from typing import Any

import numpy as np

try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from .shared_state import SharedState

logger = logging.getLogger(__name__)

TARGET_HZ = 20
LOOP_DT = 1.0 / TARGET_HZ

RESTING_STATE = {
    "shoulder_pan":   0.0,
    "shoulder_lift":  0.3,
    "elbow_flex":     0.1,
    "wrist_flex":     0.0,
    "wrist_roll":     0.0,
    "gripper":       90.0,
}

# Joint targets keyed by substring of phase name
PHASE_TARGETS = {
    "grasp":   {"gripper": 8.0},
    "insert":  {"elbow_flex": -0.6, "shoulder_lift": 0.6},
    "press":   {"wrist_flex": -0.9},
    "lever":   {"wrist_flex": -0.9},
    "release": {"gripper": 90.0},
}

SPEEDS = {
    "shoulder_pan":  0.4,
    "shoulder_lift": 0.3,
    "elbow_flex":    0.3,
    "wrist_flex":    0.4,
    "wrist_roll":    0.4,
    "gripper":      25.0,
}

PHASE_COLOURS = {
    "grasp":   (60, 120, 60),
    "insert":  (60, 60, 140),
    "press":   (140, 80, 60),
    "lever":   (140, 80, 60),
    "release": (100, 100, 60),
}
DEFAULT_COLOUR = (80, 80, 80)


def _targets_for_phase(phase_name: str) -> dict:
    phase_lower = phase_name.lower()
    targets = {}
    for kw, t in PHASE_TARGETS.items():
        if kw in phase_lower:
            targets.update(t)
    return targets if targets else {}


def _step_toward(current: float, target: float, speed: float, dt: float) -> float:
    delta = target - current
    max_step = speed * dt
    if abs(delta) <= max_step:
        return target
    return current + max_step * (1 if delta > 0 else -1)


def _make_camera_frame(phase_name: str, joint_state: dict, width: int = 320, height: int = 240) -> np.ndarray:
    colour = DEFAULT_COLOUR
    for kw, col in PHASE_COLOURS.items():
        if kw in phase_name.lower():
            colour = col
            break

    if PIL_AVAILABLE:
        img = Image.new("RGB", (width, height), colour)
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), f"PHASE: {phase_name}", fill=(255, 255, 255))
        draw.text((10, 30), "ACT policy active", fill=(200, 200, 200))
        g = joint_state["gripper"]
        bar_w = int((g / 100.0) * (width - 20))
        draw.rectangle([10, height - 30, 10 + bar_w, height - 10], fill=(200, 200, 80))
        draw.text((10, height - 45), f"gripper: {g:.1f}", fill=(255, 255, 255))
        return np.array(img, dtype=np.uint8)
    else:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = colour
        return frame


def run(state: SharedState, cameras: dict, **_kwargs) -> None:
    """Entry point — drop-in replacement for fast_loop.run() in dry-run mode.

    Extra kwargs (robot_port, phase_model_map, device_str) are accepted and ignored.
    """
    logger.info("Mock robot starting (no hardware, no ACT models)")
    joint_state = dict(RESTING_STATE)
    camera_names = list(cameras.keys()) if cameras else ["front_cam"]
    current_phase = ""

    try:
        while not state.stop:
            loop_start = time.perf_counter()

            with state.lock:
                policy_name = state.active_policy_name
                should_reset = state.reset_requested

            # On policy switch, snap joints back toward resting (simulates arm repositioning)
            if should_reset and policy_name != current_phase:
                current_phase = policy_name
                with state.lock:
                    state.reset_requested = False
                logger.debug("Mock: switched to phase '%s'", policy_name)

            if state.paused or not policy_name:
                with state.lock:
                    state.obs = _build_obs(joint_state, camera_names, policy_name or "idle")
                time.sleep(LOOP_DT)
                continue

            # Simulate joint movement based on active phase
            targets = _targets_for_phase(policy_name)
            for joint, current_val in joint_state.items():
                target = targets.get(joint, RESTING_STATE[joint])
                joint_state[joint] = _step_toward(current_val, target, SPEEDS[joint], LOOP_DT)

            with state.lock:
                state.obs = _build_obs(joint_state, camera_names, policy_name)

            elapsed = time.perf_counter() - loop_start
            sleep_time = LOOP_DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception:
        logger.exception("Mock robot loop crashed")
    finally:
        logger.info("Mock robot stopped")


def _build_obs(joint_state: dict, camera_names: list, phase_name: str) -> dict[str, Any]:
    joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    state_array = np.array([joint_state[j] for j in joints], dtype=np.float32)
    obs: dict[str, Any] = {"observation.state": state_array}
    for cam in camera_names:
        obs[f"observation.images.{cam}"] = _make_camera_frame(phase_name, joint_state)
    return obs
