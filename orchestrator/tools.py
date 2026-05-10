"""Strands agent tools for observing and controlling the robot (ACT version).

All tools close over a single SharedState instance and a list of valid phase names.
Call make_tools(state, phase_names) to get the bound list to pass to the Strands Agent.

Tool categories:
  Observe   — get_robot_state, capture_camera_frame
  Control   — pause_robot, resume_robot
  Phase     — advance_phase, get_phase_status
  Sensor    — check_gripper_closed, check_joint_angle
  Terminate — complete_task
"""

import base64
import io
import logging
import time
from typing import Any

import numpy as np

from .shared_state import SharedState

logger = logging.getLogger(__name__)


def make_tools(state: SharedState, phase_names: list[str]) -> list:
    """Return Strands-compatible tool functions bound to `state`.

    Args:
        state:        Shared state instance.
        phase_names:  Ordered list of valid phase names (e.g. ["grasp", "insert", "press"]).
                      Used to validate calls and to know what comes next.
    """
    from strands import tool

    # ── Observe ─────────────────────────────────────────────────────────────

    @tool
    def get_robot_state() -> dict[str, Any]:
        """Return current joint positions and gripper value from the robot.

        Joint angles are normalised floats in roughly [-1, 1] (body joints) or
        [0, 100] (gripper). Gripper near 0 = closed/gripping; near 100 = open.
        """
        obs = state.snapshot_obs()
        if not obs:
            return {"error": "No observation available yet — fast loop may not have started"}

        joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        result = {}
        for joint in joints:
            key = f"{joint}.pos"
            if key in obs:
                v = obs[key]
                result[joint] = round(float(v), 3)
        if not result:
            return {"error": f"No joint keys found. Available keys: {list(obs.keys())}"}
        logger.debug("get_robot_state: %s", result)
        return result

    @tool
    def capture_camera_frame(camera_name: str = "front_cam") -> dict[str, str]:
        """Capture the latest frame from a named camera and return it as base64 JPEG.

        Use this to visually assess the scene — is the object grasped, is the bread
        in the toaster slot, is the lever down? This is your primary state-assessment tool.

        Args:
            camera_name: Camera name as configured (e.g. "front_cam", "hand_cam").
        """
        obs = state.snapshot_obs()
        # Raw obs uses camera name directly; formatted obs uses observation.images.<name>
        img_key = camera_name if camera_name in obs else f"observation.images.{camera_name}"

        if img_key not in obs:
            available = [k for k in obs if isinstance(obs[k], np.ndarray) and obs[k].ndim == 3]
            return {"error": f"Camera '{camera_name}' not found. Available: {available}"}

        frame = obs[img_key]
        if not isinstance(frame, np.ndarray):
            return {"error": f"Expected numpy array, got {type(frame)}"}

        from PIL import Image
        img = Image.fromarray(frame)
        w, h = img.size
        if w > 512:
            img = img.resize((512, int(h * 512 / w)), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return {
            "type": "image",
            "media_type": "image/jpeg",
            "data": b64,
            "description": f"Latest frame from {camera_name} ({frame.shape[1]}x{frame.shape[0]})",
        }

    # ── Control ─────────────────────────────────────────────────────────────

    @tool
    def wait_for_phase(seconds: float) -> str:
        """Wait for the ACT policy to complete its action sequence before checking state.

        Call this immediately after advance_phase and before any sensor checks.
        The ACT policy runs its full action chunk during this time — do not interrupt.

        Typical durations:
          - Grasp phase:   4–6 s  (reach + close gripper)
          - Insert phase:  5–8 s  (transport + insert)
          - Press phase:   3–5 s  (press lever)

        Args:
            seconds: How long to wait. Base this on how long the phase action takes.
        """
        logger.info("wait_for_phase: sleeping %.1fs for phase '%s'", seconds, state.phase)
        time.sleep(seconds)
        logger.info("wait_for_phase: done — ready for end-of-phase check")
        return f"Waited {seconds}s. ACT policy has completed its action sequence. Ready for end-of-phase check."

    @tool
    def pause_robot() -> str:
        """Pause robot motion. The fast loop stops sending actions immediately.

        Use during phase transitions or when you need a stable observation.
        Call resume_robot() to restart. Don't hold pause longer than 3 seconds
        unless the arm is in a safe resting pose (gravity will cause drift).
        """
        with state.lock:
            state.paused = True
        return "Robot paused"

    @tool
    def resume_robot() -> str:
        """Resume robot motion after a pause."""
        with state.lock:
            state.paused = False
        return "Robot resumed"

    # ── Phase management ────────────────────────────────────────────────────

    @tool
    def advance_phase(phase_name: str) -> str:
        """Switch to the next phase by activating its pre-trained ACT policy.

        This is the primary phase transition tool. It:
        1. Pauses the robot for 300 ms (arm stabilises)
        2. Signals the fast loop to flush the outgoing policy's action queue
        3. Switches the active policy to the new phase
        4. Resumes the robot

        The fast loop will begin running the new phase's ACT policy within one cycle (~50 ms).
        Wait at least 1.5 seconds after calling this before checking sensor state — ACT
        needs time to fill its action chunk queue from the new policy.

        Args:
            phase_name: Must be one of the configured phases: {phase_names}
        """
        if phase_name not in phase_names:
            return f"Unknown phase '{phase_name}'. Valid phases: {phase_names}"

        with state.lock:
            state.paused = True

        time.sleep(0.3)

        state.set_phase(phase_name)
        state.set_active_policy(phase_name)   # also sets reset_requested = True

        with state.lock:
            state.paused = False

        logger.info("Phase → %s (ACT policy switched)", phase_name)
        return f"Switched to phase '{phase_name}'. ACT policy now active. Wait 1.5 s before sensor checks."

    @tool
    def get_phase_status() -> dict[str, Any]:
        """Return current phase name, elapsed seconds in this phase, and active policy."""
        return {
            "phase": state.phase,
            "active_policy": state.active_policy_name,
            "elapsed_s": round(state.elapsed_in_phase(), 1),
            "paused": state.paused,
            "available_phases": phase_names,
        }

    # ── Sensor checks ────────────────────────────────────────────────────────

    @tool
    def check_gripper_closed(threshold: float = 15.0) -> dict[str, Any]:
        """Check whether the gripper is holding an object.

        Gripper value: 0 = fully closed, 100 = fully open.
        A value below threshold strongly indicates the gripper is gripping something.

        Args:
            threshold: Gripper value below which we consider it closed (default 15).
                       Use 8 for thin objects (levers, cards).
        """
        obs = state.snapshot_obs()
        if "gripper.pos" not in obs:
            return {"error": f"No gripper reading. Keys: {list(obs.keys())}"}
        gripper_val = float(obs["gripper.pos"])
        result = {
            "gripper_value": round(gripper_val, 2),
            "is_closed": gripper_val < threshold,
            "threshold": threshold,
        }
        logger.info("check_gripper_closed: value=%.2f threshold=%.1f is_closed=%s",
                    gripper_val, threshold, result["is_closed"])
        return result

    @tool
    def check_joint_angle(joint_name: str, comparator: str, threshold: float) -> dict[str, Any]:
        """Check if a specific joint satisfies a threshold condition.

        Useful for detecting whether the arm has reached a target pose, e.g.:
        - wrist_flex < -0.8  →  lever is pressed down
        - shoulder_lift > 0.5  →  arm raised for transport

        Args:
            joint_name:  One of: shoulder_pan, shoulder_lift, elbow_flex,
                         wrist_flex, wrist_roll, gripper.
            comparator:  One of: "<", ">", "<=", ">=".
            threshold:   Float value to compare against.
        """
        joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        if joint_name not in joints:
            return {"error": f"Unknown joint '{joint_name}'. Valid: {joints}"}

        obs = state.snapshot_obs()
        key = f"{joint_name}.pos"
        if key not in obs:
            return {"error": f"Joint '{joint_name}' not found. Keys: {list(obs.keys())}"}
        val = float(obs[key])

        ops = {"<": val < threshold, ">": val > threshold, "<=": val <= threshold, ">=": val >= threshold}
        if comparator not in ops:
            return {"error": f"Unknown comparator '{comparator}'. Use: <, >, <=, >="}

        result = {
            "joint": joint_name,
            "value": round(val, 3),
            "condition": f"{joint_name} {comparator} {threshold}",
            "satisfied": ops[comparator],
        }
        logger.info("check_joint_angle: %s=%.3f  %s %.3f  satisfied=%s",
                    joint_name, val, comparator, threshold, result["satisfied"])
        return result

    # ── Home position ────────────────────────────────────────────────────────

    # Relaxed, low-torque resting pose (joint values in robot's normalised units).
    # Gripper open (100), arm retracted roughly overhead-neutral.
    # Keys must end in .pos to match robot.send_action() expected format
    _DEFAULT_HOME: dict[str, float] = {
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": 0.0,
        "elbow_flex.pos": 0.0,
        "wrist_flex.pos": 0.0,
        "wrist_roll.pos": 0.0,
        "gripper.pos": 100.0,
    }

    @tool
    def go_to_home_position(
        timeout: float = 10.0,
        position: dict[str, float] | None = None,
    ) -> str:
        """Drive the arm to a neutral resting pose between phases.

        Sends fixed joint-position commands at 20 Hz until all joints converge
        (within 0.05 of target) or timeout expires. The gripper is opened fully.
        ACT inference is suspended while homing; it resumes automatically once
        the home position is reached.

        Call this after confirming a phase succeeded and before advance_phase for
        the next phase, so each phase starts from a known configuration.

        Args:
            timeout:  Maximum seconds to wait for convergence (default 10).
            position: Optional joint override dict. Omit to use the default home pose.
                      Keys: shoulder_pan, shoulder_lift, elbow_flex,
                            wrist_flex, wrist_roll, gripper.
        """
        target = {**_DEFAULT_HOME, **(position or {})}

        with state.lock:
            state.paused = True

        time.sleep(0.1)  # let any in-flight action finish

        with state.lock:
            state.home_position = target
            state.home_requested = True
            state.paused = False  # fast loop drives homing, must not be paused

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with state.lock:
                done = not state.home_requested
            if done:
                return f"Home position reached. Ready for next phase."
            time.sleep(0.1)

        with state.lock:
            state.home_requested = False  # abort
        return f"Warning: home position not fully reached within {timeout}s — proceeding anyway."

    # ── Termination ─────────────────────────────────────────────────────────

    @tool
    def complete_task(summary: str = "") -> str:
        """Signal that the full task is complete and shut down the robot.

        Call only when ALL phases are done and the robot is in a safe resting state
        (gripper open, arm lowered). Once called, the process exits.

        Args:
            summary: Optional description of what was accomplished.
        """
        with state.lock:
            state.paused = True
            state.stop = True
        logger.info("Task complete: %s", summary or "(no summary)")
        return f"Task complete. Robot shutting down. {summary}"

    return [
        get_robot_state,
        capture_camera_frame,
        wait_for_phase,
        pause_robot,
        resume_robot,
        advance_phase,
        get_phase_status,
        check_gripper_closed,
        check_joint_angle,
        complete_task,
    ]
