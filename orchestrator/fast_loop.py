"""20 Hz ACT inference loop — runs in a background thread.

Responsibilities:
  1. Connect to the SO-101 follower arm and cameras.
  2. Load all ACT policies upfront (one per phase) into a dict.
  3. Every 50 ms: read observation → run active policy → send action to robot.
  4. When SharedState.reset_requested is True, call policy.reset() to flush
     the outgoing policy's action chunk queue before switching.
  5. Honour pause/stop signals from the orchestrator.

ACT takes NO language instruction — behaviour is determined entirely by which
checkpoint is active. Switching policy = switching behaviour.
"""

import logging
import time
from typing import Any

import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.robots import make_robot_from_config
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.utils.constants import OBS_STR
from lerobot.utils.control_utils import predict_action

from .shared_state import SharedState

logger = logging.getLogger(__name__)

TARGET_HZ = 20
LOOP_DT = 1.0 / TARGET_HZ


def make_robot(port: str, cameras: dict[str, dict]) -> Any:
    """Build and connect an SO-101 follower from port + camera dict."""
    camera_configs = {
        name: OpenCVCameraConfig(
            index_or_path=cfg["index_or_path"],
            width=cfg.get("width", 640),
            height=cfg.get("height", 480),
            fps=cfg.get("fps", 30),
        )
        for name, cfg in cameras.items()
    }
    robot_cfg = SO101FollowerConfig(port=port, cameras=camera_configs, id="auto_follower_left")
    robot = make_robot_from_config(robot_cfg)
    robot.connect()
    logger.info("Robot connected on %s with cameras: %s", port, list(cameras))
    return robot


def load_policies(phase_model_map: dict[str, str], device: torch.device) -> dict[str, tuple]:
    """Load one ACT policy + processors per phase.

    Args:
        phase_model_map: {phase_name: hf_repo_id_or_local_path}
        device: torch device

    Returns:
        {phase_name: (policy, preprocessor, postprocessor)}
    """
    loaded = {}
    for phase_name, model_id in phase_model_map.items():
        if model_id is None:
            logger.info("Phase '%s': no model (observation-only phase)", phase_name)
            continue
        logger.info("Loading ACT policy for phase '%s' from %s ...", phase_name, model_id)
        policy = ACTPolicy.from_pretrained(model_id)
        policy = policy.to(device)
        policy.eval()
        preprocessor, postprocessor = make_pre_post_processors(
            policy.config,
            pretrained_path=model_id,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )
        loaded[phase_name] = (policy, preprocessor, postprocessor)
        n_params = sum(p.numel() for p in policy.parameters())
        logger.info("  Loaded '%s' (%s parameters)", phase_name, f"{n_params:,}")
    return loaded


def run(
    state: SharedState,
    robot_port: str,
    cameras: dict,
    phase_model_map: dict[str, str],
    device_str: str = "mps",
) -> None:
    """Entry point — call this from a daemon thread.

    Args:
        state:            Shared state written by orchestrator, read here each cycle.
        robot_port:       Serial port of the SO-101 follower arm.
        cameras:          Dict of camera configs (name → {index_or_path, width, height, fps}).
        phase_model_map:  {phase_name: hf_repo_id_or_local_path} for all phases.
        device_str:       Torch device ("mps" on Apple Silicon, "cuda" for GPU, "cpu" fallback).
    """
    device = torch.device(device_str)
    robot = make_robot(robot_port, cameras)
    policies = load_policies(phase_model_map, device)

    lerobot_features = hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=False)
    action_names: list[str] = list(robot.action_features.keys())

    logger.info("Fast loop starting at %d Hz with %d loaded policies: %s",
                TARGET_HZ, len(policies), list(policies.keys()))

    try:
        while not state.stop:
            loop_start = time.perf_counter()

            if state.paused:
                time.sleep(LOOP_DT)
                continue

            with state.lock:
                home_req = state.home_requested
                home_pos = dict(state.home_position)

            if home_req and home_pos:
                obs_raw = robot.get_observation()
                with state.lock:
                    state.obs = obs_raw

                robot.send_action(home_pos)

                # Check convergence: all joints within tolerance
                at_home = all(
                    abs(float(obs_raw.get(f"{j}.pos", home_pos[j])) - home_pos[j]) < 0.05
                    for j in home_pos
                )
                if at_home:
                    with state.lock:
                        state.home_requested = False
                    logger.info("Home position reached — resuming normal inference")

                elapsed = time.perf_counter() - loop_start
                time.sleep(max(0.0, LOOP_DT - elapsed))
                continue

            with state.lock:
                policy_name = state.active_policy_name
                should_reset = state.reset_requested

            if not policy_name or policy_name not in policies:
                # No active policy yet — still capture obs so main thread can start
                obs_raw = robot.get_observation()
                with state.lock:
                    state.obs = obs_raw
                time.sleep(LOOP_DT)
                continue

            policy, preprocessor, postprocessor = policies[policy_name]

            # Flush action queue when the orchestrator requests a policy switch
            if should_reset:
                policy.reset()
                with state.lock:
                    state.reset_requested = False
                logger.info("Policy '%s' action queue flushed — fresh inference starting", policy_name)

            # --- Read observation ---
            obs_raw = robot.get_observation()

            with state.lock:
                state.obs = obs_raw

            # --- Build dataset-format observation frame ---
            obs_frame = build_dataset_frame(lerobot_features, obs_raw, prefix=OBS_STR)

            # --- Run ACT inference ---
            # task="" is accepted by predict_action but completely ignored by ACT
            action_values = predict_action(
                observation=obs_frame,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=(device.type == "cuda"),
                task="",
                robot_type=robot.robot_type,
            )

            # --- Send action to robot ---
            robot_action = {name: float(action_values[i]) for i, name in enumerate(action_names)}
            robot.send_action(robot_action)

            elapsed = time.perf_counter() - loop_start
            sleep_time = LOOP_DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                logger.debug("Fast loop overran by %.1f ms", -sleep_time * 1000)

    except Exception:
        logger.exception("Fast loop crashed")
    finally:
        robot.disconnect()
        logger.info("Robot disconnected, fast loop exiting")
