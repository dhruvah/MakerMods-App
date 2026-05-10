"""Diagnostic script — connects to robot, runs 3 inference steps, prints everything.

Usage:
    conda activate lerobot
    python -m orchestrator.diagnose --port /dev/tty.usbmodem5B140310701 --cameras front_cam:0 side_cam:1

Prints:
  - Raw observation from robot (joint values, camera shapes)
  - Observation frame after build_dataset_frame (keys + shapes)
  - First 3 actions predicted by ACT (joint targets)
  - Whether the arm actually moves (compare obs before/after)
"""

import argparse
import time
import torch
import numpy as np

MODEL_ID = "qualia-robotics/act-makermods-pick-and-drop-bread-4c417ee8"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--cameras", nargs="+", default=["front_cam:0", "side_cam:1"])
    parser.add_argument("--device", default="mps")
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()

    device = torch.device(args.device)

    # ── Connect robot ────────────────────────────────────────────────────────
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig

    cameras = {}
    for entry in args.cameras:
        name, _, idx = entry.partition(":")
        cameras[name.strip()] = OpenCVCameraConfig(
            index_or_path=int(idx.strip()), width=640, height=480, fps=30
        )

    robot = make_robot_from_config(SO101FollowerConfig(
        port=args.port, cameras=cameras, id="auto_follower_left"
    ))
    robot.connect()
    print("\n✓ Robot connected\n")

    # ── Load policy ──────────────────────────────────────────────────────────
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
    from lerobot.utils.constants import OBS_STR
    from lerobot.utils.control_utils import predict_action

    print(f"Loading policy from {MODEL_ID} ...")
    policy = ACTPolicy.from_pretrained(MODEL_ID).to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=MODEL_ID,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    policy.reset()
    lerobot_features = hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=False)
    action_names = list(robot.action_features.keys())
    print(f"✓ Policy loaded ({sum(p.numel() for p in policy.parameters()):,} params)\n")

    # ── Step 1: Print raw observation ────────────────────────────────────────
    obs_raw = robot.get_observation()
    print("=" * 60)
    print("RAW OBSERVATION from robot.get_observation():")
    for k, v in obs_raw.items():
        if isinstance(v, np.ndarray):
            print(f"  {k:40s} shape={v.shape} dtype={v.dtype}")
        else:
            print(f"  {k:40s} value={v:.4f}")

    # ── Step 2: Print formatted frame ────────────────────────────────────────
    obs_frame = build_dataset_frame(lerobot_features, obs_raw, prefix=OBS_STR)
    print("\nFORMATTED FRAME after build_dataset_frame():")
    for k, v in obs_frame.items():
        if isinstance(v, np.ndarray):
            if v.ndim == 1:
                print(f"  {k:40s} {v.round(3)}")
            else:
                print(f"  {k:40s} shape={v.shape} dtype={v.dtype}  min={v.min()} max={v.max()}")

    # ── Step 3: Run inference steps ──────────────────────────────────────────
    print(f"\nRUNNING {args.steps} INFERENCE STEPS:")
    print(f"  Action names: {action_names}")
    print()

    obs_before = {k: v for k, v in obs_raw.items() if isinstance(v, float)}

    for step in range(args.steps):
        obs_raw = robot.get_observation()
        obs_frame = build_dataset_frame(lerobot_features, obs_raw, prefix=OBS_STR)

        action_values = predict_action(
            observation=obs_frame,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=False,
            task="",
            robot_type=robot.robot_type,
        )

        action_dict = {name: float(action_values[i]) for i, name in enumerate(action_names)}
        gripper_obs  = float(obs_raw.get("gripper.pos", -1))
        gripper_cmd  = action_dict.get("gripper.pos", -1)

        print(f"  Step {step+1:02d}:  gripper_obs={gripper_obs:7.3f}  gripper_cmd={gripper_cmd:7.3f}  "
              f"shoulder_pan_cmd={action_dict.get('shoulder_pan.pos', 0):7.3f}")

        robot.send_action(action_dict)
        time.sleep(0.05)  # 20 Hz

    # ── Step 4: Compare before/after ─────────────────────────────────────────
    obs_after = robot.get_observation()
    print("\nJOINT MOVEMENT (before → after inference):")
    joints = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
              "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
    moved = False
    for j in joints:
        before = obs_before.get(j, 0)
        after  = float(obs_after.get(j, 0))
        delta  = abs(after - before)
        flag   = " ← MOVED" if delta > 0.5 else ""
        print(f"  {j:25s} {before:7.3f} → {after:7.3f}  Δ={delta:.3f}{flag}")
        if delta > 0.5:
            moved = True

    print()
    if moved:
        print("✓ Arm moved — inference pipeline is working")
    else:
        print("✗ Arm did NOT move — check starting position / policy mismatch")

    robot.disconnect()


if __name__ == "__main__":
    main()
