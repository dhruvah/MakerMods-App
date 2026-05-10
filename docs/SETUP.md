# Orchestrator Setup Guide

How to go from "model trained" to "robot doing things."

---

## Before you start: four things to find

Write these down before doing anything else.

1. **ACT model IDs** — one per phase, each looks like `your_hf_user/act_grasp` (shown in the Training step of the MakerMods UI after training completes, or the local path to your training output directory)
2. **Robot serial port** — looks like `/dev/tty.usbserial-FT8ISL9O` (Mac) or `/dev/ttyUSB0` (Linux)
3. **Camera index numbers** — numbers like `0`, `1`, `2` that OpenCV uses to identify each physical camera
4. **Anthropic API key** — looks like `sk-ant-api03-...`

---

## Step 1 — Find your ACT model IDs

Each phase of your task requires a separately trained ACT checkpoint. ACT takes no language input — behaviour is fully determined by which checkpoint is active. The orchestrator switches checkpoints as the task progresses.

You need one model ID per phase. These are shown in the MakerMods UI Training step once each training run completes. Copy-paste them, don't retype. A local training output path (e.g. `outputs/train/act_grasp/`) also works if you trained locally.

### Critical: camera names must match exactly

Each ACT checkpoint was trained on recordings that used specific camera names (e.g. `front_cam`, `hand_cam`). The ACTPolicy config encodes which camera names it expects — you must use the exact same names at inference time or the model will receive the wrong observations.

Open each model's config on HuggingFace (replace with your actual model ID):

```
https://huggingface.co/your_hf_user/YOUR-MODEL-NAME/blob/main/config.json
```

Look for `"input_features"`. You'll see entries like:

```json
"observation.images.front_cam": { "type": "VISUAL", ... },
"observation.images.hand_cam":  { "type": "VISUAL", ... }
```

The part after `"observation.images."` is your camera name. All phases should have been trained with the same camera names — verify this before running.

---

## Step 2 — Install missing dependencies

The lerobot conda environment doesn't include Strands or Pillow:

```bash
conda activate lerobot
pip install strands-agents pillow
```

---

## Step 3 — Find your robot's serial port

**Method A — lerobot's built-in tool:**

```bash
conda activate lerobot
python -m lerobot.scripts.lerobot_find_port
```

Follow the prompts (it asks you to unplug and replug the cable). It prints the port name at the end.

**Method B — manual:**

```bash
# Before plugging in
ls /dev/tty.usbserial-* /dev/ttyUSB* 2>/dev/null

# Plug in the robot, wait 2 seconds, then run again
ls /dev/tty.usbserial-* /dev/ttyUSB* 2>/dev/null
```

The new entry that appeared is your port. For bimanual setups, repeat with each cable.

---

## Step 4 — Find your camera index numbers

```bash
conda activate lerobot
python3 -c "
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    ok, frame = cap.read()
    cap.release()
    print(f'Camera {i}: {\"WORKS\" if ok else \"nothing here\"}' + (f' — shape {frame.shape}' if ok else ''))
"
```

To figure out which index is which physical camera, cover one camera with your hand and rerun — the one that goes dark is that index.

---

## Step 5 — Set your Anthropic API key

```bash
export ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
```

Add this to `~/.zshrc` (Mac) or `~/.bashrc` (Linux) to avoid setting it every session.

Verify: `echo $ANTHROPIC_API_KEY`

---

## Step 6 — Sanity checks before the real run

Run these in isolation so you're not debugging two things at once.

### Check A — Robot connects

```bash
conda activate lerobot
python3 -c "
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots import make_robot_from_config

cfg = SO101FollowerConfig(port='/dev/tty.usbserial-YOURPORT')
robot = make_robot_from_config(cfg)
robot.connect()
obs = robot.get_observation()
print('Connected! Keys:', list(obs.keys()))
print('Values:', obs)
robot.disconnect()
"
```

Expected: joint values printed. If you see `SerialException`, the port is wrong or the robot isn't powered.

### Check B — ACT policy loads

```bash
conda activate lerobot
python3 -c "
from lerobot.policies.act.modeling_act import ACTPolicy
print('Loading... (30-60s first time)')
policy = ACTPolicy.from_pretrained('your_hf_user/YOUR-MODEL-NAME')
print('Action shape:', policy.config.action_feature.shape)
print('Input features:', list(policy.config.input_features.keys()))
"
```

Expected: `Action shape: (6,)` and input features listing your camera names. Repeat for each phase's model ID — all must load successfully before attempting the full run. First run downloads weights; subsequent runs are instant from local cache.

---

## Step 7 — Run the orchestrator

Run from the `makermod_app/` repo root (not from inside `orchestrator/`).

```bash
conda activate lerobot
python -m orchestrator.run \
  --port    /dev/tty.usbserial-YOURPORT \
  --task    "place the bread in the toaster" \
  --phases  "grasp:your_hf_user/act_grasp" \
            "insert:your_hf_user/act_insert" \
            "press:your_hf_user/act_press" \
  --cameras front_cam:0 hand_cam:1
```

`--task` is passed to Claude as high-level context only. ACT never sees it.

**Argument reference:**

| Argument | What it is | Example |
|---|---|---|
| `--port` | Serial port of follower arm | `/dev/tty.usbserial-FT8ISL9O` |
| `--task` | High-level task description (for Claude only) | `"place the bread in the toaster"` |
| `--phases` | Ordered `name:hf_model_id` pairs | `"grasp:your_hf_user/act_grasp"` |
| `--cameras` | `name:opencv_index` pairs — name must match model config | `front_cam:0 hand_cam:1` |
| `--device` | Torch device (default: `mps` for Apple Silicon) | `mps` / `cuda` / `cpu` |
| `--claude-model` | Claude model for orchestration (default: `claude-sonnet-4-6`) | `claude-opus-4-7` |

There is no `--model` argument. Each phase carries its own model ID in `--phases`.

---

## Step 8 — What you'll see

The fast loop loads all policies at startup before the first observation is published:

```
INFO fast_loop — Robot connected on /dev/tty.usbserial-XXX with cameras: ['front_cam', 'hand_cam']
INFO fast_loop — Loading ACT policy for phase 'grasp' from your_hf_user/act_grasp ...
INFO fast_loop —   Loaded 'grasp' (83,208,576 parameters)
INFO fast_loop — Loading ACT policy for phase 'insert' from your_hf_user/act_insert ...
INFO fast_loop —   Loaded 'insert' (83,208,576 parameters)
INFO fast_loop — Loading ACT policy for phase 'press' from your_hf_user/act_press ...
INFO fast_loop —   Loaded 'press' (83,208,576 parameters)
```
*(wait 30–90 seconds for all policies to download/load)*
```
INFO fast_loop — Fast loop starting at 20 Hz with 3 loaded policies
INFO run — Robot online. Starting Strands orchestrator.
INFO run — Sending initial prompt to orchestrator agent...
```

The robot is idle — `active_policy_name` is empty, so the fast loop does nothing until Claude calls `advance_phase`.

```
INFO tools — Phase → grasp (ACT policy switched)
```

**The robot arm will start moving here.** Stand clear.

As the agent works through phases you'll see tool calls logged:
```
[Tool: get_robot_state]      → {shoulder_pan: 0.12, ..., gripper: 87.3}
[Tool: capture_camera_frame] → (Claude looks at the scene)
[Tool: check_gripper_closed] → {gripper_value: 11.2, is_closed: True}
[Tool: advance_phase]        → "insert" (ACT policy switched)
```

When complete:
```
[Tool: complete_task] → Task complete. Robot shutting down.
INFO run — Shutdown complete
```

Press **Ctrl+C** at any time for an immediate clean stop.

---

## Common errors and fixes

**`No observation available yet — fast loop may not have started`**
All ACT policies are still loading. Wait up to 90 seconds (longer if loading multiple large checkpoints on a slow connection).

**`Camera 'front_cam' not found. Available: []`**
Either the camera index is wrong (Step 4), or the camera name in `--cameras` doesn't match the name in the ACTPolicy config's `input_features` (Step 1).

**`serial.serialutil.SerialException: [Errno 16] Resource busy`**
The port is already open — the MakerMods web app or a leftover process is using it. Kill it:
```bash
lsof /dev/tty.usbserial-YOURPORT
kill <PID>
```

**Robot executes previous phase's motion after policy switch**
The action chunk queue from the outgoing policy was not flushed. This should not happen when using `advance_phase` correctly. If it does, check that `advance_phase` is the only mechanism used to switch phases — writing to `SharedState.active_policy_name` directly skips the `reset_requested` flag and leaves stale actions in the queue.

**`Unknown phase 'grasp_bread'. Valid phases: ['grasp', 'insert', 'press']`**
The phase name passed to `advance_phase` must exactly match the name given in `--phases`. Use the short left-hand side of the `name:hf_model_id` pair.

**Phase never advances — Claude keeps calling tools without deciding**
Check your API key. If the key is fine, the camera angle may not give Claude enough visual information to confirm success — reposition the camera so the action area is clearly visible.

---

## Architecture recap

```
┌─────────────────────────────────────────────────────┐
│  Fast loop (background thread, 20 Hz)               │
│                                                     │
│  robot.get_observation()                            │
│    → build_dataset_frame()                          │
│    → policies[active_policy_name].select_action()   │
│    → robot.send_action()                            │
│                                                     │
│  On reset_requested: policy.reset() flushes queue   │
│                                                     │
│  Reads:  SharedState.active_policy_name (every 50ms)│
│  Writes: SharedState.obs                (every 50ms)│
└────────────────────┬────────────────────────────────┘
                     │ SharedState (one Python dict + lock)
┌────────────────────┴────────────────────────────────┐
│  Strands agent (main thread, ~1 Hz)                 │
│                                                     │
│  Claude + 9 tools:                                  │
│    get_robot_state       check_gripper_closed        │
│    capture_camera_frame  check_joint_angle           │
│    pause_robot           get_phase_status            │
│    resume_robot          advance_phase               │
│    complete_task                                     │
│                                                     │
│  Reads:  SharedState.obs                (via tools) │
│  Writes: SharedState.active_policy_name (via tools) │
└─────────────────────────────────────────────────────┘
```

The fast loop executes faithfully without ever judging whether it's going well.
The slow loop watches and decides, but never touches the motors directly.
The active policy name is the only wire between them.
