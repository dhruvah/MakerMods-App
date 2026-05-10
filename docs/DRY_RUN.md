# Dry-Run Test Mechanism

**File:** `orchestrator/dry_run.py` + `orchestrator/mock_robot.py`  
**Status:** Validated — full 3-phase toaster task completed successfully

---

## 1. Purpose

The dry run exists to test the orchestrator's logic — Claude's decision-making, tool call
sequencing, phase transitions, timeout recovery, and shutdown behaviour — without requiring
physical hardware or trained ACT model checkpoints.

In production, `run.py` starts a real LeRobot fast loop (`fast_loop.py`) that controls
servo motors via a serial port and runs ACT inference on GPU. Both of those dependencies
are expensive to set up and impossible to run in CI or on a laptop. The dry run replaces
`fast_loop.py` with `mock_robot.py`: a Python thread that simulates joint movement based on
the active policy name (phase name). Everything else — `SharedState`,
all 9 Strands tools, the Strands `Agent`, `SYSTEM_PROMPT`, `build_initial_prompt` — runs
unchanged.

**What the dry run validates (in short):**

- Claude calls tools in the correct order and advances phases via `advance_phase`
- Policy switching via `advance_phase` fires at the right times based on sensor readings, not clocks
- The `reset_requested` flag flushes the action queue on phase transitions
- Phase names are validated against the configured list
- `complete_task` fires only after all sensor thresholds are confirmed
- Thread safety between the 20 Hz mock loop and the agent thread

**What it does not validate:** whether any of this actually works on a physical robot.
See Section 5 for the full honest list.

---

## 2. Architecture

### Production (run.py)

```
┌─────────────────────────────────────────────────────┐
│  fast_loop.run()  (background thread, 20 Hz)        │
│                                                     │
│  robot.get_observation()                            │
│    → policies[active_policy_name].select_action()   │
│    → robot.send_action()                            │
│                                                     │
│  Reads:  SharedState.active_policy_name             │
│  Writes: SharedState.obs                            │
└────────────────────┬────────────────────────────────┘
                     │ SharedState (lock-protected)
┌────────────────────┴────────────────────────────────┐
│  Strands Agent  (main thread, ~1 Hz)                │
│                                                     │
│  Claude + 9 tools                                   │
│                                                     │
│  Reads:  SharedState.obs                (via tools) │
│  Writes: SharedState.active_policy_name (via tools) │
└─────────────────────────────────────────────────────┘
```

### Dry run (dry_run.py) — only one component changes

```
┌─────────────────────────────────────────────────────┐
│  mock_robot.run()  (background thread, 20 Hz)       │  ← REPLACED
│                                                     │
│  phase name match against active_policy_name        │
│    → step joints toward per-phase targets           │
│    → generate labelled PIL camera frames            │
│                                                     │
│  Reads:  SharedState.active_policy_name             │
│  Writes: SharedState.obs                            │
└────────────────────┬────────────────────────────────┘
                     │ SharedState (unchanged)
┌────────────────────┴────────────────────────────────┐
│  Strands Agent  (main thread — UNCHANGED)           │
│                                                     │
│  Claude + 9 tools  (UNCHANGED)                      │
│                                                     │
│  Reads:  SharedState.obs                (via tools) │
│  Writes: SharedState.active_policy_name (via tools) │
└─────────────────────────────────────────────────────┘
```

The swap is one line in `dry_run.py`:

```python
# Production (run.py):
from .fast_loop import run as run_fast_loop

# Dry run (dry_run.py):
from .mock_robot import run as run_mock
```

The mock's `run()` signature accepts the same `state` and `cameras` kwargs as
`fast_loop.run()` (plus `**_kwargs` to swallow `robot_port`, `phase_model_map`, etc.),
so `dry_run.py` can call it identically.

---

## 3. How the Mock Robot Works

### 3.1 Startup and thread model

`dry_run.main()` creates a `SharedState`, starts `mock_robot.run()` in a daemon thread,
then blocks in `wait_for_mock()` until at least one observation is available:

```python
mock_thread = threading.Thread(
    target=run_mock,
    kwargs={"state": state, "cameras": cameras},
    daemon=True,
    name="mock-robot",
)
mock_thread.start()
wait_for_mock(state)          # polls state.snapshot_obs() for up to 5 s
logger.info("Mock robot online. Starting dry-run orchestrator.")
```

Once the first observation is published the Strands agent is created and the initial prompt
is sent.

### 3.2 20 Hz loop

The mock runs at `TARGET_HZ = 20` (50 ms per iteration). Each cycle it:

1. Reads `state.active_policy_name` and `state.reset_requested` under the lock
2. If `reset_requested` is set and the phase name has changed, updates `current_phase`,
   clears `state.reset_requested`, and logs the switch
3. Calls `_targets_for_phase()` to get joint targets for the current phase name
4. Steps every joint toward its target at a fixed speed (`SPEEDS`)
5. Publishes a new `obs` dict under the lock
6. Sleeps for the remainder of the 50 ms window

When `state.paused` is `True` or no policy is active, the loop still publishes observations
(so tools can read state) but does not advance joints:

```python
if state.paused or not policy_name:
    with state.lock:
        state.obs = _build_obs(joint_state, camera_names, policy_name or "idle")
    time.sleep(LOOP_DT)
    continue
```

This matches the production behaviour where `fast_loop` skips `send_action` when paused.

### 3.3 Phase-name-to-target mapping

`_targets_for_phase()` does a case-insensitive substring search of `active_policy_name`
against the `PHASE_TARGETS` dict. All matching keys contribute their joint targets (later
entries overwrite earlier ones for the same joint):

```python
PHASE_TARGETS = {
    "grasp":   {"gripper": 8.0},
    "insert":  {"elbow_flex": -0.6, "shoulder_lift": 0.6},
    "press":   {"wrist_flex": -0.9},
    "lever":   {"wrist_flex": -0.9},
    "release": {"gripper": 90.0},
}
```

For any joint **not** listed in the matched targets, the target falls back to `RESTING_STATE`
(the arm's neutral pose), so joints not involved in the current phase drift back to rest.

Full phase target table:

| Phase name contains | Joint(s) affected | Target value | Behaviour |
|---|---|---|---|
| `grasp` | `gripper` | 8.0 | Gripper closes |
| `release` | `gripper` | 90.0 | Gripper opens |
| `insert` | `elbow_flex`, `shoulder_lift` | −0.6, +0.6 | Arm extends forward and up |
| `press` / `lever` | `wrist_flex` | −0.9 | Wrist flexes down |
| (no match) | all | resting values | Joints return to neutral |

### 3.4 Joint movement speed and timing

Joints move at fixed speeds (units per second):

```python
SPEEDS = {
    "shoulder_pan":  0.4,
    "shoulder_lift": 0.3,
    "elbow_flex":    0.3,
    "wrist_flex":    0.4,
    "wrist_roll":    0.4,
    "gripper":      25.0,
}
```

The `_step_toward()` function advances each joint by at most `speed * dt` per tick and
clamps at the target:

```python
def _step_toward(current: float, target: float, speed: float, dt: float) -> float:
    delta = target - current
    max_step = speed * dt
    if abs(delta) <= max_step:
        return target
    return current + max_step * (1 if delta > 0 else -1)
```

**Approximate phase durations at 20 Hz (dt = 0.05 s):**

| Action | Joint | Distance | Speed | Time |
|---|---|---|---|---|
| Gripper close (90 → 8) | `gripper` | 82 units | 25.0/s | ~3.3 s |
| Gripper open (8 → 90) | `gripper` | 82 units | 25.0/s | ~3.3 s |
| Elbow extend (0.1 → −0.6) | `elbow_flex` | 0.7 rad | 0.3/s | ~2.3 s |
| Shoulder up (0.3 → 0.6) | `shoulder_lift` | 0.3 rad | 0.3/s | ~1.0 s |
| Wrist flex (0.0 → −0.9) | `wrist_flex` | 0.9 rad | 0.4/s | ~2.3 s |

The gripper close/open is the slowest operation on a per-unit basis but its large range
(0–100 normalised) means it takes roughly 3–4 seconds to complete — which is why the
agent monitors for `gripper < 15` rather than declaring grasp complete immediately after
`advance_phase`.

### 3.5 Camera frame generation

Each 20 Hz tick generates a 320×240 RGB numpy array per configured camera. If Pillow is
installed the frame includes:

- A solid background colour keyed to the current phase name (green for grasp,
  blue for insert, orange for press)
- White text at the top: `PHASE: <phase_name>` and `ACT policy active`
- A yellow horizontal bar at the bottom whose width encodes the current gripper value
  (full width = gripper fully open at 100)
- Text above the bar: `gripper: <value>`

```python
PHASE_COLOURS = {
    "grasp":   (60, 120, 60),    # green-ish
    "insert":  (60, 60, 140),    # blue-ish
    "press":   (140, 80, 60),    # orange-ish
    "lever":   (140, 80, 60),
    "release": (100, 100, 60),   # yellow-ish
}
```

If Pillow is not installed the frame is a solid colour array (no text or bar).

The `capture_camera_frame` tool in `tools.py` encodes these frames as base64 JPEG at up
to 512px wide before sending them to Claude. Claude's vision capability can read the
phase label and gripper bar, which is intentionally helpful — the mock provides
enough visual signal to satisfy the "always use visual confirmation" rule from `CLAUDE.md`
without any real camera hardware.

---

## 4. What Gets Validated

### 4.1 Agent decision-making and tool call sequencing

The Strands agent must call tools in the correct causal order:
`advance_phase` → monitor loop (`get_phase_status`, `get_robot_state`,
`capture_camera_frame`, `check_gripper_closed` / `check_joint_angle`) → next
`advance_phase` → ... → `complete_task`.

The dry run confirms that Claude does not:
- Call `complete_task` before all phases are done
- Skip sensor confirmation and advance phases on elapsed time alone
- Call tools in a tight polling loop (violating the 1 Hz cadence recommendation)

### 4.2 Policy switching via advance_phase

`advance_phase(phase_name)` is the only way to switch policies. The dry run validates
that Claude uses it correctly and never attempts to modify `active_policy_name` through
any other mechanism.

Each `advance_phase` call:
1. Sets `state.reset_requested = True` so the mock (and in production, the fast loop)
   knows to flush the action queue on the next tick
2. Writes the new `active_policy_name`
3. Handles its own pause/resume around the switch

The dry run confirms the mock observes `reset_requested`, clears it, and switches to
the correct phase targets without stale joint motion carrying over.

### 4.3 `reset_requested` flag flushes action queue

When `advance_phase` fires, it sets `state.reset_requested = True` before writing the
new policy name. The mock loop checks this flag at the top of every tick:

```python
if should_reset and policy_name != current_phase:
    current_phase = policy_name
    with state.lock:
        state.reset_requested = False
    logger.debug("Mock: switched to phase '%s'", policy_name)
```

In production the fast loop calls `policy.reset()` at this point to clear the ACT
action-chunk queue. The dry run validates that `reset_requested` is set on every phase
transition and cleared exactly once per switch.

### 4.4 Phase names validated against configured list

`advance_phase(phase_name)` only accepts names present in the list passed to `make_tools()`
at startup. Passing an unrecognised name returns an error without modifying `SharedState`.
The dry run confirms Claude reads `available_phases` from `get_phase_status()` and only
ever calls `advance_phase` with names from that list.

### 4.5 Sensor thresholds

The mock produces deterministic joint trajectories, so threshold checks exercise the
real threshold logic:

| Phase | Tool call | Threshold | Mock delivers |
|---|---|---|---|
| `grasp_bread` | `check_gripper_closed(threshold=15)` | gripper < 15 | gripper → 8.0 in ~3.3 s |
| `insert_bread` | visual confirm via `capture_camera_frame` | arm extended | elbow_flex → −0.6 in ~2.3 s |
| `press_lever` | `check_joint_angle("wrist_flex", "<", -0.8)` | wrist_flex < −0.8 | wrist_flex → −0.9 in ~2.3 s |

The tool implementations in `tools.py` read from `state.snapshot_obs()` — the same code
path the production tools use. The only difference is the source of the numpy arrays.

### 4.6 `complete_task` discipline

`complete_task` sets `state.stop = True` and `state.paused = True`. It should only be
called after every phase's success condition is sensor-confirmed. The dry run validates
that Claude does not call it:

- Immediately after the last `advance_phase` (before sensor confirmation)
- After only some phases are complete
- As an error handler (giving up rather than completing)

In the confirmed run Claude called `complete_task` only after `wrist_flex < -0.9` was
confirmed, with a summary that named all three phases explicitly.

### 4.7 Thread safety

`SharedState` protects all fields with a single `threading.Lock`. Tools acquire the lock
via `state.snapshot_obs()` (shallow copy under lock), `state.set_phase()`, and related
setters. The mock loop acquires the same lock when reading `active_policy_name` and
writing `obs`.

The dry run exercises this concurrency: the mock loop runs at 20 Hz while the agent thread
makes tool calls at irregular intervals (1–60 s apart depending on rate limiting). No
deadlocks or race conditions were observed in the confirmed run.

`advance_phase` holds the lock briefly in two separate critical sections with a 300 ms
sleep between them (intentionally outside the lock):

```python
with state.lock:
    state.paused = True         # pause under lock

time.sleep(0.3)                 # sleep outside lock — mock loop can still publish obs

state.set_phase(phase_name)     # each method acquires its own lock internally

with state.lock:
    state.paused = False        # resume under lock
```

This pattern ensures the mock loop is never blocked for 300 ms by a held lock.

### 4.8 Clean shutdown

After `complete_task` sets `state.stop = True`, the mock loop exits its `while not state.stop`
loop and logs `"Mock robot stopped"`. `dry_run.main()` then calls `mock_thread.join(timeout=3.0)`.
The confirmed run exited cleanly with no hanging threads.

---

## 5. What the Dry Run Does NOT Validate

Be honest with yourself before marking a hardware test as "done" because the dry run passed.

### 5.1 ACT's neural network response to observations

The mock robot is purely phase-name-based. It does not run ACT. The gripper closes
during a `grasp` phase because the string contains `"grasp"` — not because a trained
ACT checkpoint observed the scene and generated appropriate motor commands.

Whether the trained ACT checkpoints produce good actions for the actual camera views and
object positions they will encounter on hardware is completely untested. On hardware you
may find that a checkpoint that works in the dry run produces erratic motion because the
real scene differs from training distribution.

### 5.2 Real camera frame interpretation

The mock generates labelled colour blocks. Claude reads these and extracts the gripper bar
width and phase label. A real camera will show an actual scene — the bread, the
toaster, the arm, lighting variation, occlusion. Claude's visual reasoning on real frames
may produce different decisions.

### 5.3 Real physics and timing

The mock closes the gripper deterministically in ~3.3 s with a linear ramp. A real servo
drive may close faster or slower depending on load, current limits, and PID tuning. The
mock never drops an object, never oscillates, and never fails to reach its target. Real
hardware can do all of these things.

### 5.4 Real failure modes

The mock has no failure states. It cannot simulate:

- Gripper grasping the wrong object or closing on empty air
- Arm hitting a mechanical stop and stalling
- Object falling out of the gripper mid-transport
- ACT producing jerky or oscillating actions when the scene is out of distribution
- Serial communication errors or packet loss
- Camera frame drops or exposure problems

If the orchestrator needs to handle any of these robustly, those failure paths must be
tested on hardware.

### 5.5 ACT action-chunk queue behaviour on real hardware

The dry run validates that `reset_requested` is set and cleared correctly, but the mock
does not actually maintain an action-chunk queue. On hardware, if the fast loop does not
correctly call `policy.reset()` when `reset_requested` is set, stale motor commands from
the previous phase will keep executing for up to `n_action_steps` cycles (~5 s), which
can cause dangerous unexpected motion. This must be verified on hardware with the real
fast loop.

---

## 6. Actual Test Output

The following is the timestamped run output from the confirmed successful dry run,
annotated line by line.

```
2025-05-09 21:22:32  INFO  orchestrator.dry_run — Mock robot online. Starting dry-run orchestrator.
```
`wait_for_mock()` returned — the 20 Hz loop published at least one `obs` dict within 5 s.
`SharedState.obs` is non-empty and tools are safe to call.

```
2025-05-09 21:22:32  INFO  orchestrator.dry_run — Sending initial prompt...
```
`agent(prompt)` called with `build_initial_prompt("place the bread in the toaster", phases)`.
The Strands HTTP call is in flight.

```
2025-05-09 21:22:40  INFO  orchestrator.tools — Phase → grasp_bread | active_policy: 'mock/act_grasp_bread'
```
**Phase 1 starts.** Elapsed from prompt send: ~8 s (first API round trip).
Claude called `advance_phase("grasp_bread")`.
`active_policy_name` is set to `"mock/act_grasp_bread"`. `"grasp"` is a substring of the policy
name, so `PHASE_TARGETS["grasp"]` applies — mock begins closing gripper from 90 → 8.

*Note: next log line is ~12 minutes later due to free-tier rate limiting (34–59 s per tool call)*

```
2025-05-09 21:22:52  INFO  orchestrator.tools — Phase → insert_bread | active_policy: 'mock/act_insert_bread'
```
**Phase 2 starts.** Elapsed in `grasp_bread`: ~12 s from phase entry.
Before this line, Claude called `check_gripper_closed()` and received `gripper_value: 8.0, is_closed: True`
(gripper reached target). It also called `capture_camera_frame()` and saw a green-background
frame with the gripper bar near zero width — visual confirmation of grasp.
`active_policy_name` set to `"mock/act_insert_bread"`. `"insert"` matches — mock begins
moving `elbow_flex` → −0.6 and `shoulder_lift` → 0.6.

```
2025-05-09 21:26:44  INFO  orchestrator.tools — Phase → press_lever | active_policy: 'mock/act_press_lever'
```
**Phase 3 starts.** Elapsed from `insert_bread` entry: ~3 min 52 s (rate limiting dominated).
Claude confirmed the insert via `capture_camera_frame()` (blue background, arm extended).
`active_policy_name` set to `"mock/act_press_lever"`. `"press"` matches — mock begins
moving `wrist_flex` → −0.9.

```
INFO  orchestrator.tools — Task complete: Successfully completed all 3 phases: (1) Grasped the bread loaf
(gripper closed to 8.0), (2) Inserted the bread into the toaster slot (gripper opened to release),
(3) Pressed the toaster lever all the way down (wrist_flex reached -0.9, below the -0.8 threshold).
The bread is now in the toaster with the lever pressed down.
```
**`complete_task` fired.** Claude's summary names all three phases and references the exact
sensor readings — confirming it tracked the observations correctly. `state.stop = True` and
`state.paused = True` are set.
The `wrist_flex < -0.8` threshold check passed before this call (`wrist_flex` reached −0.9).

```
2025-05-09 21:26:44  INFO  orchestrator.mock_robot — Mock robot stopped
```
Mock loop exited `while not state.stop`. No exception path.

```
2025-05-09 21:26:44  INFO  orchestrator.dry_run — Dry run complete
```
`mock_thread.join(timeout=3.0)` returned. Clean exit.

---

## 7. How to Run It

### Prerequisites

```bash
conda activate lerobot          # or your venv — must have strands, anthropic, pillow, numpy
export ANTHROPIC_API_KEY=sk-ant-...
```

Pillow (`pip install pillow`) is technically optional — the mock falls back to solid colour
arrays — but without it Claude sees no phase label or gripper bar in camera frames, which may
reduce its ability to confirm phases visually.

### Default invocation (from the `makermod_app/` root)

```bash
python -m orchestrator.dry_run
```

This runs the default 3-phase toaster task using phases:
`grasp:mock/act_grasp_bread`, `insert:mock/act_insert_bread`, `press:mock/act_press_lever`.

### Custom task

```bash
python -m orchestrator.dry_run \
  --task "place the bread in the toaster" \
  --phases "grasp_bread:mock/act_grasp_bread" \
           "insert_bread:mock/act_insert_bread" \
           "press_lever:mock/act_press_lever"
```

Each `--phases` entry is `PHASE_NAME:MODEL_ID`. The phase name is what Claude passes to
`advance_phase()`. The model ID is passed to the fast loop in production; in dry-run mode it
is ignored except that the mock robot uses it as the `active_policy_name` string and matches
substrings of it against `PHASE_TARGETS`.

### Custom model

```bash
python -m orchestrator.dry_run --claude-model claude-opus-4-5
```

Default is `claude-sonnet-4-6`.

### What to watch for

1. **Phase transitions logged at `INFO`** — each `advance_phase` call prints:
   ```
   Phase → <name> | active_policy: '<policy_name>'
   ```
   Confirm the phase name is from the configured list.

2. **`complete_task` fires last** — it should only appear after all three phase transition
   lines.

3. **No `ERROR` lines** — any `ERROR` in `orchestrator.tools` or `orchestrator.mock_robot`
   indicates a broken observation or missing state key.

4. **`Dry run complete` is the last line** — confirms clean thread shutdown.

5. **Rate limit delays** — on the free tier, individual tool call round trips may take
   34–59 s. The full run may take 5–15 minutes despite the mock completing actions in
   seconds. This is expected. See Section 8.

---

## 8. Known Issues

### 8.1 `count_tokens` 400 error

**Symptom:** Strands throws a 400 Bad Request during token counting before the first agent
call.

**Root cause:** The default `AnthropicModel` configuration calls the Anthropic
`messages.count_tokens` API to estimate context size before sending each request. On some
API tiers or with certain `strands` versions, the token-counting endpoint rejects the
request (e.g. does not support vision content in the counting payload, or returns a
quota error on the free tier).

**Fix applied:**

```python
model = AnthropicModel(
    model_id=args.claude_model,
    max_tokens=4096,
    use_native_token_count=False,   # ← this line
)
```

Setting `use_native_token_count=False` disables the separate `count_tokens` API call.
Strands falls back to a local token estimator. This approximately halves the number of
API calls made per agent turn (no separate counting request before each inference request),
which also reduces rate-limit pressure.

**Why it happens:** The Anthropic free tier has separate per-minute rate limits for the
`messages` endpoint and the `count_tokens` endpoint. Hitting the `count_tokens` limit
returns a 400 rather than a 429, which is confusing. `use_native_token_count=False`
avoids the endpoint entirely.

### 8.2 Rate limiting on the free tier

**Symptom:** Tool calls take 34–59 seconds instead of ~1 second. The full 3-phase run
takes 5–15 minutes.

**Root cause:** Free-tier Anthropic API quota is exhausted quickly by multi-turn agent
loops that include base64 camera frames (each frame is ~15–30 KB of base64 = hundreds of
tokens). Each agent turn that calls `capture_camera_frame` incurs significant token cost.

**Observed behaviour:** After hitting the rate limit, Strands retries automatically with
exponential backoff. The agent log shows gaps of 34–59 s between tool calls. During these
gaps the phase `elapsed_s` advances even though no robot progress is occurring, which can
trigger the 15-second timeout recovery heuristic.

**Mitigations:**

- Use `use_native_token_count=False` (already applied) to halve API call count
- Reduce camera frame size in `capture_camera_frame` — the current resize cap is 512 px
  wide; reducing to 256 px halves image token cost
- Use a paid API tier with higher rate limits for hardware testing
- Reduce `capture_camera_frame` call frequency in the agent's monitoring loop

**Note for hardware runs:** On a paid tier with `claude-sonnet-4-6`, round trips typically
take 1–3 seconds. The 34–59 s delays are a free-tier artifact and should not affect
production operation.

---

## 9. Extending the Mock

### 9.1 Adding new phase name substrings

Add a new entry to `PHASE_TARGETS` in `mock_robot.py`:

```python
PHASE_TARGETS = {
    # existing entries ...
    "rotate":  {"wrist_roll": 0.8},     # new: wrist roll for rotating an object
    "lower":   {"shoulder_lift": -0.2}, # new: lower arm
}
```

And optionally a background colour for visual feedback:

```python
PHASE_COLOURS = {
    # existing entries ...
    "rotate": (120, 60, 120),   # purple-ish
    "lower":  (60, 100, 100),   # teal-ish
}
```

The new key will be matched case-insensitively as a substring of `active_policy_name`
on every tick. Multiple keys can match simultaneously — their target dicts are merged.
For example, a policy named `"mock/act_grasp_rotate"` would match both `"grasp"` and
`"rotate"`, closing the gripper and rotating the wrist simultaneously.

### 9.2 Simulating failure modes

**Gripper never closes (test timeout recovery):**

Override the `gripper` speed to zero so the gripper never reaches the closed target:

```python
SPEEDS = {
    # ...
    "gripper": 0.0,   # gripper is stuck
}
```

With this change, `check_gripper_closed()` will never return `is_closed: True`. The agent
should detect `elapsed_s > 15`, re-trigger the policy via `advance_phase(current_phase_name)`,
and eventually call `complete_task` with a failure summary.

**Arm reaches target and oscillates (test phase stability):**

Change `_step_toward` to add noise at the target:

```python
def _step_toward(current, target, speed, dt):
    import random
    delta = target - current
    max_step = speed * dt
    if abs(delta) <= max_step:
        return target + random.uniform(-0.05, 0.05)   # ← add noise at target
    return current + max_step * (1 if delta > 0 else -1)
```

This causes the joint to jitter around the target value. Threshold checks
(`check_joint_angle`, `check_gripper_closed`) will intermittently return False even after
the joint has "arrived". The agent must either retry or use a more lenient threshold.

**Phase that never completes (test complete timeout handling):**

Set the target to a value that will never satisfy the agent's threshold:

```python
PHASE_TARGETS = {
    # ...
    "press":   {"wrist_flex": -0.5},   # only reaches -0.5, threshold is -0.8
    "lever":   {"wrist_flex": -0.5},
}
```

The agent will wait for `wrist_flex < -0.8`, never see it, and should eventually give up
and either re-trigger the policy or call `complete_task` with a failure message. This
validates that the agent does not loop indefinitely.

### 9.3 Adding new camera behaviours

The `_make_camera_frame` function generates the PIL image. To add a second camera with
different content (e.g. a hand camera that shows a close-up gripper view):

```python
def _build_obs(joint_state, camera_names, phase_name):
    joints = ["shoulder_pan", "shoulder_lift", "elbow_flex",
              "wrist_flex", "wrist_roll", "gripper"]
    state_array = np.array([joint_state[j] for j in joints], dtype=np.float32)
    obs = {"observation.state": state_array}

    for cam in camera_names:
        if cam == "hand_cam":
            obs[f"observation.images.{cam}"] = _make_hand_cam_frame(joint_state)
        else:
            obs[f"observation.images.{cam}"] = _make_camera_frame(phase_name, joint_state)
    return obs


def _make_hand_cam_frame(joint_state, width=320, height=240):
    """Close-up gripper view: zoomed gripper bar only."""
    img = Image.new("RGB", (width, height), (30, 30, 30))
    draw = ImageDraw.Draw(img)
    g = joint_state["gripper"]
    # Large gripper indicator
    bar_h = int((g / 100.0) * (height - 40))
    draw.rectangle([width//2 - 20, height - 20 - bar_h, width//2 + 20, height - 20],
                   fill=(200, 200, 80))
    draw.text((10, 10), f"HAND CAM\ngripper: {g:.1f}", fill=(255, 255, 255))
    return np.array(img, dtype=np.uint8)
```

The new camera name must also be passed in `--cameras` when running the dry run:

```bash
python -m orchestrator.dry_run --cameras front_cam:0 hand_cam:1
```

Camera names are arbitrary strings; the mock uses them only as dict keys when building
`obs`. The tools use the same keys when looking up `observation.images.<name>`.

---

## File Reference

| File | Role in dry run |
|---|---|
| `orchestrator/dry_run.py` | Entry point. Starts mock thread, builds Strands agent, sends prompt. |
| `orchestrator/mock_robot.py` | 20 Hz simulation loop. Replaces `fast_loop.py`. |
| `orchestrator/tools.py` | All 9 agent tools. **Unchanged** from production. |
| `orchestrator/shared_state.py` | Thread-safe state container. **Unchanged** from production. |
| `orchestrator/prompts.py` | `SYSTEM_PROMPT`, `build_initial_prompt`, `parse_phases`. **Unchanged**. |
| `orchestrator/fast_loop.py` | Production 20 Hz loop with ACT inference. **Not imported** by dry run. |
| `orchestrator/run.py` | Production entry point. Uses `fast_loop` instead of `mock_robot`. |
