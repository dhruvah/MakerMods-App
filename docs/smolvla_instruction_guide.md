# SmolVLA Instruction Guide

How to write language instructions that produce correct robot behavior.
Read this before writing any instruction string — a bad instruction causes dangerous motor output.

---

## Why instruction quality matters

SmolVLA is a Vision-Language-Action model built on a vision-language backbone (PaliGemma).
At each inference step it does the following:

1. **Tokenize** the instruction string using the VLM's tokenizer.
2. **Embed** the tokens into the same embedding space as the image patches.
3. **Condition** the action-prediction head on the combined visual + language representation.
4. **Output** a chunk of motor commands (joint velocities and positions) for the SO-101 arm.

The critical fact is step 3: the action-prediction head was trained on demonstrations where
the language embedding came from instructions in a specific distribution. When the instruction
is out of distribution, the conditioned representation is meaningless to the action head, and
the output is effectively random noise in motor-command space.

**There is no error mode.** SmolVLA always produces an action. A garbage instruction produces
a garbage action, and the fast loop sends it to the robot at 20 Hz regardless.

---

## The training distribution

SmolVLA was fine-tuned on robot manipulation demonstrations. The instructions in these
demonstrations share these properties:

- **Length**: 3–8 words. The median is approximately 5 words.
- **Grammar**: Imperative mood. Always starts with a verb.
- **Vocabulary**: Physical, spatial, object-centric. Verbs like *pick up*, *place*, *insert*,
  *press*, *push*, *pull*, *slide*, *release*, *open*, *close*, *move*, *lower*, *raise*,
  *grasp*, *grab*, *drop*. Nouns are object names and body parts of the arm.
- **Modifiers**: Sparse. Directional adverbs are acceptable (*down*, *left*, *forward*).
  Manner adverbs are not (*carefully*, *gently*, *slowly*, *firmly*).
- **One action per instruction**: Every training instruction describes a single atomic action.
  There are no compound instructions joined by *and*, *then*, or commas.
- **No questions**: The training set contains zero interrogative sentences.
- **No meta-language**: Instructions do not describe the robot's state, the user's goals,
  or reasoning. They describe the physical action.

---

## Instruction taxonomy with examples

Use this table as a lookup when constructing phase instructions.

### Grasp instructions

The robot approaches and closes the gripper around an object.

| Good | Notes |
|---|---|
| `pick up the bread loaf` | Classic grasp. Object name is specific. |
| `grasp the white cylinder` | Color as disambiguator when multiple objects present. |
| `grab the red block` | Short form acceptable. |
| `pick up the object on the left` | Spatial position acceptable when object name is ambiguous. |
| `lift the cup` | Lift implies grasp + vertical motion — valid compound that appears in training data. |

Do not use: `pick up the bread loaf from the counter and hold it`, `grasp object`, `get bread`.

### Move / transport instructions

The robot moves a grasped object from one location to another.

| Good | Notes |
|---|---|
| `move the bread to the toaster` | Destination specified by object name. |
| `carry the cup to the right` | Directional transport. |
| `bring the block closer` | Relative motion. |
| `move arm to the left` | Body-part transport when object is implicit. |
| `raise the arm` | Vertical transport, arm as object. |
| `lower the arm slowly` | "Slowly" is marginal — prefer without. |

Do not use: `move the bread from the left side of the counter to the toaster slot on the right`,
`transport object to destination`.

### Insert instructions

The robot fits an object into a slot, hole, or cavity.

| Good | Notes |
|---|---|
| `insert bread into toaster slot` | Core training phrase for this task. |
| `insert the plug into the socket` | Generalises to other insert tasks. |
| `push bread into the toaster` | Push as alternative verb for insertion. |
| `slide the card into the reader` | Horizontal insert with specific verb. |

Do not use: `insert bread into toaster slot carefully and make sure it goes all the way in`,
`put bread in toaster`.

Note: `put` is weakly represented in the training data compared to `insert`, `place`, `push`.
Prefer `insert` or `place` for slot-targeting tasks.

### Press / toggle instructions

The robot pushes a lever, button, or switch.

| Good | Notes |
|---|---|
| `press the toaster lever down` | The exact phrase for the toaster lever task. |
| `press the button` | Generic button press. |
| `push the lever all the way down` | Emphasise full travel with "all the way". |
| `push the red button` | Color disambiguates multiple buttons. |
| `flip the switch` | Toggle action. |

Do not use: `depress the lever mechanism until it latches`, `press lever`.

Note: The fast loop confirms lever completion via `check_joint_angle("wrist_flex", "<", -0.8)`,
not via SmolVLA's output — so the instruction only needs to be close enough to trigger the
right motion trajectory.

### Release / place instructions

The robot opens the gripper to release an object at the current location.

| Good | Notes |
|---|---|
| `release the bread` | Opens gripper. |
| `open the gripper` | Direct gripper command — valid. |
| `place the cup on the table` | Place implies transport + open — valid if arm is already at target. |
| `drop the object` | Abrupt release — use only if drop is intended. |
| `let go of the bread` | Acceptable phrasing. |

Do not use: `release the bread into the toaster slot` (compound), `gently set the object down`.

---

## Good vs. bad: comprehensive table

| Task | Good instruction | Bad instruction | Problem |
|---|---|---|---|
| Start a grasp | `pick up the bread loaf` | `please pick up the bread loaf` | Politeness marker |
| Start a grasp | `pick up the bread loaf` | `pick up the bread loaf carefully` | Manner adverb |
| Start a grasp | `grasp the bread` | `get the bread and hold it` | Compound action |
| Move to position | `move arm to the right` | `move arm slightly to the right` | Vague qualifier |
| Move to position | `raise the arm` | `raise the arm to transport position` | Explanatory clause |
| Insert an object | `insert bread into toaster slot` | `put bread in the toaster slot` | Weak verb for slot insert |
| Insert an object | `insert bread into toaster slot` | `insert the bread into the toaster slot and make sure it's fully in` | Compound + length |
| Press a lever | `press the toaster lever down` | `push the lever downward until it clicks` | Conditional clause |
| Press a lever | `press the toaster lever down` | `depress the lever` | Out-of-distribution verb |
| Release object | `release the bread` | `release the bread gently` | Manner adverb |
| Release object | `open the gripper` | `open the gripper to release the object` | Explanatory clause |
| Check progress | (use tools, not instructions) | `are you done grasping?` | Question — never an instruction |
| Multi-step task | Split into phases | `pick up the bread and insert it` | Compound action |
| Describe state | (use tools, not instructions) | `the arm is near the toaster` | State description |

---

## Decomposing a complex task

A real-world manipulation task almost always requires multiple sub-actions. Each sub-action
must become its own phase with its own valid single-action instruction.

### Example: Toasting bread

High-level task: "place the bread in the toaster and start it"

Decompose as:

| Phase | Instruction | Success criterion |
|---|---|---|
| `grasp_bread` | `pick up the bread loaf` | `check_gripper_closed()` → True + visual confirm |
| `transport_bread` | `move the bread to the toaster` | `capture_camera_frame()` shows arm over toaster |
| `insert_bread` | `insert bread into toaster slot` | Visual: bread slot entry visible |
| `release_bread` | `release the bread` | `check_gripper_closed()` → False |
| `press_lever` | `press the toaster lever down` | `check_joint_angle("wrist_flex", "<", -0.8)` → True |

Do not try to encode the full task in a single instruction. SmolVLA cannot plan; it can only
execute one action at a time.

### Decomposition rules

1. **One verb per phase.** If you find yourself using "and", "then", or a comma between actions,
   split into two phases.

2. **Order matters physically.** You cannot insert before grasping. You cannot release before
   inserting. Think through physical dependencies.

3. **Release is a separate phase.** Many first attempts forget to include an explicit release
   phase. The gripper will not open on its own.

4. **Transport is usually a separate phase.** Grasp and transport can sometimes merge (`lift the
   bread and move it to the toaster`), but splitting them gives the orchestrator better control
   and makes timeout recovery easier.

5. **Consider approach angle.** If the insert requires a specific wrist orientation, you may
   need a separate "orient" phase before insert: `rotate wrist clockwise`, then `insert bread
   into toaster slot`.

---

## What happens when an out-of-scope instruction is sent

This section describes the failure mode in concrete terms so the consequences are clear.

SmolVLA's language encoder produces an embedding for any input — it does not refuse or error.
When the instruction is outside the training distribution, the embedding falls in a region of
the action-conditioning space that was never trained. The action-prediction head has no signal
to work with and produces actions that resemble whatever training examples are nearest in
embedding space, which may be:

- **Frozen output**: The arm stops moving or vibrates at a fixed frequency.
- **Drift to extremes**: One or more joints drive to their software limits, stalling the motors.
- **Erratic motion**: The arm sweeps through a sequence of unrelated movements from different
  training demonstrations.
- **Gripper cycling**: The gripper opens and closes rapidly with no task-related pattern.

All of these are physically hazardous if the arm is holding an object or is near a person.
The fast loop has no mechanism to detect that the motion is wrong — it faithfully executes
every action value SmolVLA returns.

**If you observe erratic motion**: Call `pause_robot()` immediately via the orchestrator,
diagnose the instruction, and re-issue a valid one before calling `resume_robot()`.

---

## SO-101 arm: 6 DOF and what spatial language maps to each

The SO-101 follower arm has six motorised joints. Knowing which joint each spatial instruction
activates helps you confirm success with `check_joint_angle` and helps you choose the right
instruction verb.

### Joint reference

| Index | Joint name | Physical motion | Spatial language that activates it |
|---|---|---|---|
| 0 | `shoulder_pan` | Horizontal rotation of the whole arm (left/right sweep) | "move left", "move right", "pan to the left", "turn toward" |
| 1 | `shoulder_lift` | Vertical elevation of the upper arm | "raise the arm", "lift up", "lower the arm", "reach forward" |
| 2 | `elbow_flex` | Bending of the elbow (fore/aft of end-effector) | "extend the arm", "reach out", "pull back", "reach into" |
| 3 | `wrist_flex` | Pitch of the wrist (up/down tilt of gripper face) | "tilt wrist down", "press down", "push lever down" |
| 4 | `wrist_roll` | Roll of the wrist (rotation around arm axis) | "rotate wrist", "twist", "turn gripper" |
| 5 | `gripper` | Gripper open/close (0=closed, 100=open) | "open gripper", "close gripper", "grasp", "release" |

### Joint value interpretation

All body joints (`shoulder_pan` through `wrist_roll`) use normalised float values roughly
in the range [-1, 1], where:
- 0 is the neutral / resting position
- Positive values are one extreme of travel
- Negative values are the other extreme

The `gripper` joint uses a different scale: 0–100, where 0 is fully closed and 100 is fully open.

### Confirming phase completion with joint checks

Use `check_joint_angle` to confirm that a spatial instruction has been executed:

| Instruction | Expected joint change | Check call |
|---|---|---|
| `press the toaster lever down` | `wrist_flex` goes strongly negative | `check_joint_angle("wrist_flex", "<", -0.8)` |
| `raise the arm` | `shoulder_lift` goes positive | `check_joint_angle("shoulder_lift", ">", 0.4)` |
| `extend the arm forward` | `elbow_flex` changes from neutral | `check_joint_angle("elbow_flex", ">", 0.3)` |
| `move arm to the right` | `shoulder_pan` changes | `check_joint_angle("shoulder_pan", ">", 0.3)` |
| `open the gripper` | `gripper` opens | `check_joint_angle("gripper", ">", 60)` |
| `grasp the object` | `gripper` closes | `check_gripper_closed(threshold=15)` |

Joint checks are a secondary confirmation only. Always pair them with `capture_camera_frame`
before advancing a phase. The arm can reach a joint angle without completing the intended
physical action (e.g. `shoulder_lift > 0.4` is true even if the gripper missed the object).

---

## Quick-reference: writing a new instruction

Use this checklist every time you write a phase instruction:

- [ ] Starts with an imperative verb
- [ ] 3–8 words total
- [ ] Single action (no "and", "then", or comma between actions)
- [ ] No manner adverbs (no "carefully", "gently", "slowly", "firmly")
- [ ] No politeness markers (no "please", "try to", "make sure")
- [ ] No questions or interrogatives
- [ ] No descriptions of robot state or reasoning
- [ ] Object name is specific enough to be unambiguous
- [ ] Passes the gut check: "would this phrase appear as a caption in a robot dataset?"

If you cannot satisfy all items, split the action into two phases or simplify the language
until the checklist passes.
