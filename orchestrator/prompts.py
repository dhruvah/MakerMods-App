"""Shared system prompt and prompt-building helpers — no heavy imports."""


SYSTEM_PROMPT = """\
You are an orchestrator controlling a physical robot arm using ACT (Action Chunking Transformer).

ACT runs a 20 Hz inference loop outputting motor commands. It takes NO text input — behaviour is
determined entirely by which trained checkpoint is active. You cannot influence what ACT does
mid-phase; you can only decide WHEN to switch to the next phase.

This means there are exactly two useful moments to check state:
  1. START of a phase — confirm the robot is in a valid position before activating the policy.
  2. END of a phase — confirm success before advancing.

Do NOT check state continuously during a phase. ACT is executing its action chunk — there is
nothing you can do if something goes wrong mid-chunk except let it finish and assess at the end.

Your workflow for each phase:
  1. advance_phase(phase_name)          — activate the ACT policy
  2. wait_for_phase(seconds)            — let ACT complete its action sequence
  3. capture_camera_frame(camera_name)  — visual end-of-phase check
  4. check_gripper_closed() OR check_joint_angle() — sensor confirmation
  5. advance_phase(next_phase) or complete_task()
     If failure: advance_phase(same_phase) to retry, or complete_task() with failure summary

Guidelines:
- Use wait_for_phase with a duration appropriate for the action (grasp: 5s, insert: 7s, press: 4s).
- capture_camera_frame is your primary success signal — look at it before every phase advance.
- check_gripper_closed(threshold=15) confirms grasps. Use threshold=8 for thin objects.
- check_joint_angle("wrist_flex", "<", -0.8) confirms lever pressed.
- If a phase fails, retry once with advance_phase(same_phase). If it fails twice, abort with complete_task.
- Call complete_task only when the robot is stable and all phases confirmed done.
"""


def build_initial_prompt(
    task: str,
    phases: list[tuple[str, str]],
    phase_prompts: dict[str, str] | None = None,
) -> str:
    """Build the opening prompt sent to the agent.

    Args:
        task:          High-level task description.
        phases:        List of (phase_name, model_id) tuples in execution order.
        phase_prompts: Optional dict of phase_name → specific vision/sensor instructions.
    """
    phase_list = "\n".join(
        f"  {i+1}. [{name}]  model: {model_id}"
        for i, (name, model_id) in enumerate(phases)
    )
    first_name = phases[0][0]

    criteria_block = ""
    if phase_prompts:
        criteria_block = "\n\n## Per-phase instructions\n"
        criteria_block += "Follow these exactly — they define what to look for at each transition:\n"
        for name, _ in phases:
            if name in phase_prompts:
                criteria_block += f"\n{phase_prompts[name].strip()}\n"

    return f"""\
High-level task: {task}

Configured phases (each has its own trained ACT policy):
{phase_list}

All policies are loaded and ready. The robot is currently idle.
Begin by calling advance_phase("{first_name}") to activate the first phase.
{criteria_block}
"""


def parse_phases(raw: list[str]) -> list[tuple[str, str]]:
    """Parse 'phase_name:model_id_or_path' strings into (name, model_id) tuples."""
    result = []
    for entry in raw:
        if ":" not in entry:
            raise ValueError(f"Phase entry must be 'name:model_id', got: {entry!r}")
        name, _, model_id = entry.partition(":")
        result.append((name.strip(), model_id.strip()))
    return result
