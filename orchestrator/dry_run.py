"""Dry-run entry point — full ACT agent loop with mock robot, no hardware needed.

Runs the complete Strands orchestrator + all 9 tools against the mock robot.
The mock simulates realistic joint/gripper state changes per phase so the agent
can make real sensor-based decisions.

Usage (from makermod_app/ root):

    conda activate lerobot
    export ANTHROPIC_API_KEY=sk-ant-...
    python -m orchestrator.dry_run

Custom task:

    python -m orchestrator.dry_run \
      --task "place the bread in the toaster" \
      --phases "grasp:mock/grasp" "insert:mock/insert" "press:mock/press"

Note: model IDs are ignored in dry-run mode (no real models loaded).
Use meaningful names so the mock robot can simulate correct behaviour.
"""

import argparse
import logging
import threading
import time

from .mock_robot import run as run_mock
from .prompts import SYSTEM_PROMPT, build_initial_prompt, parse_phases
from .shared_state import SharedState
from .tools import make_tools

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TASK = "place the bread in the toaster"
DEFAULT_PHASES = [
    "grasp:mock/act_grasp_bread",
    "insert:mock/act_insert_bread",
    "press:mock/act_press_lever",
]
DEFAULT_CAMERAS = ["front_cam:0", "hand_cam:1"]


def wait_for_mock(state: SharedState, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if state.snapshot_obs():
            return
        time.sleep(0.05)
    raise TimeoutError("Mock robot did not produce an observation")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run ACT orchestrator against mock robot")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--phases", nargs="+", default=DEFAULT_PHASES, metavar="NAME:MODEL_ID")
    parser.add_argument("--cameras", nargs="*", default=DEFAULT_CAMERAS, metavar="NAME:INDEX")
    parser.add_argument("--claude-model", default="claude-sonnet-4-6")
    args = parser.parse_args()

    phases = parse_phases(args.phases)
    phase_names = [name for name, _ in phases]

    cameras: dict = {}
    for entry in args.cameras:
        name, _, idx = entry.partition(":")
        cameras[name.strip()] = {"index_or_path": int(idx.strip()) if idx.strip() else 0}

    state = SharedState()

    mock_thread = threading.Thread(
        target=run_mock,
        kwargs={"state": state, "cameras": cameras},
        daemon=True,
        name="mock-robot",
    )
    mock_thread.start()
    wait_for_mock(state)
    logger.info("Mock robot online. Starting dry-run orchestrator.")

    from strands import Agent
    from strands.models.anthropic import AnthropicModel

    model = AnthropicModel(
        model_id=args.claude_model,
        max_tokens=4096,
        use_native_token_count=False,
    )
    tools = make_tools(state, phase_names)
    agent = Agent(model=model, tools=tools, system_prompt=SYSTEM_PROMPT)

    prompt = build_initial_prompt(args.task, phases)
    logger.info("Sending initial prompt...")
    try:
        response = agent(prompt)
        logger.info("Agent finished: %s", response)
    except KeyboardInterrupt:
        logger.info("Interrupted — stopping")
        state.stop = True

    mock_thread.join(timeout=3.0)
    logger.info("Dry run complete")


if __name__ == "__main__":
    main()
