"""Strands LLM orchestrator for multi-phase ACT robot control.

Usage:
    conda activate lerobot
    python -m orchestrator.run \
        --port    /dev/tty.usbserial-XXX \
        --task    "place the bread in the toaster" \
        --phases  "grasp:your_hf_user/act_grasp_bread" \
                  "insert:your_hf_user/act_insert_bread" \
                  "press:your_hf_user/act_press_lever"

Each phase must have a separately trained ACT checkpoint.
ACT takes no language input — behaviour is determined by the active checkpoint.
The Strands agent monitors sensor state and switches policies at the right time.
"""

import argparse
import logging
import threading
import time

from .fast_loop import run as run_fast_loop
from .prompts import SYSTEM_PROMPT, build_initial_prompt, parse_phases
from .shared_state import SharedState
from .tools import make_tools

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)


def wait_for_fast_loop(state: SharedState, timeout: float = 60.0) -> None:
    """Block until the fast loop has written at least one observation."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if state.snapshot_obs():
            return
        time.sleep(0.1)
    raise TimeoutError("Fast loop did not produce an observation — check robot connection and model loading")


def main() -> None:
    parser = argparse.ArgumentParser(description="ACT + Strands multi-phase orchestrator")
    parser.add_argument("--port", required=True, help="SO-101 follower serial port")
    parser.add_argument("--task", required=True, help="High-level task description")
    parser.add_argument(
        "--phases", nargs="+", required=True, metavar="NAME:MODEL_ID",
        help="Ordered phases as 'phase_name:hf_repo_id_or_local_path'",
    )
    parser.add_argument(
        "--cameras", nargs="*", default=["front_cam:0"], metavar="NAME:INDEX",
        help="Cameras as 'name:opencv_index' (default: front_cam:0)",
    )
    parser.add_argument("--device", default="mps", help="Torch device (mps / cuda / cpu)")
    parser.add_argument("--claude-model", default="claude-sonnet-4-6")
    args = parser.parse_args()

    phases = parse_phases(args.phases)
    phase_model_map = {name: model_id for name, model_id in phases}
    phase_names = [name for name, _ in phases]

    cameras: dict[str, dict] = {}
    for cam_entry in args.cameras:
        if ":" not in cam_entry:
            raise ValueError(f"Camera entry must be 'name:index', got: {cam_entry!r}")
        name, _, idx = cam_entry.partition(":")
        cameras[name.strip()] = {"index_or_path": int(idx.strip()), "width": 640, "height": 480, "fps": 30}

    state = SharedState()

    fast_thread = threading.Thread(
        target=run_fast_loop,
        args=(state, args.port, cameras, phase_model_map, args.device),
        daemon=True,
        name="act-fast-loop",
    )
    fast_thread.start()
    logger.info("Fast loop started — loading ACT policies and connecting robot (may take ~30s)...")
    wait_for_fast_loop(state)
    logger.info("Robot online. Starting Strands orchestrator.")

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
    logger.info("Sending initial prompt to orchestrator agent...")
    try:
        response = agent(prompt)
        logger.info("Agent finished: %s", response)
    except KeyboardInterrupt:
        logger.info("Interrupted — stopping robot")
        state.stop = True

    fast_thread.join(timeout=5.0)
    logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
