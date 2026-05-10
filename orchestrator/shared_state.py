"""Thread-safe shared state between the 20Hz fast loop and the Strands orchestrator."""

import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SharedState:
    # Name of the currently active ACT policy — agent writes this, fast loop reads it.
    # Must match a key in the policies dict loaded at startup (e.g. "grasp", "insert").
    active_policy_name: str = ""

    # Fast loop control
    paused: bool = False   # agent sets True to freeze robot during phase transitions
    stop: bool = False     # shutdown signal

    # Latest raw observation dict (numpy arrays) — fast loop writes, tools read.
    # Keys: "observation.images.<name>" (HxWxC uint8) and "observation.state" (float32 array)
    obs: dict[str, Any] = field(default_factory=dict)

    # Phase tracking — written by agent tools
    phase: str = "init"
    phase_start_time: float = field(default_factory=time.monotonic)

    # Signals the fast loop to call policy.reset() on the next cycle.
    # Set True by advance_phase tool; fast loop clears it after reset.
    reset_requested: bool = False

    # Home position request — agent sets home_requested = True and populates
    # home_position; fast loop drives joints there then clears the flag.
    home_requested: bool = False
    home_position: dict[str, float] = field(default_factory=dict)

    # Protects all fields above
    lock: threading.Lock = field(default_factory=threading.Lock)

    def set_active_policy(self, name: str) -> None:
        with self.lock:
            self.active_policy_name = name
            self.reset_requested = True   # flush the outgoing policy's action queue

    def set_phase(self, name: str) -> None:
        with self.lock:
            self.phase = name
            self.phase_start_time = time.monotonic()

    def elapsed_in_phase(self) -> float:
        with self.lock:
            return time.monotonic() - self.phase_start_time

    def snapshot_obs(self) -> dict[str, Any]:
        """Return a shallow copy of the latest observation (safe to read outside lock)."""
        with self.lock:
            return dict(self.obs)
