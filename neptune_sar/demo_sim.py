"""
Integration Harness & Demonstration for the Coordinator and 3 Agents (A1, A2, A3).

Demonstrates:
1. Full 1-to-1 Comms
2. Sudden Signal Cutoff (Signal stops suddenly mid-mission; always ON outside the maze)
3. Heavy Door 'D' requiring 2 agents to unblock + Offline Ground Markers ('I' & 'O')
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

from agents import BaseAgent, create_sar_team
from coordinator import Coordinator
from models import Coord, GroundMarker, StepDecision
from vision_and_grid import extract_doors_from_floor_plan

# import maze 
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maze import *
from maze_vis import *

# init maze 
myMaze = Maze(18)
myMaze.gen_wounded(5)
myMaze.seal()
myMaze.poke(4, 4)
myGrid = MazeGrid(schema=myMaze.maze, rows=myMaze.n, columns=myMaze.n)
myGrid.run()
input()
myGrid.close()


SAMPLE_FLOOR_PLAN: List[List[Any]] = [
    [1, 1, 1, 1, 1, "A1", "A2", "A3", 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 0,    0,    0,    1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0,    1,    0,    1, 0, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 0,    1,    0,    0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 0,    1,    1,    1, 1, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0,    0,    0,    0, 0, 1, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 1,    1,    "D",  1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 1, 0,    0,    0,    1, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0,    1,    1,    1, 1, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 0, 0,    0,    0,    0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1,    1,    1,    1, 1, 1, 1, 1, 1, 1],
]

SAMPLE_TARGETS: Dict[str, Coord] = {
    "P1": (1, 1),    # Top-left room (Sector 1)
    "P2": (7, 5),    # Deep central chamber behind Door 'D' at (6, 7)
    "P3": (1, 13),   # Top-right wing (Sector 3)
    "P4": (9, 13),   # Bottom-right deep corner
}


def run_episode(
    floor_plan: Sequence[Sequence[Any]],
    targets: Dict[str, Coord],
    comm_policy: Callable[[str, Coord, int, List[Coord]], bool],
    max_steps: int = 220,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Runs a full Search & Rescue simulation episode with:
      - 1 Coordinator (outside the maze)
      - 3 Agents (A1, A2, A3) going in and out of the maze multiple times
      - Heavy Doors 'D' (require 2 agents adjacent to unlock)
      - Offline Ground Markers 'I' (invite to door 'D') and 'O' (dead end ahead)
    """
    active_targets: Dict[str, Coord] = dict(targets)
    closed_doors: Set[Coord] = extract_doors_from_floor_plan(floor_plan)
    ground_markers: Dict[Coord, GroundMarker] = {}
    rescued_targets: List[str] = []

    coordinator, agents = create_sar_team(
        floor_plan=floor_plan,
        total_targets_goal=len(targets),
    )

    step_history: List[Dict[str, StepDecision]] = []
    markers_dropped_history: List[GroundMarker] = []
    doors_unlocked_history: List[Coord] = []

    for step in range(1, max_steps + 1):
        all_positions = {aid: ag.position for aid, ag in agents.items()}
        step_decisions: Dict[str, StepDecision] = {}

        for aid in ("A1", "A2", "A3"):
            agent = agents[aid]

            decision = agent.step(
                coordinator=coordinator,
                comm_available=lambda pos, _aid=aid, _s=step: comm_policy(
                    _aid, pos, _s, coordinator.entrances
                ),
                active_targets=active_targets,
                all_agent_positions=all_positions,
                closed_doors=closed_doors,
                ground_markers=ground_markers,
            )
            step_decisions[aid] = decision
            all_positions[aid] = decision.next_pos

            if decision.dropped_marker is not None:
                markers_dropped_history.append(decision.dropped_marker)

            if (
                decision.unlocked_door_pos is not None
                and decision.unlocked_door_pos not in doors_unlocked_history
            ):
                doors_unlocked_history.append(decision.unlocked_door_pos)

            if (
                decision.picked_up_target_id
                and decision.picked_up_target_id in active_targets
            ):
                del active_targets[decision.picked_up_target_id]

            if (
                decision.rescued_target_id
                and decision.rescued_target_id not in rescued_targets
            ):
                rescued_targets.append(decision.rescued_target_id)

        step_history.append(step_decisions)

        if len(rescued_targets) >= len(targets):
            break

    summary = coordinator.get_summary()
    summary["actual_rescued_count"] = len(rescued_targets)
    summary["steps_taken"] = len(step_history)
    summary["doors_unlocked"] = list(doors_unlocked_history)
    summary["markers_dropped_count"] = len(markers_dropped_history)
    summary["total_trips_completed"] = sum(ag.trips_completed for ag in agents.values())
    return summary


def main() -> None:
    print("=" * 76)
    print("NEPTUNE NORTH CHALLENGE — COORDINATOR + 3 AGENTS + DOORS ('D') + MARKERS ('I','O')")
    print("=" * 76)

    scenarios = [
        (
            "1. Full 1-to-1 Comms (Always Connected)",
            lambda aid, pos, step, entrances: True,
        ),
        (
            "2. Sudden Signal Cutoff at Step 8 (Always ON Outside at Entrance)",
            lambda aid, pos, step, entrances: (pos in entrances) or (step < 8),
        ),
        (
            "3. Total Interior Blackout (Only Entrance Has Signal + Offline 'I'/'O' Markers)",
            lambda aid, pos, step, entrances: pos in entrances,
        ),
    ]

    for name, policy in scenarios:
        res = run_episode(
            floor_plan=SAMPLE_FLOOR_PLAN,
            targets=SAMPLE_TARGETS,
            comm_policy=policy,
            max_steps=220,
            verbose=False,
        )
        print(f"\nScenario: {name}")
        print(
            f"  Targets Rescued   : {res['actual_rescued_count']} / {res['targets_goal']}"
        )
        print(f"  Steps Taken       : {res['steps_taken']}")
        print(f"  Doors 'D' Opened  : {res['doors_unlocked']}")
        print(f"  Markers ('I'/'O') : {res['markers_dropped_count']}")
        print(f"  Multi-Exit Trips  : {res['total_trips_completed']}")
        print(f"  1-to-1 Syncs      : {res['sync_counts']}")


if __name__ == "__main__":
    main()
