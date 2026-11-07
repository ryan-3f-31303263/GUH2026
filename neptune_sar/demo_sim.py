"""
Integration Harness & Demonstration for the Coordinator and 3 Agents (A1, A2, A3).

Shows how teammates' environment / GUI calls `coordinator` and `agents['A1'|'A2'|'A3']`
and compares performance under:
1. Full 1-to-1 Comms
2. Realistic RF Range / Intermittent Comms Failure
3. Total Interior Blackout (Comms severed everywhere inside)
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Sequence, Tuple

from .agents import BaseAgent, create_sar_team
from .coordinator import Coordinator
from .models import Coord, StepDecision
from .vision_and_grid import parse_floor_plan


SAMPLE_FLOOR_PLAN: List[List[Any]] = [
    [1, 1, 1, 1, 1, "A1", "A2", "A3", 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 0,    0,    0,    1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0,    1,    0,    1, 0, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 0,    1,    0,    0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 0,    1,    1,    1, 1, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0,    0,    0,    0, 0, 1, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 1,    1,    0,    1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 1, 0,    0,    0,    1, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0,    1,    1,    1, 1, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 0, 0,    0,    0,    0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1,    1,    1,    1, 1, 1, 1, 1, 1, 1],
]

SAMPLE_TARGETS: Dict[str, Coord] = {
    "P1": (1, 1),    # Top-left room (Sector 1)
    "P2": (7, 5),    # Deep central chamber (Sector 2)
    "P3": (1, 13),   # Top-right wing (Sector 3)
    "P4": (9, 13),   # Bottom-right deep corner
}


def run_episode(
    floor_plan: Sequence[Sequence[Any]],
    targets: Dict[str, Coord],
    comm_policy: Callable[[str, Coord, int, List[Coord]], bool],
    max_steps: int = 200,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Runs a full Search & Rescue simulation episode with 1 Coordinator and 3 Agents (A1, A2, A3).

    `comm_policy(agent_id, pos, step, entrances) -> bool` determines whether 1-to-1
    communication between `agent_id` and the outside Coordinator succeeds on `step`.
    """
    active_targets: Dict[str, Coord] = dict(targets)
    rescued_targets: List[str] = []

    coordinator, agents = create_sar_team(
        floor_plan=floor_plan,
        total_targets_goal=len(targets),
    )

    step_history: List[Dict[str, StepDecision]] = []

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
            )
            step_decisions[aid] = decision
            all_positions[aid] = decision.next_pos

            # Update environment target map if agent picked up or rescued a target
            if decision.picked_up_target_id and decision.picked_up_target_id in active_targets:
                del active_targets[decision.picked_up_target_id]

            if decision.rescued_target_id and decision.rescued_target_id not in rescued_targets:
                rescued_targets.append(decision.rescued_target_id)

        step_history.append(step_decisions)

        if verbose and (step <= 5 or step % 15 == 0 or len(rescued_targets) == len(targets)):
            print(f"\n--- Step {step} | Rescued: {len(rescued_targets)}/{len(targets)} ---")
            for aid, dec in step_decisions.items():
                comm_str = "ONLINE" if dec.synced_with_coordinator else "OFFLINE"
                print(
                    f"  {aid} @ {dec.next_pos} [{dec.mode.value}] ({comm_str}) "
                    f"carry={dec.carrying_target_id} | {dec.reasoning}"
                )

        if len(rescued_targets) >= len(targets):
            break

    summary = coordinator.get_summary()
    summary["actual_rescued_count"] = len(rescued_targets)
    summary["steps_taken"] = len(step_history)
    return summary


def main() -> None:
    print("=" * 72)
    print("NEPTUNE NORTH CHALLENGE — 1 COORDINATOR + 3 INDEPENDENT AGENTS (A1, A2, A3)")
    print("=" * 72)

    scenarios = [
        (
            "1. Full 1-to-1 Comms (Always Connected)",
            lambda aid, pos, step, entrances: True,
        ),
        (
            "2. Degraded RF Comms (Only within 4 tiles of Entrance)",
            lambda aid, pos, step, entrances: any(
                abs(pos[0] - e[0]) + abs(pos[1] - e[1]) <= 4 for e in entrances
            ),
        ),
        (
            "3. Total Interior Comms Blackout (Zero Comms Inside Facility)",
            lambda aid, pos, step, entrances: pos in entrances,
        ),
    ]

    for name, policy in scenarios:
        res = run_episode(
            floor_plan=SAMPLE_FLOOR_PLAN,
            targets=SAMPLE_TARGETS,
            comm_policy=policy,
            max_steps=200,
            verbose=False,
        )
        print(f"\nScenario: {name}")
        print(
            f"  Targets Rescued : {res['actual_rescued_count']} / {res['targets_goal']}"
        )
        print(f"  Steps Taken     : {res['steps_taken']}")
        print(f"  1-to-1 Syncs    : {res['sync_counts']}")
        print(f"  Comms Failures  : {res['failed_syncs']}")


if __name__ == "__main__":
    main()
