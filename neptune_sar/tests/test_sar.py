"""
Unit & Integration Tests for the Neptune North Search & Rescue (SAR)
Coordinator and 3 Autonomous Agents (A1, A2, A3).
"""

from neptune_sar import (
    AgentA1,
    AgentA2,
    AgentA3,
    AgentMode,
    Coordinator,
    compute_5x5_sight,
    create_sar_team,
    has_line_of_sight,
    parse_floor_plan,
    partition_floor_plan_into_sectors,
)
from neptune_sar.demo_sim import SAMPLE_FLOOR_PLAN, SAMPLE_TARGETS, run_episode


def test_parse_floor_plan_and_agent_ids() -> None:
    raw = [
        [1, "A1", "A2", "A3", 1],
        [1, 0,    1,    0,    1],
        [1, "P1", 0,    "P2", 1],
        [1, 1,    1,    1,    1],
    ]
    grid, agents, targets = parse_floor_plan(raw)

    assert agents == {"A1": (0, 1), "A2": (0, 2), "A3": (0, 3)}
    assert targets == {"P1": (2, 1), "P2": (2, 3)}
    # Agent start tiles and target tiles are walkable path (0)
    assert grid[0][1] == 0
    assert grid[0][2] == 0
    assert grid[1][2] == 1  # Wall preserved


def test_centred_5x5_sight_and_wall_occlusion() -> None:
    # 7x7 grid with a wall right above the agent at (3, 3) -> wall at (2, 3)
    grid = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],  # (2, 3) is a wall!
        [0, 0, 0, 0, 0, 0, 0],  # Agent at (3, 3)
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]
    obs = compute_5x5_sight(
        grid=grid,
        center=(3, 3),
        agent_id="A1",
        active_targets={"HiddenBehindWall": (1, 3), "VisibleSouth": (5, 3)},
    )

    # Window is strictly within [1..5] x [1..5] (5x5 centered at (3,3))
    for r, c in obs.visible_cells:
        assert 1 <= r <= 5 and 1 <= c <= 5

    # The wall face itself at (2, 3) is visible
    assert (2, 3) in obs.visible_cells
    # Cell directly behind the wall at (1, 3) MUST be blocked!
    assert (1, 3) not in obs.visible_cells
    assert "HiddenBehindWall" not in obs.visible_targets

    # Open cell at (5, 3) is visible
    assert (5, 3) in obs.visible_cells
    assert "VisibleSouth" in obs.visible_targets


def test_three_agents_have_distinct_specialized_logic() -> None:
    coordinator, agents = create_sar_team(SAMPLE_FLOOR_PLAN, total_targets_goal=4)

    assert isinstance(agents["A1"], AgentA1)
    assert isinstance(agents["A2"], AgentA2)
    assert isinstance(agents["A3"], AgentA3)

    # Each agent has its own primary sector and distinct spillover priority
    assert agents["A1"].primary_sector == 1
    assert agents["A2"].primary_sector == 2
    assert agents["A3"].primary_sector == 3

    assert agents["A1"].get_spillover_sector_order() != agents["A2"].get_spillover_sector_order()
    assert agents["A2"].get_spillover_sector_order() != agents["A3"].get_spillover_sector_order()

    # Given the exact same set of candidate cells, A1 (Deep Vanguard),
    # A2 (High-Visibility Scout), and A3 (Wavefront Support) apply their own scoring logic
    shared_candidates = set(coordinator.all_walkable)
    goal_a1, reason_a1 = agents["A1"].select_exploration_goal(shared_candidates)
    goal_a2, reason_a2 = agents["A2"].select_exploration_goal(shared_candidates)
    goal_a3, reason_a3 = agents["A3"].select_exploration_goal(shared_candidates)

    assert "A1 Deep-Vanguard" in reason_a1
    assert "A2 Recon-Scout" in reason_a2
    assert "A3 Support-Specialist" in reason_a3


def test_mission_succeeds_under_100_percent_interior_comms_blackout() -> None:
    # Comms only work when an agent physically stands on an entrance tile
    res = run_episode(
        floor_plan=SAMPLE_FLOOR_PLAN,
        targets=SAMPLE_TARGETS,
        comm_policy=lambda aid, pos, step, entrances: pos in entrances,
        max_steps=200,
    )
    assert res["actual_rescued_count"] == len(SAMPLE_TARGETS)
    assert res["mission_complete"] is True



def test_mission_succeeds_under_total_zero_comms_even_at_entrance() -> None:
    # Even if the Coordinator link is 100% dead at all times (including at the entrance),
    # the 3 agents independently search, spill over across sectors, pick up all 4 targets,
    # and extract them to the entrance!
    res = run_episode(
        floor_plan=SAMPLE_FLOOR_PLAN,
        targets=SAMPLE_TARGETS,
        comm_policy=lambda aid, pos, step, entrances: False,
        max_steps=220,
    )
    assert res["actual_rescued_count"] == len(SAMPLE_TARGETS)


if __name__ == "__main__":
    tests = [
        test_parse_floor_plan_and_agent_ids,
        test_centred_5x5_sight_and_wall_occlusion,
        test_three_agents_have_distinct_specialized_logic,
        test_mission_succeeds_under_100_percent_interior_comms_blackout,
        test_mission_succeeds_under_total_zero_comms_even_at_entrance,
    ]
    for fn in tests:
        fn()
        print(f"[PASS] {fn.__name__}")
    print(f"\nAll {len(tests)} tests passed successfully!")

