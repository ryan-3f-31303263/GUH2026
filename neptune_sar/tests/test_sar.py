"""
Unit & Integration Tests for the Neptune North Search & Rescue (SAR)
Coordinator, 3 Autonomous Agents (A1, A2, A3), 2-Agent Doors ('D'),
and Offline Ground Message Markers ('I' and 'O').
"""

from neptune_sar import (
    AgentA1,
    AgentA2,
    AgentA3,
    AgentMode,
    Coordinator,
    GroundMarker,
    MarkerType,
    RescueAgent,
    can_unlock_door,
    compute_5x5_sight,
    create_sar_team,
    extract_doors_from_floor_plan,
    parse_floor_plan,
)
from neptune_sar.demo_sim import SAMPLE_FLOOR_PLAN, SAMPLE_TARGETS, run_episode


def test_parse_floor_plan_and_agent_ids() -> None:
    raw = [
        [1, "A1", "A2", "A3", 1],
        [1, 0,    "D",  0,    1],
        [1, "P1", 0,    "P2", 1],
        [1, 1,    1,    1,    1],
    ]
    grid, agents, targets = parse_floor_plan(raw)
    doors = extract_doors_from_floor_plan(raw)

    assert agents == {"A1": (0, 1), "A2": (0, 2), "A3": (0, 3)}
    assert targets == {"P1": (2, 1), "P2": (2, 3)}
    assert doors == {(1, 2)}
    assert grid[0][1] == 0
    assert grid[1][2] == 0  # Underlying structural tile of 'D' is walkable once unlocked


def test_centred_5x5_sight_and_wall_occlusion() -> None:
    grid = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],  # (2, 3) is a wall
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

    for r, c in obs.visible_cells:
        assert 1 <= r <= 5 and 1 <= c <= 5

    assert (2, 3) in obs.visible_cells
    assert (1, 3) not in obs.visible_cells
    assert "HiddenBehindWall" not in obs.visible_targets
    assert (5, 3) in obs.visible_cells
    assert "VisibleSouth" in obs.visible_targets


def test_three_agents_have_distinct_specialized_logic() -> None:
    coordinator, agents = create_sar_team(SAMPLE_FLOOR_PLAN, total_targets_goal=4)

    assert isinstance(agents["A1"], AgentA1)
    assert isinstance(agents["A2"], AgentA2)
    assert isinstance(agents["A3"], AgentA3)

    assert agents["A1"].primary_sector == 1
    assert agents["A2"].primary_sector == 2
    assert agents["A3"].primary_sector == 3

    shared_candidates = set(coordinator.all_walkable)
    _, reason_a1 = agents["A1"].select_exploration_goal(shared_candidates)
    _, reason_a2 = agents["A2"].select_exploration_goal(shared_candidates)
    _, reason_a3 = agents["A3"].select_exploration_goal(shared_candidates)

    assert "A1 Deep-Vanguard" in reason_a1
    assert "A2 Recon-Scout" in reason_a2
    assert "A3 Support-Specialist" in reason_a3


def test_door_D_requires_two_agents_and_invite_marker_I_when_signal_is_down() -> None:
    # Corridor with a closed Door 'D' at (2, 2) blocking target P1 at (3, 2)
    raw = [
        [1, "A1", "A2", "A3", 1],
        [1, 0,    0,    0,    1],
        [1, 1,    "D",  1,    1],
        [1, 1,    0,    1,    1],
        [1, 1,    1,    1,    1],
    ]
    closed_doors = extract_doors_from_floor_plan(raw)
    ground_markers = {}
    active_targets = {"P1": (3, 2)}

    # Verify 1 agent alone cannot unlock 'D'
    assert not can_unlock_door((2, 2), {"A1": (1, 2), "A2": (0, 1), "A3": (0, 3)})
    # Verify 2 agents adjacent to 'D' CAN unlock 'D'
    assert can_unlock_door((2, 2), {"A1": (1, 2), "A2": (1, 2), "A3": (0, 3)})

    coordinator, agents = create_sar_team(raw, total_targets_goal=1)

    # Step 1 with interior signal DOWN: A2 at (0, 2) moves to (1, 2) adjacent to Door (2, 2)
    # and drops an 'I' marker inviting another agent to (2, 2)
    rescued = []
    for step in range(1, 25):
        positions = {aid: ag.position for aid, ag in agents.items()}
        for aid in ("A1", "A2", "A3"):
            dec = agents[aid].step(
                coordinator=coordinator,
                comm_available=False,  # Interior signal is completely DOWN
                active_targets=active_targets,
                all_agent_positions=positions,
                closed_doors=closed_doors,
                ground_markers=ground_markers,
            )
            positions[aid] = dec.next_pos
            if dec.picked_up_target_id in active_targets:
                del active_targets[dec.picked_up_target_id]
            if dec.rescued_target_id:
                rescued.append(dec.rescued_target_id)
        if rescued:
            break

    assert (2, 2) not in closed_doors, "Door 'D' should be unlocked by 2 agents!"
    assert rescued == ["P1"], "Target behind Door 'D' should be rescued and brought out!"


def test_dead_end_marker_O_prevents_redundant_dead_end_search_when_signal_down() -> None:
    raw = [
        [1, "A1", "A2", "A3", 1],
        [1, 0,    0,    0,    1],
        [1, 0,    1,    0,    1],
        [1, 0,    1,    0,    1],
        [1, 1,    1,    1,    1],
    ]
    coordinator, agents = create_sar_team(raw, total_targets_goal=1)
    # Place an 'O' marker at (2, 1) indicating dead-end cells {(2, 1), (3, 1)} are already searched
    dead_cells = {(2, 1), (3, 1)}
    ground_markers = {
        (2, 1): GroundMarker(
            marker_type=MarkerType.DEAD_END,
            position=(2, 1),
            author_agent="A1",
            dead_end_cells=dead_cells,
            created_step=1,
        )
    }
    # When A2 at (1, 2) observes (2, 1) within its 5x5 sight, it immediately marks {(2, 1), (3, 1)} as cleared
    agents["A2"].position = (1, 2)
    agents["A2"].observe(ground_markers=ground_markers)
    assert dead_cells.issubset(agents["A2"].local_cleared_cells)


def test_sudden_signal_cutoff_and_always_on_outside_entrance_multi_trips() -> None:
    # Signal starts ON, then suddenly drops at step 6 inside the maze,
    # while remaining ALWAYS ON when an agent steps out to the entrance
    res = run_episode(
        floor_plan=SAMPLE_FLOOR_PLAN,
        targets=SAMPLE_TARGETS,
        comm_policy=lambda aid, pos, step, entrances: (pos in entrances) or (step < 6),
        max_steps=220,
    )
    assert res["actual_rescued_count"] == len(SAMPLE_TARGETS)
    assert (6, 7) in res["doors_unlocked"]
    assert res["total_trips_completed"] >= len(SAMPLE_TARGETS)
    assert res["mission_complete"] is True


if __name__ == "__main__":
    tests = [
        test_parse_floor_plan_and_agent_ids,
        test_centred_5x5_sight_and_wall_occlusion,
        test_three_agents_have_distinct_specialized_logic,
        test_door_D_requires_two_agents_and_invite_marker_I_when_signal_is_down,
        test_dead_end_marker_O_prevents_redundant_dead_end_search_when_signal_down,
        test_sudden_signal_cutoff_and_always_on_outside_entrance_multi_trips,
    ]
    for fn in tests:
        fn()
        print(f"[PASS] {fn.__name__}")
    print(f"\nAll {len(tests)} tests passed successfully!")
