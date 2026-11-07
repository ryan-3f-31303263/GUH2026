"""
Autonomous Field Agents (A1, A2, A3) for the Neptune North Search & Rescue Challenge.

Features:
1. Working towards a common goal:
   All 3 agents collaborate to locate and rescue all N targets, going in and out of the
   maze multiple times until every target is extracted.
2. Independent decision-making under sudden signal loss:
   - Each agent (A1, A2, A3) runs its own specialized search policy.
   - Signal is ALWAYS ON when an agent is outside the maze (at `entrances`).
   - Inside the maze, signal may stop suddenly at any moment.
3. Cooperative 2-Agent Door ('D') Unblocking:
   - Element 'D' is a heavy door that blocks passage until 2 agents stand adjacent to it.
4. Stigmergic Ground Messages ('I' and 'O') when Signal is Down:
   - 'I' (Invite/Assist Door): Left on the ground when signal is down to ask any passing
     teammate who sees 'I' in its 5x5 sight to go to the designated Door 'D' location.
   - 'O' (Dead End): Left at the mouth of a searched dead-end branch when signal is down
     so other agents seeing 'O' know the road ahead is a dead end and skip it.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .coordinator import Coordinator, STATUS_PRIORITY
from .models import (
    AgentMode,
    AgentReport,
    Coord,
    CoordinatorDirective,
    GroundMarker,
    MarkerType,
    StepDecision,
    TargetRecord,
    TargetStatus,
    VisibleObservation,
)
from .vision_and_grid import (
    bfs_distance_map,
    bfs_shortest_path,
    can_unlock_door,
    compute_5x5_sight,
    detect_dead_end_branch,
    get_door_stand_positions,
    get_valid_neighbors,
    parse_floor_plan,
)


class BaseAgent(ABC):
    """
    Base autonomous Search & Rescue agent with:
    - Centred 5x5 line-of-sight perception (blocked by walls '1' and closed doors 'D')
    - Multi-entry / multi-exit trips in and out of the maze
    - Always-ON signal when outside the maze (at entrance), resilient to sudden interior blackouts
    - 2-Agent cooperative Door ('D') unlocking
    - Offline ground message markers ('I' for door invite, 'O' for dead-end road)
    """

    def __init__(
        self,
        agent_id: str,
        initial_pos: Coord,
        briefing: Dict[str, Any],
    ) -> None:
        self.agent_id: str = agent_id
        self.position: Coord = initial_pos
        self.grid: List[List[int]] = briefing["grid"]
        self.rows: int = len(self.grid)
        self.cols: int = len(self.grid[0]) if self.rows > 0 else 0
        self.entrances: List[Coord] = list(briefing["entrances"])
        self.total_targets_goal: int = briefing["total_targets_goal"]
        self.primary_sector: int = briefing["primary_sector"]
        self.active_sector: int = self.primary_sector
        self.sectors: Dict[int, Set[Coord]] = {
            sid: set(cells) for sid, cells in briefing["sectors"].items()
        }
        self.all_walkable: Set[Coord] = set().union(*self.sectors.values())
        self.entrance_dist_map: Dict[Coord, int] = dict(briefing["entrance_dist_map"])

        # Local belief state (maintained independently when signal stops suddenly)
        self.step_count: int = 0
        self.mode: AgentMode = AgentMode.SEARCH_PRIMARY
        self.local_cleared_cells: Set[Coord] = set()
        self.unsynced_cleared_cells: Set[Coord] = set()
        self.known_targets: Dict[str, TargetRecord] = {}
        self.rescued_target_ids: Set[str] = set()
        self.carrying_target_id: Optional[str] = None
        self.trips_completed: int = 0

        # Doors ('D') & Ground Markers ('I', 'O') tracking
        self.known_closed_doors: Set[Coord] = set()
        self.opened_doors: Set[Coord] = set()
        self.invited_door_goal: Optional[Coord] = None
        self.door_wait_steps: Dict[Coord, int] = {}
        self.placed_dead_end_markers: Set[Coord] = set()
        self.placed_invite_doors: Set[Coord] = set()

        # Comms & coordination tracking
        self.last_sync_step: int = 0
        self.steps_since_sync: int = 0
        self.dark_teammates: Set[str] = set()
        self.last_known_peer_positions: Dict[str, Coord] = {}
        self.coordinator_priority_target: Optional[str] = None
        self.coordinator_priority_door: Optional[Coord] = None
        self.coordinator_waypoint: Optional[Coord] = None
        self.mission_complete_flag: bool = False

        # Navigation history
        self.current_goal: Optional[Coord] = None
        self.recent_positions: List[Coord] = [initial_pos]

    def is_outside_maze(self, pos: Optional[Coord] = None) -> bool:
        """
        Returns True if the agent is outside the maze at the entrance/perimeter.
        Signal to the Coordinator is ALWAYS ON when outside the maze.
        """
        check_pos = self.position if pos is None else pos
        return check_pos in self.entrances

    # ------------------------------------------------------------------
    # 1. Perception: Centred 5x5 Sight (Walls, Doors 'D', Markers 'I'/'O')
    # ------------------------------------------------------------------
    def observe(
        self,
        active_targets: Optional[Dict[str, Coord]] = None,
        all_agent_positions: Optional[Dict[str, Coord]] = None,
        closed_doors: Optional[Set[Coord]] = None,
        ground_markers: Optional[Dict[Coord, GroundMarker]] = None,
        precomputed_observation: Optional[VisibleObservation] = None,
    ) -> VisibleObservation:
        """
        Perceives the centred 5x5 area around `self.position` (blocked by walls '1'
        and closed doors 'D'), reads any visible ground markers ('I' or 'O'),
        and updates local belief state.
        """
        effective_doors = (
            closed_doors if closed_doors is not None else self.known_closed_doors
        )
        obs = precomputed_observation or compute_5x5_sight(
            grid=self.grid,
            center=self.position,
            agent_id=self.agent_id,
            active_targets=active_targets,
            all_agent_positions=all_agent_positions,
            closed_doors=effective_doors,
            ground_markers=ground_markers,
        )

        # 1. Mark all visible walkable (0) cells as cleared
        for cell, tile_val in obs.visible_cells.items():
            if tile_val == 0 and cell not in effective_doors:
                if cell not in self.local_cleared_cells:
                    self.local_cleared_cells.add(cell)
                    self.unsynced_cleared_cells.add(cell)

        # 2. Record any closed doors ('D') visible in 5x5 sight
        for dpos in obs.visible_doors:
            if dpos not in self.opened_doors:
                self.known_closed_doors.add(dpos)

        # If a previously known closed door is now in visible_cells (as open 0), mark it opened
        for dpos in list(self.known_closed_doors):
            if closed_doors is not None and dpos not in closed_doors:
                self.known_closed_doors.discard(dpos)
                self.opened_doors.add(dpos)
                if self.invited_door_goal == dpos:
                    self.invited_door_goal = None
                if self.coordinator_priority_door == dpos:
                    self.coordinator_priority_door = None

        # 3. Read visible ground markers ('I' and 'O')
        for mpos, marker in obs.visible_markers.items():
            if marker.marker_type == MarkerType.DEAD_END:
                # 'O' Marker: The road after this marker is a dead end -> mark all as cleared!
                new_dead = marker.dead_end_cells - self.local_cleared_cells
                if new_dead:
                    self.local_cleared_cells.update(new_dead)
                    self.unsynced_cleared_cells.update(new_dead)
                if mpos not in self.local_cleared_cells:
                    self.local_cleared_cells.add(mpos)
                    self.unsynced_cleared_cells.add(mpos)

            elif marker.marker_type == MarkerType.INVITE_DOOR:
                # 'I' Marker: Another agent asks us to go to designated door location 'D'!
                dpos = marker.target_door_pos
                if dpos is not None and dpos not in self.opened_doors:
                    if closed_doors is None or dpos in closed_doors:
                        self.known_closed_doors.add(dpos)
                        if (
                            marker.author_agent != self.agent_id
                            and self.carrying_target_id is None
                        ):
                            self.invited_door_goal = dpos

        # 4. Record any targets visible in 5x5 sight
        for tid, tpos in obs.visible_targets.items():
            if tid in self.rescued_target_ids:
                continue
            if tid not in self.known_targets:
                self.known_targets[tid] = TargetRecord(
                    target_id=tid,
                    position=tpos,
                    status=TargetStatus.DISCOVERED,
                    assigned_agent=None,
                    updated_step=self.step_count,
                )
            else:
                rec = self.known_targets[tid]
                if rec.status == TargetStatus.DISCOVERED:
                    rec.position = tpos
                    rec.updated_step = self.step_count

        # 5. Update local knowledge of nearby peers seen in 5x5 window
        for peer_id, peer_pos in obs.visible_agents.items():
            self.last_known_peer_positions[peer_id] = peer_pos

        return obs

    # ------------------------------------------------------------------
    # 2. 1-to-1 Communication with Coordinator
    # ------------------------------------------------------------------
    def build_report(self) -> AgentReport:
        """Constructs the 1-to-1 uplink report for the Coordinator."""
        return AgentReport(
            agent_id=self.agent_id,
            step=self.step_count,
            position=self.position,
            mode=self.mode,
            carrying_target_id=self.carrying_target_id,
            newly_cleared_cells=set(self.unsynced_cleared_cells),
            all_cleared_count=len(self.local_cleared_cells),
            known_targets={k: v.copy() for k, v in self.known_targets.items()},
            rescued_target_ids=set(self.rescued_target_ids),
            known_closed_doors=set(self.known_closed_doors),
            opened_doors=set(self.opened_doors),
        )

    def apply_coordinator_directive(self, directive: CoordinatorDirective) -> None:
        """Merges the Coordinator's 1-to-1 downlink response into local belief state."""
        self.last_sync_step = self.step_count
        self.steps_since_sync = 0
        self.unsynced_cleared_cells.clear()

        # Merge global cleared cells discovered by teammates
        self.local_cleared_cells.update(directive.global_cleared_cells)

        # Merge door states
        self.opened_doors.update(directive.opened_doors)
        self.known_closed_doors.update(directive.known_closed_doors - self.opened_doors)
        self.known_closed_doors -= self.opened_doors
        if self.invited_door_goal in self.opened_doors:
            self.invited_door_goal = None

        # Merge global target statuses
        for tid, grec in directive.global_targets.items():
            if grec.status == TargetStatus.RESCUED:
                self.rescued_target_ids.add(tid)
                self.known_targets[tid] = grec.copy()
                continue

            if tid not in self.known_targets:
                self.known_targets[tid] = grec.copy()
            else:
                local_rec = self.known_targets[tid]
                if STATUS_PRIORITY[grec.status] >= STATUS_PRIORITY[local_rec.status]:
                    self.known_targets[tid] = grec.copy()

        self.dark_teammates = set(directive.dark_agents)
        self.last_known_peer_positions.update(directive.other_agent_positions)
        self.coordinator_priority_target = directive.priority_target_id
        self.coordinator_priority_door = directive.priority_door_pos
        self.coordinator_waypoint = directive.priority_waypoint
        self.active_sector = directive.recommended_sector

        if directive.mission_complete:
            self.mission_complete_flag = True

    def try_sync_with_coordinator(
        self,
        coordinator: Optional[Coordinator],
        comm_available: bool = True,
    ) -> bool:
        """
        Attempts 1-to-1 communication with the Coordinator.
        Rule: Signal is ALWAYS ON when the agent is outside the maze (`is_outside_maze()`).
        """
        if coordinator is None:
            self.steps_since_sync += 1
            return False

        effective_comm = True if self.is_outside_maze() else bool(comm_available)
        if not effective_comm:
            coordinator.record_comm_failure(self.agent_id, self.step_count)
            self.steps_since_sync += 1
            return False

        report = self.build_report()
        directive = coordinator.sync_with_agent(self.agent_id, report)
        self.apply_coordinator_directive(directive)
        return True

    # ------------------------------------------------------------------
    # 3. Doors ('D'), Ground Markers ('I', 'O'), & Target Extraction
    # ------------------------------------------------------------------
    def _try_cooperative_door_unlock(
        self,
        closed_doors: Optional[Set[Coord]],
        all_agent_positions: Optional[Dict[str, Coord]],
        ground_markers: Optional[Dict[Coord, GroundMarker]],
    ) -> Optional[Coord]:
        """
        Checks if this agent and at least 1 other agent are adjacent to any closed Door 'D'.
        If >= 2 agents are adjacent, the door 'D' is removed/unblocked!
        """
        if not closed_doors or not all_agent_positions:
            return None

        positions = dict(all_agent_positions)
        positions[self.agent_id] = self.position

        for dpos in list(closed_doors):
            if can_unlock_door(dpos, positions):
                closed_doors.discard(dpos)
                self.known_closed_doors.discard(dpos)
                self.opened_doors.add(dpos)
                if self.invited_door_goal == dpos:
                    self.invited_door_goal = None
                if self.coordinator_priority_door == dpos:
                    self.coordinator_priority_door = None
                # Remove any 'I' markers that were pointing to this now-open door
                if ground_markers is not None:
                    for mpos in [
                        k
                        for k, m in ground_markers.items()
                        if m.marker_type == MarkerType.INVITE_DOOR
                        and m.target_door_pos == dpos
                    ]:
                        del ground_markers[mpos]
                return dpos
        return None

    def _maybe_drop_ground_marker(
        self,
        synced_with_coordinator: bool,
        closed_doors: Optional[Set[Coord]],
        ground_markers: Optional[Dict[Coord, GroundMarker]],
        active_targets: Optional[Dict[str, Coord]],
    ) -> Optional[GroundMarker]:
        """
        When signal is down (`not synced_with_coordinator` and inside the maze):
        1. Drops 'I' (Invite/Assist) if we know of a closed door 'D' that needs a 2nd agent,
           placing 'I' at corridor junctions or along our path so teammates spot it.
        2. Drops 'O' (Dead End) if we are exiting a searched dead-end branch so teammates
           know the road ahead is a dead end.
        """
        if synced_with_coordinator or self.is_outside_maze() or ground_markers is None:
            return None

        effective_doors = (
            closed_doors if closed_doors is not None else self.known_closed_doors
        )

        # Priority A: Drop 'I' marker to recruit a 2nd agent for a closed door 'D'
        active_closed_doors = self.known_closed_doors & effective_doors
        if active_closed_doors:
            target_door = min(
                active_closed_doors,
                key=lambda d: abs(d[0] - self.position[0]) + abs(d[1] - self.position[1]),
            )
            neighbors = get_valid_neighbors(
                self.grid, self.position, closed_doors=effective_doors
            )
            is_junction = len(neighbors) >= 3
            is_near_door = (
                abs(self.position[0] - target_door[0])
                + abs(self.position[1] - target_door[1])
                <= 2
            )
            if (is_junction or is_near_door) and self.position not in ground_markers:
                marker = GroundMarker(
                    marker_type=MarkerType.INVITE_DOOR,
                    position=self.position,
                    author_agent=self.agent_id,
                    target_door_pos=target_door,
                    created_step=self.step_count,
                )
                ground_markers[self.position] = marker
                self.placed_invite_doors.add(target_door)
                return marker

        # Priority B: Drop 'O' marker at the neck of a searched dead-end corridor
        target_coords = set(active_targets.values()) if active_targets else set()
        dead_end_info = detect_dead_end_branch(
            grid=self.grid,
            current_pos=self.position,
            cleared_cells=self.local_cleared_cells,
            entrances=self.entrances,
            closed_doors=effective_doors,
            active_target_positions=target_coords,
        )
        if dead_end_info is not None:
            neck_pos, branch_cells = dead_end_info
            if (
                neck_pos not in self.placed_dead_end_markers
                and neck_pos not in ground_markers
            ):
                marker = GroundMarker(
                    marker_type=MarkerType.DEAD_END,
                    position=neck_pos,
                    author_agent=self.agent_id,
                    dead_end_cells=branch_cells,
                    created_step=self.step_count,
                )
                ground_markers[neck_pos] = marker
                self.placed_dead_end_markers.add(neck_pos)
                return marker

        return None

    def _check_pickup_and_dropoff(
        self, active_targets: Optional[Dict[str, Coord]] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Handles immediate target pickup inside the maze and rescue dropoff at the entrance.
        Agents can go in and out of the maze multiple times.
        """
        picked_up: Optional[str] = None
        rescued: Optional[str] = None

        # Dropoff at entrance (outside maze) if currently carrying a target
        if self.carrying_target_id is not None and self.is_outside_maze():
            tid = self.carrying_target_id
            self.carrying_target_id = None
            self.rescued_target_ids.add(tid)
            self.trips_completed += 1
            if tid in self.known_targets:
                self.known_targets[tid].status = TargetStatus.RESCUED
                self.known_targets[tid].updated_step = self.step_count
            rescued = tid

        # Pickup if standing on a discovered target and not carrying anyone
        if self.carrying_target_id is None:
            for tid, rec in self.known_targets.items():
                if rec.status == TargetStatus.DISCOVERED and rec.position == self.position:
                    if active_targets is None or tid in active_targets:
                        self.carrying_target_id = tid
                        rec.status = TargetStatus.CARRYING
                        rec.assigned_agent = self.agent_id
                        rec.updated_step = self.step_count
                        picked_up = tid
                        break

        return picked_up, rescued

    def _select_unclaimed_target_to_rescue(
        self, closed_doors: Set[Coord]
    ) -> Optional[TargetRecord]:
        """
        Checks if there is a known DISCOVERED target reachable without crossing closed doors.
        """
        dist_map = bfs_distance_map(self.grid, [self.position], closed_doors=closed_doors)
        candidates: List[TargetRecord] = []
        for tid, rec in self.known_targets.items():
            if tid in self.rescued_target_ids or rec.status != TargetStatus.DISCOVERED:
                continue
            if rec.position not in dist_map:
                continue
            if rec.assigned_agent not in (None, self.agent_id):
                if rec.assigned_agent not in self.dark_teammates:
                    continue
            candidates.append(rec)

        if not candidates:
            return None

        for rec in candidates:
            if rec.target_id == self.coordinator_priority_target:
                return rec

        return min(candidates, key=lambda r: dist_map[r.position])

    def _select_door_to_open(self, closed_doors: Set[Coord]) -> Optional[Coord]:
        """
        Determines if this agent should move to a closed Door 'D' to unlock it with a partner:
        1. If the agent saw an 'I' (Invite) ground marker pointing to a closed door.
        2. If the Coordinator assigned `coordinator_priority_door`.
        3. If the agent discovered a closed door 'D' and is recruiting/waiting for a partner.
        """
        if self.invited_door_goal and self.invited_door_goal in closed_doors:
            return self.invited_door_goal
        if self.coordinator_priority_door and self.coordinator_priority_door in closed_doors:
            return self.coordinator_priority_door
        active_known = self.known_closed_doors & closed_doors
        if active_known:
            return min(
                active_known,
                key=lambda d: abs(d[0] - self.position[0]) + abs(d[1] - self.position[1]),
            )
        return None

    def _is_locally_mission_complete(self, closed_doors: Set[Coord]) -> bool:
        if self.mission_complete_flag:
            return True
        if len(self.rescued_target_ids) >= self.total_targets_goal:
            return True
        if (
            not (self.known_closed_doors & closed_doors)
            and self.all_walkable.issubset(self.local_cleared_cells)
            and self.carrying_target_id is None
            and all(
                r.status == TargetStatus.RESCUED for r in self.known_targets.values()
            )
        ):
            return True
        return False

    def get_uncleared_in_sector(
        self, sector_id: int, closed_doors: Optional[Set[Coord]] = None
    ) -> Set[Coord]:
        doors = closed_doors or set()
        return (self.sectors.get(sector_id, set()) - self.local_cleared_cells) - doors

    def get_all_uncleared(
        self, closed_doors: Optional[Set[Coord]] = None
    ) -> Set[Coord]:
        doors = closed_doors or set()
        return (self.all_walkable - self.local_cleared_cells) - doors

    @abstractmethod
    def get_spillover_sector_order(self) -> List[int]:
        """Returns the deterministic sector spillover sequence for this agent."""

    @abstractmethod
    def select_exploration_goal(
        self, candidate_cells: Set[Coord], closed_doors: Optional[Set[Coord]] = None
    ) -> Tuple[Coord, str]:
        """
        Agent-specific independent heuristic to pick the best next target coordinate
        from `candidate_cells` and a human-readable reasoning string.
        """

    def should_initiate_relay_sync(self) -> bool:
        """
        Returns True if the agent should temporarily exit the maze to the entrance
        (where signal is always ON) to sync critical discoveries with the Coordinator.
        """
        return False

    # ------------------------------------------------------------------
    # 4. Main Step Execution
    # ------------------------------------------------------------------
    def step(
        self,
        coordinator: Optional[Coordinator] = None,
        comm_available: Any = True,
        active_targets: Optional[Dict[str, Coord]] = None,
        all_agent_positions: Optional[Dict[str, Coord]] = None,
        closed_doors: Optional[Set[Coord]] = None,
        ground_markers: Optional[Dict[Coord, GroundMarker]] = None,
        precomputed_observation: Optional[VisibleObservation] = None,
    ) -> StepDecision:
        """
        Executes one complete perception -> sync -> marker/door check -> decision -> move cycle.
        """
        self.step_count += 1
        effective_doors = (
            closed_doors if closed_doors is not None else self.known_closed_doors
        )

        def _is_comm_ok(pos: Coord) -> bool:
            # Rule 2: Signal is ALWAYS ON when agent is outside the maze (at entrance)
            if self.is_outside_maze(pos):
                return True
            if callable(comm_available):
                return bool(comm_available(pos))
            return bool(comm_available)

        # 1. Perceive 5x5 environment (walls, doors 'D', markers 'I' & 'O', targets, peers)
        obs = self.observe(
            active_targets=active_targets,
            all_agent_positions=all_agent_positions,
            closed_doors=effective_doors,
            ground_markers=ground_markers,
            precomputed_observation=precomputed_observation,
        )

        # 2. Check if 2 agents are currently adjacent to a closed Door 'D' -> unlock it!
        unlocked_1 = self._try_cooperative_door_unlock(
            closed_doors, all_agent_positions, ground_markers
        )

        # 3. Check immediate pickup or entrance dropoff before syncing
        picked_up_1, rescued_1 = self._check_pickup_and_dropoff(active_targets)

        # 4. Attempt 1-to-1 communication with Coordinator (always succeeds at entrance)
        synced = self.try_sync_with_coordinator(
            coordinator=coordinator, comm_available=_is_comm_ok(self.position)
        )

        # 5. Determine operational mode and goal coordinate
        goal_pos: Coord = self.position
        reasoning: str = ""
        door_goal = self._select_door_to_open(effective_doors)

        if self._is_locally_mission_complete(effective_doors):
            self.mode = AgentMode.MISSION_COMPLETE
            goal_pos = self._nearest_entrance(effective_doors)
            reasoning = "All targets rescued; exiting maze to base perimeter."

        elif self.carrying_target_id is not None:
            self.mode = AgentMode.EXTRACT_TARGET
            goal_pos = self._nearest_entrance(effective_doors)
            reasoning = (
                f"Carrying target {self.carrying_target_id} out of maze to {goal_pos}."
            )

        else:
            # Check if there is a reachable discovered target to pick up
            target_to_rescue = self._select_unclaimed_target_to_rescue(effective_doors)
            if target_to_rescue is not None:
                self.mode = AgentMode.APPROACH_TARGET
                goal_pos = target_to_rescue.position
                reasoning = (
                    f"Moving to rescue target {target_to_rescue.target_id} at {goal_pos}."
                )

            elif door_goal is not None:
                # Coordinate 2-agent door unlocking ('D')
                wait_cnt = self.door_wait_steps.get(door_goal, 0)
                dist_to_door = abs(self.position[0] - door_goal[0]) + abs(
                    self.position[1] - door_goal[1]
                )
                if dist_to_door <= 1:
                    self.door_wait_steps[door_goal] = wait_cnt + 1
                # If we waited at a door offline for > 12 steps and no partner came yet,
                # walk toward corridor junction / entrance to drop 'I' markers or sync at entrance!
                if not synced and wait_cnt >= 12 and not self.is_outside_maze():
                    self.mode = AgentMode.RELAY_SYNC
                    goal_pos = self._nearest_entrance(effective_doors)
                    if self.is_outside_maze(goal_pos):
                        self.door_wait_steps[door_goal] = 0
                    reasoning = (
                        f"Signal down at Door 'D' {door_goal}; dropping 'I' markers / "
                        f"heading to entrance to recruit 2nd agent."
                    )
                else:
                    self.mode = (
                        AgentMode.ASSIST_OPEN_DOOR
                        if (
                            self.invited_door_goal == door_goal
                            or self.coordinator_priority_door == door_goal
                        )
                        else AgentMode.WAIT_FOR_DOOR_PARTNER
                    )
                    stand_cells = get_door_stand_positions(
                        self.grid, door_goal, closed_doors=effective_doors
                    )
                    goal_pos = stand_cells[0] if stand_cells else door_goal
                    reasoning = (
                        f"Moving/waiting at Door 'D' {door_goal} (requires 2 agents to unblock)."
                    )

            elif not synced and self.should_initiate_relay_sync():
                self.mode = AgentMode.RELAY_SYNC
                goal_pos = self._nearest_entrance(effective_doors)
                reasoning = (
                    f"Signal down for {self.steps_since_sync} steps; exiting to entrance "
                    f"(where signal is always ON) to sync with Coordinator."
                )

            else:
                goal_pos, reasoning = self._decide_sector_exploration_goal(effective_doors)

        self.current_goal = goal_pos

        # 6. If signal is down inside the maze, potentially drop 'I' or 'O' marker at current cell
        dropped_marker = self._maybe_drop_ground_marker(
            synced_with_coordinator=synced,
            closed_doors=closed_doors,
            ground_markers=ground_markers,
            active_targets=active_targets,
        )

        # 7. Compute next step along shortest walkable path
        next_pos = self._compute_next_step_towards(
            goal_pos, obs.visible_agents, effective_doors
        )
        move_delta = (next_pos[0] - self.position[0], next_pos[1] - self.position[1])
        prev_pos = self.position
        self.position = next_pos
        self.recent_positions.append(next_pos)
        if len(self.recent_positions) > 8:
            self.recent_positions.pop(0)

        # 8. Check post-move cooperative door unlock and target pickup/dropoff
        unlocked_2 = self._try_cooperative_door_unlock(
            closed_doors, all_agent_positions, ground_markers
        )
        picked_up_2, rescued_2 = self._check_pickup_and_dropoff(active_targets)

        unlocked_door = unlocked_1 or unlocked_2
        picked_up = picked_up_1 or picked_up_2
        rescued = rescued_1 or rescued_2

        # 9. If we stepped onto the outside entrance (or unlocked/rescued while connected), sync!
        if (
            (rescued or unlocked_door or not synced or self.is_outside_maze(self.position))
            and _is_comm_ok(self.position)
            and coordinator is not None
        ):
            self.try_sync_with_coordinator(coordinator=coordinator, comm_available=True)
            synced = True

        return StepDecision(
            agent_id=self.agent_id,
            current_pos=prev_pos,
            next_pos=self.position,
            move_delta=move_delta,
            mode=self.mode,
            synced_with_coordinator=synced,
            is_outside_maze=self.is_outside_maze(self.position),
            carrying_target_id=self.carrying_target_id,
            picked_up_target_id=picked_up,
            rescued_target_id=rescued,
            unlocked_door_pos=unlocked_door,
            dropped_marker=dropped_marker,
            target_goal_pos=goal_pos,
            visible_cells=set(obs.visible_cells.keys()),
            reasoning=reasoning,
        )

    def _decide_sector_exploration_goal(
        self, closed_doors: Set[Coord]
    ) -> Tuple[Coord, str]:
        dist_from_me = bfs_distance_map(
            self.grid, [self.position], closed_doors=closed_doors
        )
        reachable_cells = set(dist_from_me.keys())

        primary_uncleared = (
            self.get_uncleared_in_sector(self.primary_sector, closed_doors)
            & reachable_cells
        )
        if primary_uncleared:
            self.mode = AgentMode.SEARCH_PRIMARY
            self.active_sector = self.primary_sector
            return self.select_exploration_goal(primary_uncleared, closed_doors)

        # Primary sector cleared -> Autonomous Spillover to support teammates!
        self.mode = AgentMode.SEARCH_SPILLOVER
        for sec_id in self.get_spillover_sector_order():
            sec_uncleared = (
                self.get_uncleared_in_sector(sec_id, closed_doors) & reachable_cells
            )
            if sec_uncleared:
                self.active_sector = sec_id
                goal, reason = self.select_exploration_goal(sec_uncleared, closed_doors)
                return goal, f"[Spillover->Sector {sec_id}] {reason}"

        all_rem = self.get_all_uncleared(closed_doors) & reachable_cells
        if all_rem:
            return self.select_exploration_goal(all_rem, closed_doors)

        self.mode = AgentMode.MISSION_COMPLETE
        return self._nearest_entrance(closed_doors), "All reachable sectors cleared; returning to entrance."

    def _nearest_entrance(self, closed_doors: Optional[Set[Coord]] = None) -> Coord:
        return min(
            self.entrances,
            key=lambda e: len(
                bfs_shortest_path(self.grid, self.position, e, closed_doors=closed_doors)
            )
            or 999999,
        )

    def _compute_next_step_towards(
        self,
        goal: Coord,
        visible_agents: Dict[str, Coord],
        closed_doors: Optional[Set[Coord]] = None,
    ) -> Coord:
        if self.position == goal:
            return self.position

        avoid: Set[Coord] = set()
        for other_id, other_pos in visible_agents.items():
            if other_id < self.agent_id and other_pos != goal:
                avoid.add(other_pos)

        path = bfs_shortest_path(
            self.grid,
            self.position,
            goal,
            avoid_positions=avoid,
            closed_doors=closed_doors,
        )
        if len(path) >= 2:
            return path[1]
        return self.position


# ======================================================================
# Specialized Autonomous Agents: A1, A2, A3
# ======================================================================


class AgentA1(BaseAgent):
    """
    Agent A1 — "Deep-Infiltration Vanguard" (Primary Sector 1)
    Prioritizes deep interior frontiers first so deep rooms are searched even during sudden blackouts.
    """

    def __init__(self, initial_pos: Coord, briefing: Dict[str, Any]) -> None:
        super().__init__(agent_id="A1", initial_pos=initial_pos, briefing=briefing)

    def get_spillover_sector_order(self) -> List[int]:
        order = [2, 3]
        if "A3" in self.dark_teammates and "A2" not in self.dark_teammates:
            order = [3, 2]
        return order

    def select_exploration_goal(
        self, candidate_cells: Set[Coord], closed_doors: Optional[Set[Coord]] = None
    ) -> Tuple[Coord, str]:
        dist_from_me = bfs_distance_map(
            self.grid, [self.position], closed_doors=closed_doors
        )
        reachable_candidates = [c for c in candidate_cells if c in dist_from_me]
        if not reachable_candidates:
            fallback = next(iter(candidate_cells))
            return fallback, "Fallback unreachable candidate."

        def _vanguard_score(cell: Coord) -> float:
            d_travel = dist_from_me[cell]
            d_depth = self.entrance_dist_map.get(cell, 0)
            peer_penalty = sum(
                8.0
                for ppos in self.last_known_peer_positions.values()
                if abs(cell[0] - ppos[0]) + abs(cell[1] - ppos[1]) <= 3
            )
            return d_travel - (0.35 * d_depth) + peer_penalty

        best_cell = min(reachable_candidates, key=_vanguard_score)
        return (
            best_cell,
            f"A1 Deep-Vanguard selecting deep frontier {best_cell} "
            f"(depth={self.entrance_dist_map.get(best_cell, 0)}).",
        )


class AgentA2(BaseAgent):
    """
    Agent A2 — "High-Visibility Recon Scout" (Primary Sector 2)
    Evaluates candidate frontiers by expected 5x5 line-of-sight Information Gain per step.
    """

    def __init__(self, initial_pos: Coord, briefing: Dict[str, Any]) -> None:
        super().__init__(agent_id="A2", initial_pos=initial_pos, briefing=briefing)
        self._sight_mask_cache: Dict[Coord, Set[Coord]] = {}

    def _get_visible_walkable_from(self, cell: Coord) -> Set[Coord]:
        if cell not in self._sight_mask_cache:
            obs = compute_5x5_sight(self.grid, cell, agent_id=self.agent_id)
            self._sight_mask_cache[cell] = {
                pos for pos, val in obs.visible_cells.items() if val == 0
            }
        return self._sight_mask_cache[cell]

    def get_spillover_sector_order(self) -> List[int]:
        order = [3, 1]
        if "A1" in self.dark_teammates and "A3" not in self.dark_teammates:
            order = [1, 3]
        return order

    def select_exploration_goal(
        self, candidate_cells: Set[Coord], closed_doors: Optional[Set[Coord]] = None
    ) -> Tuple[Coord, str]:
        dist_from_me = bfs_distance_map(
            self.grid, [self.position], closed_doors=closed_doors
        )
        reachable_candidates = [c for c in candidate_cells if c in dist_from_me]
        if not reachable_candidates:
            fallback = next(iter(candidate_cells))
            return fallback, "Fallback unreachable candidate."

        reachable_candidates.sort(key=lambda c: dist_from_me[c])
        shortlist = reachable_candidates[:35]

        def _info_gain_utility(cell: Coord) -> float:
            d_travel = max(1, dist_from_me[cell])
            newly_revealed = len(
                self._get_visible_walkable_from(cell) - self.local_cleared_cells
            )
            peer_penalty = sum(
                6.0
                for ppos in self.last_known_peer_positions.values()
                if abs(cell[0] - ppos[0]) + abs(cell[1] - ppos[1]) <= 3
            )
            return (d_travel - 0.65 * newly_revealed) + peer_penalty

        best_cell = min(shortlist, key=_info_gain_utility)
        gain = len(self._get_visible_walkable_from(best_cell) - self.local_cleared_cells)
        return (
            best_cell,
            f"A2 Recon-Scout selecting high-visibility vantage {best_cell} "
            f"(expected 5x5 gain={gain} tiles).",
        )


class AgentA3(BaseAgent):
    """
    Agent A3 — "Adaptive Support & Comms-Relay Specialist" (Primary Sector 3)
    Uses wavefront sweeping and periodically exits to the entrance (where signal is always ON)
    during prolonged interior blackouts to sync with the Coordinator and assist teammates.
    """

    def __init__(
        self,
        initial_pos: Coord,
        briefing: Dict[str, Any],
        relay_step_threshold: int = 24,
        relay_min_new_tiles: int = 20,
    ) -> None:
        super().__init__(agent_id="A3", initial_pos=initial_pos, briefing=briefing)
        self.relay_step_threshold = relay_step_threshold
        self.relay_min_new_tiles = relay_min_new_tiles

    def should_initiate_relay_sync(self) -> bool:
        if self.is_outside_maze():
            return False
        has_unreported_target_or_door = any(
            r.status == TargetStatus.DISCOVERED for r in self.known_targets.values()
        ) or bool(self.known_closed_doors)
        if (
            self.steps_since_sync >= self.relay_step_threshold
            and (
                len(self.unsynced_cleared_cells) >= self.relay_min_new_tiles
                or has_unreported_target_or_door
            )
            and len(self.get_all_uncleared()) > 0
        ):
            return True
        return False

    def get_spillover_sector_order(self) -> List[int]:
        order = [1, 2]
        if "A2" in self.dark_teammates and "A1" not in self.dark_teammates:
            order = [2, 1]
        return order

    def select_exploration_goal(
        self, candidate_cells: Set[Coord], closed_doors: Optional[Set[Coord]] = None
    ) -> Tuple[Coord, str]:
        dist_from_me = bfs_distance_map(
            self.grid, [self.position], closed_doors=closed_doors
        )
        reachable_candidates = [c for c in candidate_cells if c in dist_from_me]
        if not reachable_candidates:
            fallback = next(iter(candidate_cells))
            return fallback, "Fallback unreachable candidate."

        def _support_wavefront_score(cell: Coord) -> float:
            d_travel = dist_from_me[cell]
            d_entrance = self.entrance_dist_map.get(cell, 0)
            peer_penalty = sum(
                8.0
                for ppos in self.last_known_peer_positions.values()
                if abs(cell[0] - ppos[0]) + abs(cell[1] - ppos[1]) <= 3
            )
            return d_travel + (0.15 * d_entrance) + peer_penalty

        best_cell = min(reachable_candidates, key=_support_wavefront_score)
        return (
            best_cell,
            f"A3 Support-Specialist sweeping wavefront frontier {best_cell}.",
        )


def create_sar_team(
    floor_plan: Sequence[Sequence[Any]],
    total_targets_goal: int,
    entrances: Optional[Sequence[Coord]] = None,
) -> Tuple[Coordinator, Dict[str, BaseAgent]]:
    """
    Factory helper that initializes the outside Coordinator and the 3 specialized
    autonomous field agents (A1, A2, A3) from a floor plan array.
    """
    _, parsed_agents, _ = parse_floor_plan(floor_plan)
    coordinator = Coordinator(
        floor_plan=floor_plan,
        total_targets_goal=total_targets_goal,
        entrances=entrances,
    )

    default_entrances = coordinator.entrances
    pos_a1 = parsed_agents.get("A1", default_entrances[0])
    pos_a2 = parsed_agents.get(
        "A2", default_entrances[min(1, len(default_entrances) - 1)]
    )
    pos_a3 = parsed_agents.get(
        "A3", default_entrances[min(2, len(default_entrances) - 1)]
    )

    agents: Dict[str, BaseAgent] = {
        "A1": AgentA1(
            initial_pos=pos_a1,
            briefing=coordinator.get_initial_briefing("A1"),
        ),
        "A2": AgentA2(
            initial_pos=pos_a2,
            briefing=coordinator.get_initial_briefing("A2"),
        ),
        "A3": AgentA3(
            initial_pos=pos_a3,
            briefing=coordinator.get_initial_briefing("A3"),
        ),
    }

    return coordinator, agents
