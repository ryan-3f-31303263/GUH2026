"""
Autonomous Field Agents (A1, A2, A3) for the Neptune North Search & Rescue Challenge.

Design Principles:
1. Working towards a common goal:
   All 3 agents collaborate to locate and rescue all N targets in the known floor plan
   while minimizing redundant coverage.
2. Independent decision-making:
   Each agent (A1, A2, A3) runs its own autonomous decision loop and specialized
   frontier-selection policy to support the group, operating seamlessly even when
   1-to-1 communication with the Coordinator fails completely.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from .coordinator import Coordinator, STATUS_PRIORITY
from .models import (
    AgentMode,
    AgentReport,
    Coord,
    CoordinatorDirective,
    StepDecision,
    TargetRecord,
    TargetStatus,
    VisibleObservation,
)
from .vision_and_grid import (
    bfs_distance_map,
    bfs_shortest_path,
    compute_5x5_sight,
    get_valid_neighbors,
)


class BaseAgent(ABC):
    """
    Base autonomous Search & Rescue agent with:
    - Centred 5x5 line-of-sight perception (blocked by walls)
    - Local belief state (cleared cells, target statuses, peer sightings)
    - 1-to-1 communication client to the outside Coordinator
    - Independent target extraction and sector spillover state machine
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

        # Local belief state (maintained independently when comms fail)
        self.step_count: int = 0
        self.mode: AgentMode = AgentMode.SEARCH_PRIMARY
        self.local_cleared_cells: Set[Coord] = set()
        self.unsynced_cleared_cells: Set[Coord] = set()
        self.known_targets: Dict[str, TargetRecord] = {}
        self.rescued_target_ids: Set[str] = set()
        self.carrying_target_id: Optional[str] = None

        # Comms & coordination tracking
        self.last_sync_step: int = 0
        self.steps_since_sync: int = 0
        self.dark_teammates: Set[str] = set()
        self.last_known_peer_positions: Dict[str, Coord] = {}
        self.coordinator_priority_target: Optional[str] = None
        self.coordinator_waypoint: Optional[Coord] = None
        self.mission_complete_flag: bool = False

        # Navigation history to prevent oscillation in narrow corridors
        self.current_goal: Optional[Coord] = None
        self.recent_positions: List[Coord] = [initial_pos]

    # ------------------------------------------------------------------
    # 1. Perception: Centred 5x5 Sight with Wall Occlusion
    # ------------------------------------------------------------------
    def observe(
        self,
        active_targets: Optional[Dict[str, Coord]] = None,
        all_agent_positions: Optional[Dict[str, Coord]] = None,
        precomputed_observation: Optional[VisibleObservation] = None,
    ) -> VisibleObservation:
        """
        Perceives the centred 5x5 area around `self.position` (blocked by walls)
        and updates local belief state.
        """
        obs = precomputed_observation or compute_5x5_sight(
            grid=self.grid,
            center=self.position,
            agent_id=self.agent_id,
            active_targets=active_targets,
            all_agent_positions=all_agent_positions,
        )

        # Mark all visible walkable (0) cells as cleared
        for cell, tile_val in obs.visible_cells.items():
            if tile_val == 0:
                if cell not in self.local_cleared_cells:
                    self.local_cleared_cells.add(cell)
                    self.unsynced_cleared_cells.add(cell)

        # Record any targets visible in 5x5 sight
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

        # Update local knowledge of nearby peers seen in 5x5 window
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
        )

    def apply_coordinator_directive(self, directive: CoordinatorDirective) -> None:
        """Merges the Coordinator's 1-to-1 downlink response into local belief state."""
        self.last_sync_step = self.step_count
        self.steps_since_sync = 0
        self.unsynced_cleared_cells.clear()

        # Merge global cleared cells discovered by teammates
        self.local_cleared_cells.update(directive.global_cleared_cells)

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
        Returns True if sync succeeded, False if comms failed / unavailable.
        """
        if coordinator is None:
            self.steps_since_sync += 1
            return False

        if not comm_available:
            coordinator.record_comm_failure(self.agent_id, self.step_count)
            self.steps_since_sync += 1
            return False

        report = self.build_report()
        directive = coordinator.sync_with_agent(self.agent_id, report)
        self.apply_coordinator_directive(directive)
        return True

    # ------------------------------------------------------------------
    # 3. Independent Decision-Making & Target Handling
    # ------------------------------------------------------------------
    def _check_pickup_and_dropoff(
        self, active_targets: Optional[Dict[str, Coord]] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Handles immediate target pickup at current position and rescue dropoff at entrance.
        Returns (picked_up_target_id, rescued_target_id).
        """
        picked_up: Optional[str] = None
        rescued: Optional[str] = None

        # Dropoff at entrance if currently carrying a target
        if self.carrying_target_id is not None and self.position in self.entrances:
            tid = self.carrying_target_id
            self.carrying_target_id = None
            self.rescued_target_ids.add(tid)
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

    def _select_unclaimed_target_to_rescue(self) -> Optional[TargetRecord]:
        """
        Checks if there is a known DISCOVERED target that this agent should go pick up.
        Avoids targets already being carried or closer to another visible peer.
        """
        candidates: List[TargetRecord] = []
        for tid, rec in self.known_targets.items():
            if tid in self.rescued_target_ids or rec.status != TargetStatus.DISCOVERED:
                continue
            if rec.assigned_agent not in (None, self.agent_id):
                # Only override another agent's assignment if that agent is known to be dark
                if rec.assigned_agent not in self.dark_teammates:
                    continue
            candidates.append(rec)

        if not candidates:
            return None

        # Prioritize Coordinator's explicit priority target if valid
        for rec in candidates:
            if rec.target_id == self.coordinator_priority_target:
                return rec

        # Otherwise choose the closest discovered target by shortest path
        return min(
            candidates,
            key=lambda r: abs(r.position[0] - self.position[0])
            + abs(r.position[1] - self.position[1]),
        )

    def _is_locally_mission_complete(self) -> bool:
        if self.mission_complete_flag:
            return True
        if len(self.rescued_target_ids) >= self.total_targets_goal:
            return True
        # All walkable cells cleared and no unrescued targets remain
        if (
            self.all_walkable.issubset(self.local_cleared_cells)
            and self.carrying_target_id is None
            and all(
                r.status == TargetStatus.RESCUED for r in self.known_targets.values()
            )
        ):
            return True
        return False

    def get_uncleared_in_sector(self, sector_id: int) -> Set[Coord]:
        return self.sectors.get(sector_id, set()) - self.local_cleared_cells

    def get_all_uncleared(self) -> Set[Coord]:
        return self.all_walkable - self.local_cleared_cells

    @abstractmethod
    def get_spillover_sector_order(self) -> List[int]:
        """Returns the deterministic sector spillover sequence for this agent."""

    @abstractmethod
    def select_exploration_goal(self, candidate_cells: Set[Coord]) -> Tuple[Coord, str]:
        """
        Agent-specific independent heuristic to pick the best next target coordinate
        from `candidate_cells` and a human-readable reasoning string.
        """

    def should_initiate_relay_sync(self) -> bool:
        """
        Hook for agents that proactively ferry critical intelligence toward the entrance
        when communications have been severed for an extended period.
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
        precomputed_observation: Optional[VisibleObservation] = None,
    ) -> StepDecision:
        """
        Executes one complete perception -> sync -> independent decision -> movement cycle.
        `comm_available` may be a bool or a callable `fn(pos: Coord) -> bool`.
        """
        self.step_count += 1

        def _is_comm_ok(pos: Coord) -> bool:
            if callable(comm_available):
                return bool(comm_available(pos))
            return bool(comm_available)

        # 1. Perceive 5x5 environment first
        obs = self.observe(
            active_targets=active_targets,
            all_agent_positions=all_agent_positions,
            precomputed_observation=precomputed_observation,
        )

        # 2. Check immediate pickup or entrance dropoff before syncing
        picked_up_1, rescued_1 = self._check_pickup_and_dropoff(active_targets)

        # 3. Attempt 1-to-1 communication with Coordinator (if available at current pos)
        synced = self.try_sync_with_coordinator(
            coordinator=coordinator, comm_available=_is_comm_ok(self.position)
        )

        # 4. Determine high-level operational mode and goal coordinate
        goal_pos: Coord = self.position
        reasoning: str = ""

        if self._is_locally_mission_complete():
            self.mode = AgentMode.MISSION_COMPLETE
            goal_pos = self._nearest_entrance()
            reasoning = "All targets rescued / map cleared; returning to base perimeter."

        elif self.carrying_target_id is not None:
            self.mode = AgentMode.EXTRACT_TARGET
            goal_pos = self._nearest_entrance()
            reasoning = (
                f"Escorting target {self.carrying_target_id} to entrance {goal_pos}."
            )

        else:
            # Check if there is a known discovered target to pick up
            target_to_rescue = self._select_unclaimed_target_to_rescue()
            if target_to_rescue is not None:
                self.mode = AgentMode.APPROACH_TARGET
                goal_pos = target_to_rescue.position
                reasoning = (
                    f"Moving to rescue spotted target {target_to_rescue.target_id} "
                    f"at {goal_pos}."
                )
            elif not synced and self.should_initiate_relay_sync():
                self.mode = AgentMode.RELAY_SYNC
                goal_pos = self._nearest_entrance()
                reasoning = (
                    f"Comms blackout for {self.steps_since_sync} steps with "
                    f"{len(self.unsynced_cleared_cells)} new tiles; relaying toward entrance."
                )
            else:
                # Independent Frontier / Sector Exploration
                goal_pos, reasoning = self._decide_sector_exploration_goal()

        self.current_goal = goal_pos

        # 5. Compute next step along shortest walkable path (with peer collision avoidance)
        next_pos = self._compute_next_step_towards(goal_pos, obs.visible_agents)
        move_delta = (next_pos[0] - self.position[0], next_pos[1] - self.position[1])
        prev_pos = self.position
        self.position = next_pos
        self.recent_positions.append(next_pos)
        if len(self.recent_positions) > 8:
            self.recent_positions.pop(0)

        # 6. Check if arriving at `next_pos` immediately picks up or rescues a target
        picked_up_2, rescued_2 = self._check_pickup_and_dropoff(active_targets)
        picked_up = picked_up_1 or picked_up_2
        rescued = rescued_1 or rescued_2

        # If we just arrived in a comms zone or rescued a target at the entrance, sync immediately
        if (rescued or not synced) and _is_comm_ok(self.position) and coordinator is not None:
            self.try_sync_with_coordinator(coordinator=coordinator, comm_available=True)
            synced = True

        return StepDecision(
            agent_id=self.agent_id,
            current_pos=prev_pos,
            next_pos=self.position,
            move_delta=move_delta,
            mode=self.mode,
            synced_with_coordinator=synced,
            carrying_target_id=self.carrying_target_id,
            picked_up_target_id=picked_up,
            rescued_target_id=rescued,
            target_goal_pos=goal_pos,
            visible_cells=set(obs.visible_cells.keys()),
            reasoning=reasoning,
        )

    def _decide_sector_exploration_goal(self) -> Tuple[Coord, str]:
        """
        Determines which sector has uncleared cells (primary first, then spillover order)
        and delegates to the agent's specialized `select_exploration_goal`.
        """
        primary_uncleared = self.get_uncleared_in_sector(self.primary_sector)
        if primary_uncleared:
            self.mode = AgentMode.SEARCH_PRIMARY
            self.active_sector = self.primary_sector
            return self.select_exploration_goal(primary_uncleared)

        # Primary sector is 100% cleared -> Autonomous Spillover to support teammates!
        self.mode = AgentMode.SEARCH_SPILLOVER
        for sec_id in self.get_spillover_sector_order():
            sec_uncleared = self.get_uncleared_in_sector(sec_id)
            if sec_uncleared:
                self.active_sector = sec_id
                goal, reason = self.select_exploration_goal(sec_uncleared)
                return goal, f"[Spillover->Sector {sec_id}] {reason}"

        # Fallback: any remaining uncleared walkable cell anywhere
        all_rem = self.get_all_uncleared()
        if all_rem:
            return self.select_exploration_goal(all_rem)

        self.mode = AgentMode.MISSION_COMPLETE
        return self._nearest_entrance(), "All sectors cleared; returning to entrance."

    def _nearest_entrance(self) -> Coord:
        return min(
            self.entrances,
            key=lambda e: len(bfs_shortest_path(self.grid, self.position, e)) or 999999,
        )

    def _compute_next_step_towards(
        self, goal: Coord, visible_agents: Dict[str, Coord]
    ) -> Coord:
        if self.position == goal:
            return self.position

        # Yield to higher-priority agents (A1 > A2 > A3) to prevent hallway deadlocks
        avoid: Set[Coord] = set()
        for other_id, other_pos in visible_agents.items():
            if other_id < self.agent_id:
                avoid.add(other_pos)

        path = bfs_shortest_path(
            self.grid, self.position, goal, avoid_positions=avoid
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

    Independent Decision-Making Logic:
    - Prioritizes deep/distant frontiers inside its sector first (balances travel distance
      with high topological depth from the entrance).
    - Rationale: If communication fails deep inside the structure, A1 ensures the hardest-to-reach
      interior rooms are searched first and sweeps naturally back toward the exit, preventing
      any deep-trapped survivors from being missed.
    """

    def __init__(self, initial_pos: Coord, briefing: Dict[str, Any]) -> None:
        super().__init__(agent_id="A1", initial_pos=initial_pos, briefing=briefing)

    def get_spillover_sector_order(self) -> List[int]:
        # If coordinator noted a dark agent, help their sector first; otherwise 1 -> 2 -> 3
        order = [2, 3]
        if "A3" in self.dark_teammates and "A2" not in self.dark_teammates:
            order = [3, 2]
        return order

    def select_exploration_goal(self, candidate_cells: Set[Coord]) -> Tuple[Coord, str]:
        dist_from_me = bfs_distance_map(self.grid, [self.position])
        reachable_candidates = [c for c in candidate_cells if c in dist_from_me]
        if not reachable_candidates:
            fallback = next(iter(candidate_cells))
            return fallback, "Fallback unreachable candidate."

        # Score: minimize travel distance while rewarding deep interior cells
        # (higher entrance_dist_map value reduces effective cost)
        def _vanguard_score(cell: Coord) -> float:
            d_travel = dist_from_me[cell]
            d_depth = self.entrance_dist_map.get(cell, 0)
            peer_penalty = 0.0
            for ppos in self.last_known_peer_positions.values():
                if abs(cell[0] - ppos[0]) + abs(cell[1] - ppos[1]) <= 3:
                    peer_penalty += 8.0
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

    Independent Decision-Making Logic:
    - Evaluates candidate frontier tiles by computing the expected 5x5 line-of-sight
      Information Gain (number of currently uncleared walkable cells that will be revealed
      from that vantage point) divided by travel distance.
    - Rationale: Rapidly clears open rooms, junctions, and multi-corridor intersections
      to maximize the probability of spotting targets early for the group.
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

    def select_exploration_goal(self, candidate_cells: Set[Coord]) -> Tuple[Coord, str]:
        dist_from_me = bfs_distance_map(self.grid, [self.position])
        reachable_candidates = [c for c in candidate_cells if c in dist_from_me]
        if not reachable_candidates:
            fallback = next(iter(candidate_cells))
            return fallback, "Fallback unreachable candidate."

        # Filter to closest 35 candidates for fast 5x5 information-gain evaluation
        reachable_candidates.sort(key=lambda c: dist_from_me[c])
        shortlist = reachable_candidates[:35]

        def _info_gain_utility(cell: Coord) -> float:
            d_travel = max(1, dist_from_me[cell])
            newly_revealed = len(
                self._get_visible_walkable_from(cell) - self.local_cleared_cells
            )
            peer_penalty = 0.0
            for ppos in self.last_known_peer_positions.values():
                if abs(cell[0] - ppos[0]) + abs(cell[1] - ppos[1]) <= 3:
                    peer_penalty += 6.0
            # Higher newly_revealed / d_travel is better -> negate for min()
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

    Independent Decision-Making Logic:
    - Uses systematic wavefront sweeping (nearest-first frontier clearing) to ensure zero gaps.
    - Autonomous Relay Protocol: If comms with the Coordinator have been lost for >= `relay_threshold`
      steps AND A3 has accumulated substantial unsynced map/target intelligence, A3 briefly
      routes toward the entrance/uplink zone to synchronize with the Coordinator, learn which
      teammates (A1/A2) have gone dark, and dynamically pivot to cover their sectors.
    """

    def __init__(
        self,
        initial_pos: Coord,
        briefing: Dict[str, Any],
        relay_step_threshold: int = 28,
        relay_min_new_tiles: int = 25,
    ) -> None:
        super().__init__(agent_id="A3", initial_pos=initial_pos, briefing=briefing)
        self.relay_step_threshold = relay_step_threshold
        self.relay_min_new_tiles = relay_min_new_tiles

    def should_initiate_relay_sync(self) -> bool:
        # Only relay if we are not already at the entrance and have critical unsynced intel
        if self.position in self.entrances:
            return False
        has_unreported_target = any(
            r.status == TargetStatus.DISCOVERED for r in self.known_targets.values()
        )
        if (
            self.steps_since_sync >= self.relay_step_threshold
            and (
                len(self.unsynced_cleared_cells) >= self.relay_min_new_tiles
                or has_unreported_target
            )
            and len(self.get_all_uncleared()) > 0
        ):
            return True
        return False

    def get_spillover_sector_order(self) -> List[int]:
        # Prioritize any dark teammate's sector first; default 3 -> 1 -> 2
        order = [1, 2]
        if "A2" in self.dark_teammates and "A1" not in self.dark_teammates:
            order = [2, 1]
        return order

    def select_exploration_goal(self, candidate_cells: Set[Coord]) -> Tuple[Coord, str]:
        dist_from_me = bfs_distance_map(self.grid, [self.position])
        reachable_candidates = [c for c in candidate_cells if c in dist_from_me]
        if not reachable_candidates:
            fallback = next(iter(candidate_cells))
            return fallback, "Fallback unreachable candidate."

        def _support_wavefront_score(cell: Coord) -> float:
            d_travel = dist_from_me[cell]
            d_entrance = self.entrance_dist_map.get(cell, 0)
            # Slight preference for clearing near-to-far wavefront systematically
            peer_penalty = 0.0
            for ppos in self.last_known_peer_positions.values():
                if abs(cell[0] - ppos[0]) + abs(cell[1] - ppos[1]) <= 3:
                    peer_penalty += 8.0
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

    Floor plan format:
      - 0: traversable path
      - 1: wall (blocks movement and 5x5 line-of-sight)
      - 'A1', 'A2', 'A3': agent starting positions (optional; defaults to entrance)
    """
    _, parsed_agents, _ = parse_floor_plan_positions(floor_plan)
    coordinator = Coordinator(
        floor_plan=floor_plan,
        total_targets_goal=total_targets_goal,
        entrances=entrances,
    )

    default_entrances = coordinator.entrances
    pos_a1 = parsed_agents.get("A1", default_entrances[0])
    pos_a2 = parsed_agents.get("A2", default_entrances[min(1, len(default_entrances) - 1)])
    pos_a3 = parsed_agents.get("A3", default_entrances[min(2, len(default_entrances) - 1)])

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


def parse_floor_plan_positions(
    floor_plan: Sequence[Sequence[Any]],
) -> Tuple[List[List[int]], Dict[str, Coord], Dict[str, Coord]]:
    from .vision_and_grid import parse_floor_plan

    return parse_floor_plan(floor_plan)
