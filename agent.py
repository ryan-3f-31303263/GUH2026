"""
Combined & Enhanced Rescue Agent for the Neptune North Search & Rescue Challenge.
Merges teammate's RescueAgent API with decentralized multi-agent intelligence:
1. Working towards a common goal (locating & rescuing all N targets, going in/out of maze multiple times).
2. Independent decision-making under sudden signal drops:
   - Signal is ALWAYS ON when an agent is outside the maze (at the entrance/exit).
   - Inside the maze, signal may stop suddenly at any step; agents continue automatically.
3. Cooperative 2-Agent Door ('D') Unblocking:
   - 'D' on the maze is a heavy door requiring 2 agents adjacent to remove/unblock.
4. Stigmergic Ground Messages ('I' and 'O') when Signal is Down:
   - 'I': Left on the floor when signal is down to ask any agent who sees 'I' in its 5x5
          sight to go to the designated Door 'D' location to help open it.
   - 'O': Left on the floor when signal is down to mark that the road after is a dead end.
"""

from __future__ import annotations

import re
from collections import deque
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from neptune_sar.models import (
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
from neptune_sar.vision_and_grid import (
    bfs_distance_map,
    can_unlock_door,
    compute_5x5_sight,
    detect_dead_end_branch,
    extract_doors_from_floor_plan,
    get_door_stand_positions,
    get_valid_neighbors,
    has_line_of_sight,
    parse_floor_plan,
)

EMPTY, WALL = 0, 1
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class Link(Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"


class RescueAgent:
    """
    Combined Rescue Agent supporting both:
    1. Teammate's API:
       `agent.observe(exits, people, doors, markers); msgs = agent.exchange(); agent.receive(...); agent.step()`
    2. Coordinator 1-to-1 sync API:
       `decision = agent.step(coordinator=coordinator, comm_available=is_online, closed_doors=doors, ground_markers=markers)`
    """

    def __init__(
        self,
        agent_id: str,
        maze: Sequence[Sequence[Any]],
        start: Sequence[int],
        vision: int = 2,
        primary_sector: Optional[int] = None,
        sectors: Optional[Dict[int, Set[Coord]]] = None,
    ):
        self.id: str = str(agent_id)
        self.agent_id: str = self.id
        self.raw_maze = maze
        self.grid, _, _ = parse_floor_plan(maze)
        self.maze = self.grid
        self.rows: int = len(self.grid)
        self.cols: int = len(self.grid[0]) if self.rows > 0 else 0
        self.pos: Coord = (int(start[0]), int(start[1]))
        self.position: Coord = self.pos
        self.vision: int = vision

        # Link state & instructions
        self.link: Link = Link.CONNECTED
        self.instruction: Dict[str, Any] = {"action": "explore"}
        self.carrying: bool = False
        self.carrying_target_id: Optional[str] = None
        self.trips_completed: int = 0

        # Knowledge base
        self.known_exits: Set[Coord] = {self.pos}
        self.known_people: Set[Coord] = set()
        self.known_targets: Dict[str, TargetRecord] = {}
        self.rescued_target_ids: Set[str] = set()
        self.other_agents: Dict[str, Coord] = {}
        self.seen: Set[Coord] = set()
        self.local_cleared_cells: Set[Coord] = self.seen
        self.unsynced_cleared_cells: Set[Coord] = set()

        # Doors ('D') & Ground Markers ('I', 'O')
        self.closed_doors: Set[Coord] = extract_doors_from_floor_plan(maze)
        self.known_closed_doors: Set[Coord] = set()
        self.opened_doors: Set[Coord] = set()
        self.invited_door_goal: Optional[Coord] = None
        self.coordinator_priority_door: Optional[Coord] = None
        self.door_wait_steps: Dict[Coord, int] = {}
        self.placed_dead_end_markers: Set[Coord] = set()
        self.placed_invite_doors: Set[Coord] = set()

        # Sectors & topology
        default_sector = 1 if "1" in self.id else (2 if "2" in self.id else 3)
        self.primary_sector: int = (
            primary_sector if primary_sector is not None else default_sector
        )
        self.active_sector: int = self.primary_sector
        self.sectors: Dict[int, Set[Coord]] = sectors or {}

        # Message & step tracking
        self.outbox: List[Dict[str, Any]] = []
        self.path: List[Coord] = []
        self.step_count: int = 0
        self.steps_since_sync: int = 0
        self.dark_teammates: Set[str] = set()
        self.mode: AgentMode = AgentMode.SEARCH_PRIMARY

        self._mark_seen()

    # ------------------------------------------------------------------
    # Link State (Always ON when outside the maze at entrance/exit)
    # ------------------------------------------------------------------
    def is_outside_maze(self, pos: Optional[Coord] = None) -> bool:
        check_pos = self.pos if pos is None else pos
        return check_pos in self.known_exits

    def set_connection(self, connected: bool):
        if self.is_outside_maze():
            self.link = Link.CONNECTED
        else:
            self.link = Link.CONNECTED if connected else Link.DISCONNECTED

    @property
    def connected(self) -> bool:
        if self.is_outside_maze():
            return True
        return self.link is Link.CONNECTED

    # ------------------------------------------------------------------
    # Perception (Centred 5x5 with Wall & Closed-Door 'D' Occlusion)
    # ------------------------------------------------------------------
    def visible_cells(self, closed_doors: Optional[Set[Coord]] = None) -> Set[Coord]:
        r0, c0 = self.pos
        v = self.vision
        doors = self.closed_doors if closed_doors is None else closed_doors
        visible: Set[Coord] = set()
        for r in range(r0 - v, r0 + v + 1):
            for c in range(c0 - v, c0 + v + 1):
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    cell = (r, c)
                    if cell not in doors and self.maze[r][c] != WALL:
                        if has_line_of_sight(
                            self.grid, self.pos, cell, closed_doors=doors
                        ):
                            visible.add(cell)
        return visible

    def visible_doors(self, closed_doors: Optional[Set[Coord]] = None) -> Set[Coord]:
        r0, c0 = self.pos
        v = self.vision
        doors = self.closed_doors if closed_doors is None else closed_doors
        found: Set[Coord] = set()
        for r in range(r0 - v, r0 + v + 1):
            for c in range(c0 - v, c0 + v + 1):
                cell = (r, c)
                if cell in doors and has_line_of_sight(
                    self.grid, self.pos, cell, closed_doors=doors
                ):
                    found.add(cell)
        return found

    def observe(
        self,
        exits=(),
        people=(),
        doors: Optional[Set[Coord]] = None,
        ground_markers: Optional[Dict[Coord, GroundMarker]] = None,
    ):
        """
        Observes exits, people, closed doors ('D'), and ground markers ('I' and 'O')
        within the agent's 5x5 line-of-sight window.
        """
        if doors is not None:
            self.closed_doors = set(doors)

        curr_visible = self.visible_cells(self.closed_doors)
        curr_doors = self.visible_doors(self.closed_doors)

        exits = {tuple(e) for e in exits if tuple(e) in curr_visible}
        people = {tuple(p) for p in people if tuple(p) in curr_visible}

        for e in exits - self.known_exits:
            self.known_exits.add(e)
            self._signal("exit_found", pos=e)

        for dpos in curr_doors - self.known_closed_doors:
            if dpos not in self.opened_doors:
                self.known_closed_doors.add(dpos)
                self._signal("door_found", pos=dpos)

        # Clean up doors that were unlocked
        for dpos in list(self.known_closed_doors):
            if dpos not in self.closed_doors:
                self.known_closed_doors.discard(dpos)
                self.opened_doors.add(dpos)
                if self.invited_door_goal == dpos:
                    self.invited_door_goal = None

        # Read visible ground markers ('I' and 'O')
        if ground_markers:
            for mpos, marker in ground_markers.items():
                if mpos not in curr_visible:
                    continue
                if marker.marker_type == MarkerType.DEAD_END:
                    # 'O' marker: road ahead is a dead end -> mark cells as seen!
                    new_dead = marker.dead_end_cells - self.seen
                    if new_dead:
                        self.seen.update(new_dead)
                        self.unsynced_cleared_cells.update(new_dead)
                    self.seen.add(mpos)
                elif marker.marker_type == MarkerType.INVITE_DOOR:
                    # 'I' marker: go to designated location to help open Door 'D'!
                    dpos = marker.target_door_pos
                    if (
                        dpos is not None
                        and dpos in self.closed_doors
                        and marker.author_agent != self.id
                        and not self.carrying
                    ):
                        self.known_closed_doors.add(dpos)
                        self.invited_door_goal = dpos
                        self.path = []

        for p in people - self.known_people:
            self.known_people.add(p)
            tid = f"P_{p[0]}_{p[1]}"
            if tid not in self.known_targets:
                self.known_targets[tid] = TargetRecord(
                    target_id=tid,
                    position=p,
                    status=TargetStatus.DISCOVERED,
                    updated_step=self.step_count,
                )
            self._signal("person_found", pos=p)

        for p in (self.known_people & curr_visible) - people:
            self.known_people.discard(p)
            for rec in list(self.known_targets.values()):
                if rec.position == p and rec.status == TargetStatus.DISCOVERED:
                    rec.status = TargetStatus.RESCUED

    # ------------------------------------------------------------------
    # Communication (Teammate's exchange/receive + Coordinator Directive)
    # ------------------------------------------------------------------
    def exchange(self) -> List[Dict[str, Any]]:
        if not self.connected:
            return []
        msgs = list(self.outbox) + [
            {
                "type": "status",
                "agent": self.id,
                "pos": self.pos,
                "carrying": self.carrying,
                "instruction": self.instruction,
                "cleared_count": len(self.seen),
                "known_doors": list(self.known_closed_doors),
            }
        ]
        self.outbox.clear()
        return msgs

    def receive(self, instruction=None, agent_locations=None) -> bool:
        if not self.connected:
            return False
        if instruction is not None:
            new = self.parse_instruction(instruction)
            if new != self.instruction:
                self.instruction = new
                self.path = []
        if agent_locations:
            self.other_agents = {
                str(a): tuple(p) for a, p in agent_locations.items() if str(a) != self.id
            }
        return True

    @staticmethod
    def parse_instruction(instr) -> Dict[str, Any]:
        if isinstance(instr, dict):
            out = {"action": instr.get("action", "explore")}
            if instr.get("target") is not None:
                out["target"] = tuple(instr["target"])
            return out
        text = str(instr).lower()
        nums = [int(n) for n in re.findall(r"-?\d+", text)]
        target = (nums[0], nums[1]) if len(nums) >= 2 else None
        if "door" in text or "unblock" in text or "open" in text:
            action = "open_door"
        elif "rescue" in text or "person" in text:
            action = "rescue"
        elif "exit" in text or "return" in text:
            action = "return"
        elif "hold" in text or "wait" in text or "stay" in text:
            action = "hold"
        elif target and ("go" in text or "move" in text):
            action = "go_to"
        else:
            action = "explore"
        out = {"action": action}
        if target:
            out["target"] = target
        return out

    def build_report(self) -> AgentReport:
        return AgentReport(
            agent_id=self.id,
            step=self.step_count,
            position=self.pos,
            mode=self.mode,
            carrying_target_id=self.carrying_target_id,
            newly_cleared_cells=set(self.unsynced_cleared_cells),
            all_cleared_count=len(self.seen),
            known_targets={k: v.copy() for k, v in self.known_targets.items()},
            rescued_target_ids=set(self.rescued_target_ids),
            known_closed_doors=set(self.known_closed_doors),
            opened_doors=set(self.opened_doors),
        )

    def apply_coordinator_directive(self, directive: CoordinatorDirective) -> None:
        self.steps_since_sync = 0
        self.unsynced_cleared_cells.clear()
        self.seen.update(directive.global_cleared_cells)
        self.other_agents.update(directive.other_agent_positions)
        self.dark_teammates = set(directive.dark_agents)
        self.active_sector = directive.recommended_sector

        self.opened_doors.update(directive.opened_doors)
        self.closed_doors -= self.opened_doors
        self.known_closed_doors.update(directive.known_closed_doors - self.opened_doors)
        self.known_closed_doors -= self.opened_doors
        self.coordinator_priority_door = directive.priority_door_pos

        for tid, grec in directive.global_targets.items():
            if grec.status == TargetStatus.RESCUED:
                self.rescued_target_ids.add(tid)
                self.known_people.discard(grec.position)
                self.known_targets[tid] = grec.copy()
            elif (
                tid not in self.known_targets
                or grec.updated_step >= self.known_targets[tid].updated_step
            ):
                self.known_targets[tid] = grec.copy()
                if grec.status == TargetStatus.DISCOVERED:
                    self.known_people.add(grec.position)

        if directive.priority_door_pos:
            self.instruction = {
                "action": "open_door",
                "target": directive.priority_door_pos,
            }
        elif directive.priority_waypoint:
            self.instruction = {"action": "go_to", "target": directive.priority_waypoint}

    # ------------------------------------------------------------------
    # Step & Decision Making
    # ------------------------------------------------------------------
    def step(
        self,
        coordinator: Any = None,
        comm_available: Any = None,
        active_targets: Optional[Dict[str, Coord]] = None,
        all_agent_positions: Optional[Dict[str, Coord]] = None,
        closed_doors: Optional[Set[Coord]] = None,
        ground_markers: Optional[Dict[Coord, GroundMarker]] = None,
        precomputed_observation: Optional[VisibleObservation] = None,
    ) -> Any:
        self.step_count += 1
        prev_pos = self.pos

        if coordinator is not None:
            self.known_exits.update(coordinator.entrances)

        if closed_doors is not None:
            self.closed_doors = closed_doors

        if all_agent_positions:
            self.other_agents = {
                k: v for k, v in all_agent_positions.items() if k != self.id
            }

        def _is_comm_ok(pos: Coord) -> bool:
            if self.is_outside_maze(pos):
                return True
            if comm_available is None:
                return self.link is Link.CONNECTED
            if callable(comm_available):
                return bool(comm_available(pos))
            return bool(comm_available)

        self.set_connection(_is_comm_ok(self.pos))

        # 1. Perceive surroundings (5x5 sight, doors 'D', markers 'I' & 'O')
        obs_targets = active_targets if active_targets else {}
        self.observe(
            exits=self.known_exits,
            people=[pos for tid, pos in obs_targets.items()]
            if obs_targets
            else self.known_people,
            doors=self.closed_doors,
            ground_markers=ground_markers,
        )

        # 2. Check cooperative 2-agent Door 'D' unlock before moving
        unlocked_1 = self._try_door_unlock(all_agent_positions, ground_markers)

        # 3. Attempt 1-to-1 sync with Coordinator (always succeeds at entrance)
        synced = False
        if coordinator is not None and self.connected:
            directive = coordinator.sync_with_agent(self.id, self.build_report())
            self.apply_coordinator_directive(directive)
            synced = True
        else:
            if coordinator is not None:
                coordinator.record_comm_failure(self.id, self.step_count)
            self.steps_since_sync += 1

        # 4. Drop ground marker ('I' or 'O') if signal is down inside the maze
        dropped_marker = self._maybe_drop_marker(
            synced=synced,
            ground_markers=ground_markers,
            active_targets=active_targets,
        )

        # 5. Choose goal and move one step
        goal = self._choose_goal()
        if goal is not None and goal != self.pos:
            if not self.path or self.path[-1] != goal:
                self.path = self.bfs_path(self.pos, goal)
            if self.path:
                nxt = self.path.pop(0)
                if nxt not in self.closed_doors:
                    self.pos = nxt
                    self.position = self.pos
                else:
                    self.path = []

        self._mark_seen()
        unlocked_2 = self._try_door_unlock(all_agent_positions, ground_markers)
        picked_up, rescued = self._interact(active_targets)

        unlocked_door = unlocked_1 or unlocked_2
        self.set_connection(_is_comm_ok(self.pos))

        # 6. If arriving at entrance (outside maze) or unlocking/rescuing while connected, sync!
        if (
            (rescued or unlocked_door or not synced or self.is_outside_maze(self.pos))
            and self.connected
            and coordinator is not None
        ):
            directive = coordinator.sync_with_agent(self.id, self.build_report())
            self.apply_coordinator_directive(directive)
            synced = True

        if coordinator is not None:
            move_delta = (self.pos[0] - prev_pos[0], self.pos[1] - prev_pos[1])
            return StepDecision(
                agent_id=self.id,
                current_pos=prev_pos,
                next_pos=self.pos,
                move_delta=move_delta,
                mode=self.mode,
                synced_with_coordinator=synced,
                is_outside_maze=self.is_outside_maze(self.pos),
                carrying_target_id=self.carrying_target_id,
                picked_up_target_id=picked_up,
                rescued_target_id=rescued,
                unlocked_door_pos=unlocked_door,
                dropped_marker=dropped_marker,
                target_goal_pos=goal,
                visible_cells=self.visible_cells(self.closed_doors),
                reasoning=f"Agent {self.id} moving towards {goal} [{self.mode.value}]",
            )
        return self.pos

    def _try_door_unlock(
        self,
        all_agent_positions: Optional[Dict[str, Coord]],
        ground_markers: Optional[Dict[Coord, GroundMarker]],
    ) -> Optional[Coord]:
        positions = dict(all_agent_positions or self.other_agents)
        positions[self.id] = self.pos
        for dpos in list(self.closed_doors):
            if can_unlock_door(dpos, positions):
                self.closed_doors.discard(dpos)
                self.known_closed_doors.discard(dpos)
                self.opened_doors.add(dpos)
                if self.invited_door_goal == dpos:
                    self.invited_door_goal = None
                if self.coordinator_priority_door == dpos:
                    self.coordinator_priority_door = None
                self._signal("door_unlocked", pos=dpos)
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

    def _maybe_drop_marker(
        self,
        synced: bool,
        ground_markers: Optional[Dict[Coord, GroundMarker]],
        active_targets: Optional[Dict[str, Coord]],
    ) -> Optional[GroundMarker]:
        if (
            synced
            or self.connected
            or self.is_outside_maze()
            or ground_markers is None
        ):
            return None

        # 1. Leave 'I' marker if we know of a closed door 'D' that needs a 2nd agent
        active_doors = self.known_closed_doors & self.closed_doors
        if active_doors:
            target_door = min(
                active_doors,
                key=lambda d: abs(d[0] - self.pos[0]) + abs(d[1] - self.pos[1]),
            )
            neighbors = get_valid_neighbors(
                self.grid, self.pos, closed_doors=self.closed_doors
            )
            if (
                len(neighbors) >= 3
                or abs(self.pos[0] - target_door[0]) + abs(self.pos[1] - target_door[1])
                <= 2
            ) and self.pos not in ground_markers:
                marker = GroundMarker(
                    marker_type=MarkerType.INVITE_DOOR,
                    position=self.pos,
                    author_agent=self.id,
                    target_door_pos=target_door,
                    created_step=self.step_count,
                )
                ground_markers[self.pos] = marker
                self.placed_invite_doors.add(target_door)
                self._signal("marker_dropped", marker="I", pos=self.pos, door=target_door)
                return marker

        # 2. Leave 'O' marker if exiting a searched dead-end branch
        t_coords = set(active_targets.values()) if active_targets else set(self.known_people)
        dead_info = detect_dead_end_branch(
            grid=self.grid,
            current_pos=self.pos,
            cleared_cells=self.seen,
            entrances=list(self.known_exits),
            closed_doors=self.closed_doors,
            active_target_positions=t_coords,
        )
        if dead_info is not None:
            neck_pos, branch_cells = dead_info
            if (
                neck_pos not in self.placed_dead_end_markers
                and neck_pos not in ground_markers
            ):
                marker = GroundMarker(
                    marker_type=MarkerType.DEAD_END,
                    position=neck_pos,
                    author_agent=self.id,
                    dead_end_cells=branch_cells,
                    created_step=self.step_count,
                )
                ground_markers[neck_pos] = marker
                self.placed_dead_end_markers.add(neck_pos)
                self._signal("marker_dropped", marker="O", pos=neck_pos)
                return marker

        return None

    # ------------------------------------------------------------------
    # Goal Selection & Specialized Heuristics (A1, A2, A3)
    # ------------------------------------------------------------------
    def _choose_goal(self) -> Optional[Coord]:
        if self.carrying:
            self.mode = AgentMode.EXTRACT_TARGET
            return self._nearest(self.known_exits) or self._frontier_goal()

        # Check reachable known people
        dist = self.distances_from(self.pos)
        unclaimed = [
            p
            for p in self.known_people
            if p in dist and p not in self.other_agents.values()
        ]
        if unclaimed:
            self.mode = AgentMode.APPROACH_TARGET
            return min(unclaimed, key=lambda c: dist[c])

        # Check if we need to help open a Door 'D' (via 'I' marker, Coordinator order, or discovery)
        door_target = (
            self.invited_door_goal
            or self.coordinator_priority_door
            or (
                min(
                    self.known_closed_doors & self.closed_doors,
                    key=lambda d: abs(d[0] - self.pos[0]) + abs(d[1] - self.pos[1]),
                )
                if (self.known_closed_doors & self.closed_doors)
                else None
            )
        )
        if door_target is not None and door_target in self.closed_doors:
            wait_cnt = self.door_wait_steps.get(door_target, 0)
            if abs(self.pos[0] - door_target[0]) + abs(self.pos[1] - door_target[1]) <= 1:
                self.door_wait_steps[door_target] = wait_cnt + 1
            if not self.connected and wait_cnt >= 12 and not self.is_outside_maze():
                self.mode = AgentMode.RELAY_SYNC
                ext = self._nearest(self.known_exits)
                if ext == self.pos:
                    self.door_wait_steps[door_target] = 0
                return ext

            self.mode = (
                AgentMode.ASSIST_OPEN_DOOR
                if (self.invited_door_goal == door_target or self.coordinator_priority_door == door_target)
                else AgentMode.WAIT_FOR_DOOR_PARTNER
            )
            stand = get_door_stand_positions(
                self.grid, door_target, closed_doors=self.closed_doors
            )
            reachable_stand = [s for s in stand if s in dist]
            if reachable_stand:
                return min(reachable_stand, key=lambda s: dist[s])

        action = self.instruction.get("action")
        target = self.instruction.get("target")

        if action in ("rescue", "go_to", "open_door") and target == self.pos:
            self.instruction = self._fallback()
            action, target = self.instruction["action"], self.instruction.get("target")

        if action == "hold":
            return None
        if action in ("rescue", "go_to") and target and self.is_open(target) and target in dist:
            return target
        if action == "return":
            self.mode = AgentMode.EXTRACT_TARGET
            return self._nearest(self.known_exits) or self._frontier_goal()

        return self._frontier_goal(region=target)

    def _frontier_goal(self, region: Optional[Coord] = None) -> Optional[Coord]:
        dist = self.distances_from(self.pos)
        unseen = [c for c in dist if c not in self.seen]
        if not unseen:
            self.mode = AgentMode.MISSION_COMPLETE
            return self._nearest(self.known_exits)

        if "1" in self.id:
            self.mode = AgentMode.SEARCH_PRIMARY
            entrance_dist = bfs_distance_map(
                self.grid,
                list(self.known_exits or [self.pos]),
                closed_doors=self.closed_doors,
            )

            def _a1_score(c: Coord) -> float:
                d_travel = dist[c]
                d_depth = entrance_dist.get(c, 0)
                peer_penalty = sum(
                    max(0, 5 - (abs(c[0] - p[0]) + abs(c[1] - p[1]))) * 3
                    for p in self.other_agents.values()
                )
                return d_travel - (0.35 * d_depth) + peer_penalty

            return min(unseen, key=_a1_score)

        elif "2" in self.id:
            self.mode = AgentMode.SEARCH_PRIMARY
            shortlist = sorted(unseen, key=lambda c: dist[c])[:30]

            def _a2_score(c: Coord) -> float:
                d_travel = max(1, dist[c])
                visible_from_c = len(
                    [
                        (r, col)
                        for r in range(c[0] - self.vision, c[0] + self.vision + 1)
                        for col in range(c[1] - self.vision, c[1] + self.vision + 1)
                        if self.is_open((r, col)) and (r, col) not in self.seen
                    ]
                )
                peer_penalty = sum(
                    max(0, 5 - (abs(c[0] - p[0]) + abs(c[1] - p[1]))) * 3
                    for p in self.other_agents.values()
                )
                return (d_travel - 0.65 * visible_from_c) + peer_penalty

            return min(shortlist, key=_a2_score)

        elif "3" in self.id:
            if (
                not self.connected
                and self.steps_since_sync >= 25
                and len(self.unsynced_cleared_cells) >= 20
            ):
                self.mode = AgentMode.RELAY_SYNC
                return self._nearest(self.known_exits)

            self.mode = AgentMode.SEARCH_PRIMARY
            return min(
                unseen,
                key=lambda c: dist[c]
                + sum(
                    max(0, 5 - (abs(c[0] - p[0]) + abs(c[1] - p[1]))) * 3
                    for p in self.other_agents.values()
                ),
            )

        def score(c: Coord) -> float:
            s = float(dist[c])
            if region:
                s += abs(c[0] - region[0]) + abs(c[1] - region[1])
            for p in self.other_agents.values():
                d = abs(c[0] - p[0]) + abs(c[1] - p[1])
                s += max(0, 6 - d) * 2
            return s

        return min(unseen, key=score)

    def _fallback(self) -> Dict[str, Any]:
        person = self._nearest(self.known_people)
        return (
            {"action": "rescue", "target": person}
            if person
            else {"action": "explore"}
        )

    def _interact(
        self, active_targets: Optional[Dict[str, Coord]] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        picked_up: Optional[str] = None
        rescued: Optional[str] = None

        if not self.carrying and self.pos in self.known_people:
            self.carrying = True
            tid = f"P_{self.pos[0]}_{self.pos[1]}"
            if active_targets:
                for k, v in active_targets.items():
                    if v == self.pos:
                        tid = k
                        break
            self.carrying_target_id = tid
            self.known_people.discard(self.pos)
            if tid in self.known_targets:
                self.known_targets[tid].status = TargetStatus.CARRYING
                self.known_targets[tid].assigned_agent = self.id
            self._signal("person_picked_up", pos=self.pos)
            self.instruction = {"action": "return"}
            self.path = []
            picked_up = tid

        elif self.carrying and self.pos in self.known_exits:
            tid = self.carrying_target_id or f"P_{self.pos[0]}_{self.pos[1]}"
            self.carrying = False
            self.carrying_target_id = None
            self.trips_completed += 1
            self.rescued_target_ids.add(tid)
            if tid in self.known_targets:
                self.known_targets[tid].status = TargetStatus.RESCUED
            self._signal("person_rescued", pos=self.pos)
            self.instruction = self._fallback()
            self.path = []
            rescued = tid

        return picked_up, rescued

    # ------------------------------------------------------------------
    # Pathfinding & Helpers
    # ------------------------------------------------------------------
    def is_open(self, cell: Coord) -> bool:
        r, c = cell
        return (
            0 <= r < self.rows
            and 0 <= c < self.cols
            and self.maze[r][c] != WALL
            and cell not in self.closed_doors
        )

    def _neighbours(self, cell: Coord):
        r, c = cell
        for dr, dc in MOVES:
            n = (r + dr, c + dc)
            if self.is_open(n):
                yield n

    def distances_from(self, start: Coord) -> Dict[Coord, int]:
        dist = {start: 0}
        q = deque([start])
        while q:
            cur = q.popleft()
            for n in self._neighbours(cur):
                if n not in dist:
                    dist[n] = dist[cur] + 1
                    q.append(n)
        return dist

    def bfs_path(self, start: Coord, goal: Coord) -> List[Coord]:
        """Shortest path from start to goal (start excluded). [] if unreachable."""
        if goal in self.closed_doors:
            stand = get_door_stand_positions(
                self.grid, goal, closed_doors=self.closed_doors
            )
            if not stand:
                return []
            if start in stand:
                return []
            goal = stand[0]

        prev: Dict[Coord, Optional[Coord]] = {start: None}
        q = deque([start])
        while q:
            cur = q.popleft()
            if cur == goal:
                break
            for n in self._neighbours(cur):
                if n not in prev:
                    prev[n] = cur
                    q.append(n)
        if goal not in prev:
            return []
        path, cur = [], goal
        while cur != start:
            path.append(cur)
            cur = prev[cur]
        return path[::-1]

    def _nearest(self, cells: Sequence[Coord] | Set[Coord]) -> Optional[Coord]:
        if not cells:
            return None
        dist = self.distances_from(self.pos)
        reachable = [c for c in cells if c in dist]
        return min(reachable, key=lambda c: dist[c]) if reachable else None

    def _mark_seen(self):
        new_cells = self.visible_cells(self.closed_doors) - self.seen
        self.seen |= new_cells
        self.unsynced_cleared_cells |= new_cells

    def _signal(self, kind: str, **data):
        self.outbox.append({"type": kind, "agent": self.id, **data})
