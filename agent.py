"""
Combined & Enhanced Rescue Agent for the Neptune North Search & Rescue Challenge.
Merges teammate's RescueAgent API with decentralized multi-agent intelligence:
1. Working towards a common goal (locating & rescuing targets, zero duplicate work).
2. Independent decision-making (specialized heuristics for A1, A2, and A3 under comms failure).
3. 5x5 centred sight with wall-occlusion raycasting.
4. Backwards-compatible with teammate's `observe()`, `exchange()`, `receive()`, and `step()`.
"""

from __future__ import annotations

import re
from collections import deque
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

# Re-use robust raycasting and spatial utilities from neptune_sar
from neptune_sar.models import (
    AgentMode,
    AgentReport,
    Coord,
    CoordinatorDirective,
    StepDecision,
    TargetRecord,
    TargetStatus,
    VisibleObservation,
)
from neptune_sar.vision_and_grid import (
    bfs_distance_map,
    compute_5x5_sight,
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
    1. Teammate's lightweight API:
       `agent.observe(exits, people); msgs = agent.exchange(); agent.receive(...); agent.step()`
    2. Coordinator 1-to-1 sync API:
       `decision = agent.step(coordinator=coordinator, comm_available=is_online)`
    3. Specialized independent behaviors for A1, A2, and A3:
       - A1: Deep-Infiltration Vanguard (sweeps deepest frontiers first)
       - A2: High-Visibility Recon Scout (maximizes 5x5 information gain per step)
       - A3: Adaptive Support & Comms-Relay Mule (wavefront sweeping + emergency relay)
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
        self.agent_id: str = self.id  # alias for neptune_sar compatibility
        self.raw_maze = maze
        self.grid, _, _ = parse_floor_plan(maze)
        self.maze = self.grid  # 2D list of ints (0=path, 1=wall)
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

        # Knowledge base
        self.known_exits: Set[Coord] = set()
        self.known_people: Set[Coord] = set()
        self.known_targets: Dict[str, TargetRecord] = {}
        self.rescued_target_ids: Set[str] = set()
        self.other_agents: Dict[str, Coord] = {}
        self.seen: Set[Coord] = set()
        self.local_cleared_cells: Set[Coord] = self.seen
        self.unsynced_cleared_cells: Set[Coord] = set()

        # Sectors & topology
        default_sector = 1 if "1" in self.id else (2 if "2" in self.id else 3)
        self.primary_sector: int = primary_sector if primary_sector is not None else default_sector
        self.active_sector: int = self.primary_sector
        self.sectors: Dict[int, Set[Coord]] = sectors or {}

        # Message & step tracking
        self.outbox: List[Dict[str, Any]] = []
        self.path: List[Coord] = []
        self.step_count: int = 0
        self.steps_since_sync: int = 0
        self.dark_teammates: Set[str] = set()
        self.mode: AgentMode = AgentMode.SEARCH_PRIMARY

        # Pre-seed entrance if start is at entrance
        self.known_exits.add(self.pos)
        self._mark_seen()

    # ------------------------------------------------------------------
    # Link State
    # ------------------------------------------------------------------
    def set_connection(self, connected: bool):
        self.link = Link.CONNECTED if connected else Link.DISCONNECTED

    @property
    def connected(self) -> bool:
        return self.link is Link.CONNECTED

    # ------------------------------------------------------------------
    # Perception (Centred 5x5 with Wall Occlusion)
    # ------------------------------------------------------------------
    def visible_cells(self) -> Set[Coord]:
        """
        Centred 5x5 line-of-sight sight window (v=2).
        Accounts for wall occlusion using raycasting: cells strictly behind a wall
        cannot be seen.
        """
        r0, c0 = self.pos
        v = self.vision
        visible: Set[Coord] = set()
        for r in range(r0 - v, r0 + v + 1):
            for c in range(c0 - v, c0 + v + 1):
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    if self.is_open((r, c)) and has_line_of_sight(self.grid, self.pos, (r, c)):
                        visible.add((r, c))
        return visible

    def observe(self, exits=(), people=()):
        """
        Simulator calls this with exits and people currently inside 5x5 vision.
        Maintains backward compatibility with teammate's interface while populating
        `known_targets`.
        """
        curr_visible = self.visible_cells()
        exits = {tuple(e) for e in exits if tuple(e) in curr_visible}
        people = {tuple(p) for p in people if tuple(p) in curr_visible}

        for e in exits - self.known_exits:
            self.known_exits.add(e)
            self._signal("exit_found", pos=e)

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

        # People we remembered at visible cells that are now gone were rescued by someone else
        for p in (self.known_people & curr_visible) - people:
            self.known_people.discard(p)
            for rec in list(self.known_targets.values()):
                if rec.position == p and rec.status == TargetStatus.DISCOVERED:
                    rec.status = TargetStatus.RESCUED

    # ------------------------------------------------------------------
    # Communication (Teammate's exchange/receive + Coordinator Directive)
    # ------------------------------------------------------------------
    def exchange(self) -> List[Dict[str, Any]]:
        """Teammate format: returns queued outbox signals + status report when connected."""
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
            }
        ]
        self.outbox.clear()
        return msgs

    def receive(self, instruction=None, agent_locations=None) -> bool:
        """Teammate format: take orders and peer locations from HQ. Ignored if disconnected."""
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
        """Dict {"action": ..., "target": [r, c]} or natural language string."""
        if isinstance(instr, dict):
            out = {"action": instr.get("action", "explore")}
            if instr.get("target") is not None:
                out["target"] = tuple(instr["target"])
            return out
        text = str(instr).lower()
        nums = [int(n) for n in re.findall(r"-?\d+", text)]
        target = (nums[0], nums[1]) if len(nums) >= 2 else None
        if "rescue" in text or "person" in text:
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

    # ------------------------------------------------------------------
    # 1-to-1 Coordinator Protocol Integration
    # ------------------------------------------------------------------
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
        )

    def apply_coordinator_directive(self, directive: CoordinatorDirective) -> None:
        self.steps_since_sync = 0
        self.unsynced_cleared_cells.clear()
        self.seen.update(directive.global_cleared_cells)
        self.other_agents.update(directive.other_agent_positions)
        self.dark_teammates = set(directive.dark_agents)
        self.active_sector = directive.recommended_sector

        # Merge known targets
        for tid, grec in directive.global_targets.items():
            if grec.status == TargetStatus.RESCUED:
                self.rescued_target_ids.add(tid)
                self.known_people.discard(grec.position)
                self.known_targets[tid] = grec.copy()
            elif tid not in self.known_targets or grec.updated_step >= self.known_targets[tid].updated_step:
                self.known_targets[tid] = grec.copy()
                if grec.status == TargetStatus.DISCOVERED:
                    self.known_people.add(grec.position)

        if directive.priority_waypoint:
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
        precomputed_observation: Optional[VisibleObservation] = None,
    ) -> Any:
        """
        Executes one movement step.
        Supports both:
        - Teammate calling `agent.step()` (returns new position `(r, c)`)
        - Coordinator harness calling `agent.step(coordinator=...)` (returns `StepDecision`)
        """
        self.step_count += 1
        prev_pos = self.pos

        # Update peer locations if passed
        if all_agent_positions:
            self.other_agents = {
                k: v for k, v in all_agent_positions.items() if k != self.id
            }

        # Sync connection status
        if comm_available is not None:
            is_comm = (
                comm_available(self.pos)
                if callable(comm_available)
                else bool(comm_available)
            )
            self.set_connection(is_comm)

        # Perceive current surroundings
        obs_targets = active_targets if active_targets else {}
        self.observe(
            exits=self.known_exits,
            people=[pos for tid, pos in obs_targets.items()]
            if obs_targets
            else self.known_people,
        )

        # Attempt 1-to-1 sync with Coordinator if available
        synced = False
        if coordinator is not None and self.connected:
            directive = coordinator.sync_with_agent(self.id, self.build_report())
            self.apply_coordinator_directive(directive)
            synced = True
        else:
            self.steps_since_sync += 1

        # Pick goal using independent decision-making
        goal = self._choose_goal()
        if goal is not None and goal != self.pos:
            if not self.path or self.path[-1] != goal:
                self.path = self.bfs_path(self.pos, goal)
            if self.path:
                self.pos = self.path.pop(0)
                self.position = self.pos

        self._mark_seen()
        picked_up, rescued = self._interact(active_targets)

        # If arriving in comms range upon rescue, sync with Coordinator immediately
        if rescued and coordinator is not None and self.connected:
            coordinator.sync_with_agent(self.id, self.build_report())
            synced = True

        # If running in full harness, return StepDecision; otherwise return tuple pos
        if coordinator is not None:
            move_delta = (self.pos[0] - prev_pos[0], self.pos[1] - prev_pos[1])
            return StepDecision(
                agent_id=self.id,
                current_pos=prev_pos,
                next_pos=self.pos,
                move_delta=move_delta,
                mode=self.mode,
                synced_with_coordinator=synced,
                carrying_target_id=self.carrying_target_id,
                picked_up_target_id=picked_up,
                rescued_target_id=rescued,
                target_goal_pos=goal,
                visible_cells=self.visible_cells(),
                reasoning=f"Agent {self.id} moving towards {goal} [{self.mode.value}]",
            )
        return self.pos

    # ------------------------------------------------------------------
    # Goal Selection & Specialized Heuristics (A1, A2, A3)
    # ------------------------------------------------------------------
    def _choose_goal(self) -> Optional[Coord]:
        if self.carrying:
            self.mode = AgentMode.EXTRACT_TARGET
            return self._nearest(self.known_exits) or self._frontier_goal()

        # Check if known people need rescue
        unclaimed = [p for p in self.known_people if p not in self.other_agents.values()]
        if unclaimed:
            self.mode = AgentMode.APPROACH_TARGET
            nearest_person = self._nearest(unclaimed)
            if nearest_person:
                return nearest_person

        action = self.instruction.get("action")
        target = self.instruction.get("target")

        if action in ("rescue", "go_to") and target == self.pos:
            self.instruction = self._fallback()
            action, target = self.instruction["action"], self.instruction.get("target")

        if action == "hold":
            return None
        if action in ("rescue", "go_to") and target and self.is_open(target):
            return target
        if action == "return":
            self.mode = AgentMode.EXTRACT_TARGET
            return self._nearest(self.known_exits) or self._frontier_goal()

        # Frontier exploration
        return self._frontier_goal(region=target)

    def _frontier_goal(self, region: Optional[Coord] = None) -> Optional[Coord]:
        """
        Chooses next unexplored cell. Incorporates specialized independent logic:
        - A1 (Deep Vanguard): favors deep interior cells away from entrance.
        - A2 (Recon Scout): favors high 5x5 information gain (open junctions/rooms).
        - A3 (Wavefront / Relay): sweeps systematically and covers dark teammates.
        """
        dist = self.distances_from(self.pos)
        unseen = [c for c in dist if c not in self.seen]
        if not unseen:
            self.mode = AgentMode.MISSION_COMPLETE
            return self._nearest(self.known_exits)

        # 1. Specialized A1 Logic: Deep-Infiltration Vanguard
        if "1" in self.id:
            self.mode = AgentMode.SEARCH_PRIMARY
            entrance_dist = bfs_distance_map(self.grid, list(self.known_exits or [self.pos]))

            def _a1_score(c: Coord) -> float:
                d_travel = dist[c]
                d_depth = entrance_dist.get(c, 0)
                peer_penalty = sum(
                    max(0, 5 - (abs(c[0] - p[0]) + abs(c[1] - p[1]))) * 3
                    for p in self.other_agents.values()
                )
                return d_travel - (0.35 * d_depth) + peer_penalty

            return min(unseen, key=_a1_score)

        # 2. Specialized A2 Logic: High-Visibility Recon Scout
        elif "2" in self.id:
            self.mode = AgentMode.SEARCH_PRIMARY
            shortlist = sorted(unseen, key=lambda c: dist[c])[:30]

            def _a2_score(c: Coord) -> float:
                d_travel = max(1, dist[c])
                # Compute expected newly revealed walkable tiles in 5x5
                visible_from_c = len([
                    (r, col)
                    for r in range(c[0] - self.vision, c[0] + self.vision + 1)
                    for col in range(c[1] - self.vision, c[1] + self.vision + 1)
                    if self.is_open((r, col)) and (r, col) not in self.seen
                ])
                peer_penalty = sum(
                    max(0, 5 - (abs(c[0] - p[0]) + abs(c[1] - p[1]))) * 3
                    for p in self.other_agents.values()
                )
                return (d_travel - 0.65 * visible_from_c) + peer_penalty

            return min(shortlist, key=_a2_score)

        # 3. Specialized A3 Logic: Adaptive Wavefront & Relay Mule
        elif "3" in self.id:
            # If dark for a long time with critical intel, ferry toward exit
            if not self.connected and self.steps_since_sync >= 25 and len(self.unsynced_cleared_cells) >= 20:
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

        # Default general exploration scoring
        def score(c: Coord) -> float:
            s = float(dist[c])
            if region:
                s += abs(c[0] - region[0]) + abs(c[1] - region[1])
            for p in self.other_agents.values():
                d = abs(c[0] - p[0]) + abs(c[1] - p[1])
                s += max(0, 6 - d) * 2  # spread away from teammates
            return s

        return min(unseen, key=score)

    def _fallback(self) -> Dict[str, Any]:
        person = self._nearest(self.known_people)
        return {"action": "rescue", "target": person} if person else {"action": "explore"}

    def _interact(
        self, active_targets: Optional[Dict[str, Coord]] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        picked_up: Optional[str] = None
        rescued: Optional[str] = None

        if not self.carrying and self.pos in self.known_people:
            self.carrying = True
            # Find target ID if available
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
        return 0 <= r < self.rows and 0 <= c < self.cols and self.maze[r][c] != WALL

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
        new_cells = self.visible_cells() - self.seen
        self.seen |= new_cells
        self.unsynced_cleared_cells |= new_cells

    def _signal(self, kind: str, **data):
        self.outbox.append({"type": kind, "agent": self.id, **data})
