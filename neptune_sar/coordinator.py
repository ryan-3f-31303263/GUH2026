"""
Outside Coordinator logic for the Neptune North Search & Rescue Challenge.

Responsibilities:
1. Pre-mission spatial decomposition of the known floor plan into 3 balanced sectors (S1, S2, S3).
2. 1-to-1 communication handler (`sync_with_agent`) with individual field agents (A1, A2, A3).
   - Signal is ALWAYS ON when an agent is outside the maze (at the entrance/perimeter).
   - Inside the maze, signal may drop suddenly at any time.
3. Conflict-free state aggregation (cleared tiles, discovered targets, carrying & rescued states,
   closed & unlocked heavy doors 'D').
4. 2-Agent Door ('D') breach coordination and dynamic sector rebalancing when agents go dark.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .models import (
    AgentMode,
    AgentReport,
    Coord,
    CoordinatorDirective,
    TargetRecord,
    TargetStatus,
)
from .vision_and_grid import (
    bfs_distance_map,
    extract_doors_from_floor_plan,
    parse_floor_plan,
    partition_floor_plan_into_sectors,
)

STATUS_PRIORITY = {
    TargetStatus.DISCOVERED: 1,
    TargetStatus.CARRYING: 2,
    TargetStatus.RESCUED: 3,
}


@dataclass
class AgentTelemetry:
    agent_id: str
    assigned_sector: int
    last_known_pos: Coord
    last_sync_step: int
    last_mode: AgentMode = AgentMode.SEARCH_PRIMARY
    carrying_target_id: Optional[str] = None


class Coordinator:
    """
    Outside Base Station Coordinator.
    Communicates strictly 1-to-1 with each agent (A1, A2, A3) when a link is available,
    and is ALWAYS reachable when an agent steps out of the maze to the entrance.
    """

    def __init__(
        self,
        floor_plan: Sequence[Sequence[Any]],
        total_targets_goal: int,
        entrances: Optional[Sequence[Coord]] = None,
        dark_agent_timeout: int = 18,
    ) -> None:
        self.grid, parsed_agents, _ = parse_floor_plan(floor_plan)
        self.initial_doors: Set[Coord] = extract_doors_from_floor_plan(floor_plan)
        self.rows = len(self.grid)
        self.cols = len(self.grid[0]) if self.rows > 0 else 0
        self.total_targets_goal = total_targets_goal
        self.dark_agent_timeout = dark_agent_timeout

        # Determine entrance(s) (outside perimeter base where Coordinator sits)
        if entrances:
            self.entrances: List[Coord] = list(entrances)
        elif parsed_agents:
            self.entrances = list(dict.fromkeys(parsed_agents.values()))
        else:
            border_openings = self._find_border_entrances()
            self.entrances = (
                border_openings if border_openings else [self._find_first_walkable()]
            )

        # Partition reachable walkable floor plan into 3 sectors for A1, A2, A3
        self.sectors: Dict[int, Set[Coord]] = partition_floor_plan_into_sectors(
            self.grid, self.entrances, num_sectors=3
        )
        self.all_walkable: Set[Coord] = set().union(*self.sectors.values())
        self.entrance_dist_map: Dict[Coord, int] = bfs_distance_map(
            self.grid, self.entrances
        )

        # Global Knowledge Base
        self.global_cleared_cells: Set[Coord] = set()
        self.global_targets: Dict[str, TargetRecord] = {}
        self.rescued_target_ids: Set[str] = set()
        self.known_closed_doors: Set[Coord] = set()
        self.opened_doors: Set[Coord] = set()
        self.door_assigned_agents: Dict[Coord, Set[str]] = {}
        self.current_step: int = 0

        # Agent Telemetry Registry (A1 -> Sector 1, A2 -> Sector 2, A3 -> Sector 3)
        self.agent_sector_map: Dict[str, int] = {"A1": 1, "A2": 2, "A3": 3}
        self.telemetry: Dict[str, AgentTelemetry] = {}
        for idx, aid in enumerate(("A1", "A2", "A3"), start=1):
            start_pos = parsed_agents.get(
                aid, self.entrances[(idx - 1) % len(self.entrances)]
            )
            self.telemetry[aid] = AgentTelemetry(
                agent_id=aid,
                assigned_sector=idx,
                last_known_pos=start_pos,
                last_sync_step=0,
            )

        # Communication statistics
        self.sync_events_count: Dict[str, int] = {"A1": 0, "A2": 0, "A3": 0}
        self.failed_sync_attempts: Dict[str, int] = {"A1": 0, "A2": 0, "A3": 0}

    def _find_border_entrances(self) -> List[Coord]:
        """Finds all walkable ('0') openings along the outer border of the maze (from `maze.poke`)."""
        openings: List[Coord] = []
        for c in range(self.cols):
            if self.rows > 0 and self.grid[0][c] == 0:
                openings.append((0, c))
            if self.rows > 1 and self.grid[self.rows - 1][c] == 0:
                openings.append((self.rows - 1, c))
        for r in range(1, self.rows - 1):
            if self.cols > 0 and self.grid[r][0] == 0:
                openings.append((r, 0))
            if self.cols > 1 and self.grid[r][self.cols - 1] == 0:
                openings.append((r, self.cols - 1))
        return openings

    def _find_first_walkable(self) -> Coord:
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == 0:
                    return (r, c)
        return (0, 0)

    def is_outside_maze(self, pos: Coord) -> bool:
        """
        Returns True when the agent is at the outside entrance/exit perimeter.
        Signal to the Coordinator is ALWAYS ON when `is_outside_maze(pos)` is True.
        """
        return pos in self.entrances

    def get_initial_briefing(self, agent_id: str) -> Dict[str, Any]:
        """
        Pre-deployment briefing given to each agent at the entrance (t=0)
        before entering the maze.
        """
        primary_sector = self.agent_sector_map.get(agent_id, 1)
        return {
            "agent_id": agent_id,
            "grid": [row[:] for row in self.grid],
            "entrances": list(self.entrances),
            "total_targets_goal": self.total_targets_goal,
            "primary_sector": primary_sector,
            "sectors": {sid: set(cells) for sid, cells in self.sectors.items()},
            "entrance_dist_map": dict(self.entrance_dist_map),
            "initial_doors": set(self.initial_doors),
        }

    def record_comm_failure(self, agent_id: str, step: int) -> None:
        """Logs when an interior 1-to-1 communication attempt fails due to sudden signal loss."""
        self.current_step = max(self.current_step, step)
        if agent_id in self.failed_sync_attempts:
            self.failed_sync_attempts[agent_id] += 1

    def sync_with_agent(
        self, agent_id: str, report: AgentReport
    ) -> CoordinatorDirective:
        """
        1-to-1 communication endpoint between Coordinator and a single Agent.
        1. Merges the agent's local discoveries (cleared cells, targets, doors 'D') into global state.
        2. Evaluates group progress and checks if any teammate is dark/overdue.
        3. Assigns 2-agent teams to unlock any discovered heavy Door ('D') or rescue targets.
        """
        self.current_step = max(self.current_step, report.step)
        self.sync_events_count[agent_id] = self.sync_events_count.get(agent_id, 0) + 1

        # 1. Merge cleared cells
        self.global_cleared_cells.update(report.newly_cleared_cells)

        # 2. Merge door states ('D')
        for dpos in report.opened_doors:
            self.opened_doors.add(dpos)
            self.known_closed_doors.discard(dpos)
            self.door_assigned_agents.pop(dpos, None)

        for dpos in report.known_closed_doors:
            if dpos not in self.opened_doors:
                self.known_closed_doors.add(dpos)

        # 3. Merge rescued target IDs
        for tid in report.rescued_target_ids:
            self.rescued_target_ids.add(tid)
            if tid in self.global_targets:
                self.global_targets[tid].status = TargetStatus.RESCUED
                self.global_targets[tid].updated_step = self.current_step

        # 4. Reconcile target records (CRDT-style state merge)
        for tid, incoming in report.known_targets.items():
            if tid in self.rescued_target_ids or incoming.status == TargetStatus.RESCUED:
                self.rescued_target_ids.add(tid)
                rec = incoming.copy()
                rec.status = TargetStatus.RESCUED
                self.global_targets[tid] = rec
                continue

            if tid not in self.global_targets:
                self.global_targets[tid] = incoming.copy()
            else:
                existing = self.global_targets[tid]
                inc_prio = STATUS_PRIORITY[incoming.status]
                ext_prio = STATUS_PRIORITY[existing.status]
                if inc_prio > ext_prio or (
                    inc_prio == ext_prio and incoming.updated_step >= existing.updated_step
                ):
                    self.global_targets[tid] = incoming.copy()

        # 5. Update agent telemetry
        if agent_id in self.telemetry:
            tel = self.telemetry[agent_id]
            tel.last_known_pos = report.position
            tel.last_sync_step = report.step
            tel.last_mode = report.mode
            tel.carrying_target_id = report.carrying_target_id

        # 6. Identify dark (out-of-comms) agents
        dark_agents: Set[str] = set()
        for aid, tel in self.telemetry.items():
            if aid != agent_id and (self.current_step - tel.last_sync_step) >= self.dark_agent_timeout:
                dark_agents.add(aid)

        # 7. Compute priority target, 2-agent door breach assignment, & recommended sector
        priority_target_id = self._assign_priority_target(agent_id, report)
        priority_door_pos = self._assign_door_breach(agent_id, report)
        recommended_sector, priority_waypoint = self._recommend_sector_and_waypoint(
            agent_id, report.position, dark_agents
        )

        other_positions = {
            aid: tel.last_known_pos
            for aid, tel in self.telemetry.items()
            if aid != agent_id
        }

        return CoordinatorDirective(
            step=self.current_step,
            total_targets_goal=self.total_targets_goal,
            total_rescued_count=len(self.rescued_target_ids),
            mission_complete=self.is_mission_complete(),
            global_cleared_cells=set(self.global_cleared_cells),
            global_targets={k: v.copy() for k, v in self.global_targets.items()},
            recommended_sector=recommended_sector,
            priority_target_id=priority_target_id,
            priority_waypoint=priority_waypoint,
            priority_door_pos=priority_door_pos,
            known_closed_doors=set(self.known_closed_doors),
            opened_doors=set(self.opened_doors),
            dark_agents=dark_agents,
            other_agent_positions=other_positions,
        )

    def _assign_door_breach(
        self, agent_id: str, report: AgentReport
    ) -> Optional[Coord]:
        """
        Coordinates 2 agents to unblock any discovered closed Door ('D').
        If a closed door needs breaching and fewer than 2 active agents are assigned to it
        (or this agent is already one of the 2 assigned agents), directs this agent to the door.
        """
        if report.carrying_target_id is not None or not self.known_closed_doors:
            return None

        for dpos in sorted(self.known_closed_doors):
            if dpos in self.opened_doors:
                continue
            assigned = self.door_assigned_agents.setdefault(dpos, set())
            # Prune any agent that is currently carrying a person
            assigned = {
                aid
                for aid in assigned
                if self.telemetry.get(aid) is None
                or self.telemetry[aid].carrying_target_id is None
            }
            self.door_assigned_agents[dpos] = assigned

            if agent_id in assigned:
                return dpos
            if len(assigned) < 2:
                assigned.add(agent_id)
                return dpos
        return None

    def _assign_priority_target(
        self, agent_id: str, report: AgentReport
    ) -> Optional[str]:
        """
        If there is a discovered target that is not yet being carried or rescued,
        assigns the closest available agent to pick it up.
        """
        if report.carrying_target_id is not None:
            return report.carrying_target_id

        best_tid: Optional[str] = None
        best_dist = 999999

        for tid, trec in self.global_targets.items():
            if trec.status != TargetStatus.DISCOVERED:
                continue
            assigned = trec.assigned_agent
            if assigned in (None, agent_id) or (
                assigned in self.telemetry
                and (self.current_step - self.telemetry[assigned].last_sync_step)
                >= self.dark_agent_timeout
            ):
                dist = abs(report.position[0] - trec.position[0]) + abs(
                    report.position[1] - trec.position[1]
                )
                if dist < best_dist:
                    best_dist = dist
                    best_tid = tid

        if best_tid is not None:
            self.global_targets[best_tid].assigned_agent = agent_id
        return best_tid

    def _recommend_sector_and_waypoint(
        self, agent_id: str, agent_pos: Coord, dark_agents: Set[str]
    ) -> Tuple[int, Optional[Coord]]:
        primary_sec = self.agent_sector_map.get(agent_id, 1)
        uncleared_by_sector: Dict[int, Set[Coord]] = {
            sid: (cells - self.global_cleared_cells) - self.known_closed_doors
            for sid, cells in self.sectors.items()
        }

        if uncleared_by_sector.get(primary_sec):
            chosen_sec = primary_sec
        else:
            dark_sectors = {
                self.agent_sector_map[da]
                for da in dark_agents
                if da in self.agent_sector_map
            }
            candidates = [sid for sid, rem in uncleared_by_sector.items() if rem]
            if not candidates:
                return primary_sec, None
            candidates.sort(
                key=lambda sid: (
                    0 if sid in dark_sectors else 1,
                    -len(uncleared_by_sector[sid]),
                )
            )
            chosen_sec = candidates[0]

        remaining_cells = uncleared_by_sector.get(chosen_sec, set())
        if not remaining_cells:
            return chosen_sec, None

        waypoint = min(
            remaining_cells,
            key=lambda c: abs(c[0] - agent_pos[0]) + abs(c[1] - agent_pos[1]),
        )
        return chosen_sec, waypoint

    def is_mission_complete(self) -> bool:
        if len(self.rescued_target_ids) >= self.total_targets_goal:
            return True
        if (
            self.all_walkable
            and not self.known_closed_doors
            and self.all_walkable.issubset(self.global_cleared_cells)
            and all(
                t.status == TargetStatus.RESCUED for t in self.global_targets.values()
            )
        ):
            return True
        return False

    def get_summary(self) -> Dict[str, Any]:
        total_walkable = max(1, len(self.all_walkable))
        return {
            "step": self.current_step,
            "targets_rescued": len(self.rescued_target_ids),
            "targets_goal": self.total_targets_goal,
            "targets_discovered": len(self.global_targets),
            "doors_opened": len(self.opened_doors),
            "doors_remaining_closed": len(self.known_closed_doors),
            "coverage_ratio": round(
                len(self.global_cleared_cells) / total_walkable, 4
            ),
            "cleared_cells_count": len(self.global_cleared_cells),
            "total_walkable_cells": len(self.all_walkable),
            "mission_complete": self.is_mission_complete(),
            "sync_counts": dict(self.sync_events_count),
            "failed_syncs": dict(self.failed_sync_attempts),
        }
