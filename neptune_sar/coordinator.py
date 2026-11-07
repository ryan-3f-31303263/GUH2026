"""
Outside Coordinator logic for the Neptune North Search & Rescue Challenge.

Responsibilities:
1. Pre-mission spatial decomposition of the known floor plan into 3 balanced sectors (S1, S2, S3).
2. 1-to-1 communication handler (`sync_with_agent`) with individual field agents (A1, A2, A3).
3. Conflict-free state aggregation (cleared tiles, discovered targets, carrying & rescued states).
4. Dynamic sector rebalancing and target triage when one or more agents go dark (comms failure).
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
    Communicates strictly 1-to-1 with each agent (A1, A2, A3) when a link is available.
    """

    def __init__(
        self,
        floor_plan: Sequence[Sequence[Any]],
        total_targets_goal: int,
        entrances: Optional[Sequence[Coord]] = None,
        dark_agent_timeout: int = 18,
    ) -> None:
        self.grid, parsed_agents, _ = parse_floor_plan(floor_plan)
        self.rows = len(self.grid)
        self.cols = len(self.grid[0]) if self.rows > 0 else 0
        self.total_targets_goal = total_targets_goal
        self.dark_agent_timeout = dark_agent_timeout

        # Determine entrance(s): explicit, or initial agent positions, or top-left walkable cell
        if entrances:
            self.entrances: List[Coord] = list(entrances)
        elif parsed_agents:
            self.entrances = list(dict.fromkeys(parsed_agents.values()))
        else:
            self.entrances = [self._find_first_walkable()]

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

    def _find_first_walkable(self) -> Coord:
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == 0:
                    return (r, c)
        return (0, 0)

    def get_initial_briefing(self, agent_id: str) -> Dict[str, Any]:
        """
        Pre-deployment briefing given to each agent at the entrance (t=0)
        before entering the facility.
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
        }

    def record_comm_failure(self, agent_id: str, step: int) -> None:
        """Logs when a 1-to-1 communication attempt fails due to blackout/range/walls."""
        self.current_step = max(self.current_step, step)
        if agent_id in self.failed_sync_attempts:
            self.failed_sync_attempts[agent_id] += 1

    def sync_with_agent(
        self, agent_id: str, report: AgentReport
    ) -> CoordinatorDirective:
        """
        1-to-1 communication endpoint between Coordinator and a single Agent.
        1. Merges the agent's local discoveries into the Coordinator's global state.
        2. Evaluates group progress and checks if any teammate is dark/overdue.
        3. Computes a custom directive (target assignment, sector rebalance) for this agent.
        """
        self.current_step = max(self.current_step, report.step)
        self.sync_events_count[agent_id] = self.sync_events_count.get(agent_id, 0) + 1

        # 1. Merge cleared cells
        self.global_cleared_cells.update(report.newly_cleared_cells)

        # 2. Merge rescued target IDs
        for tid in report.rescued_target_ids:
            self.rescued_target_ids.add(tid)
            if tid in self.global_targets:
                self.global_targets[tid].status = TargetStatus.RESCUED
                self.global_targets[tid].updated_step = self.current_step

        # 3. Reconcile target records (CRDT-style state merge)
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

        # 4. Update agent telemetry
        if agent_id in self.telemetry:
            tel = self.telemetry[agent_id]
            tel.last_known_pos = report.position
            tel.last_sync_step = report.step
            tel.last_mode = report.mode
            tel.carrying_target_id = report.carrying_target_id

        # 5. Identify dark (out-of-comms) agents
        dark_agents: Set[str] = set()
        for aid, tel in self.telemetry.items():
            if aid != agent_id and (self.current_step - tel.last_sync_step) >= self.dark_agent_timeout:
                dark_agents.add(aid)

        # 6. Compute priority target & recommended sector for this agent
        priority_target_id = self._assign_priority_target(agent_id, report)
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
            dark_agents=dark_agents,
            other_agent_positions=other_positions,
        )

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
            # Check if assigned to an actively connected agent or if unassigned / assigned to a dark agent
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
        """
        Determines which sector the agent should focus on next:
        1. If the agent's primary sector still has uncleared cells, stay on primary sector.
        2. If primary sector is complete, prioritize any sector belonging to a dark (disconnected)
           agent that has uncleared cells, or the sector with the most uncleared cells.
        """
        primary_sec = self.agent_sector_map.get(agent_id, 1)
        uncleared_by_sector: Dict[int, Set[Coord]] = {
            sid: cells - self.global_cleared_cells
            for sid, cells in self.sectors.items()
        }

        if uncleared_by_sector.get(primary_sec):
            chosen_sec = primary_sec
        else:
            # Prioritize dark agents' uncleared sectors first, then most uncleared cells
            dark_sectors = {self.agent_sector_map[da] for da in dark_agents if da in self.agent_sector_map}
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

        # Pick nearest uncleared cell in the recommended sector as a guide waypoint
        waypoint = min(
            remaining_cells,
            key=lambda c: abs(c[0] - agent_pos[0]) + abs(c[1] - agent_pos[1]),
        )
        return chosen_sec, waypoint

    def is_mission_complete(self) -> bool:
        """
        Mission is complete when all N targets have been rescued, OR the entire walkable
        floor plan has been 100% cleared and all discovered targets are rescued.
        """
        if len(self.rescued_target_ids) >= self.total_targets_goal:
            return True
        if (
            self.all_walkable
            and self.all_walkable.issubset(self.global_cleared_cells)
            and all(
                t.status == TargetStatus.RESCUED for t in self.global_targets.values()
            )
        ):
            return True
        return False

    def get_summary(self) -> Dict[str, Any]:
        """Returns global telemetry snapshot for UI / evaluation."""
        total_walkable = max(1, len(self.all_walkable))
        return {
            "step": self.current_step,
            "targets_rescued": len(self.rescued_target_ids),
            "targets_goal": self.total_targets_goal,
            "targets_discovered": len(self.global_targets),
            "coverage_ratio": round(
                len(self.global_cleared_cells) / total_walkable, 4
            ),
            "cleared_cells_count": len(self.global_cleared_cells),
            "total_walkable_cells": len(self.all_walkable),
            "mission_complete": self.is_mission_complete(),
            "sync_counts": dict(self.sync_events_count),
            "failed_syncs": dict(self.failed_sync_attempts),
        }
