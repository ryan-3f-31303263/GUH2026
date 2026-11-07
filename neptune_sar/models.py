"""
Data models, enums, and communication payloads for the Neptune North
Search & Rescue (SAR) Coordinator and 3 Autonomous Agents (A1, A2, A3).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

Coord = Tuple[int, int]  # (row, col)


class TileType( int, Enum ):
    PATH = 0
    WALL = 1


class TargetStatus( str, Enum ):
    DISCOVERED = "DISCOVERED"   # Spotted in 5x5 sight, not yet picked up
    CARRYING = "CARRYING"       # Currently being escorted/carried by an agent
    RESCUED = "RESCUED"         # Safely delivered to entrance / Coordinator


class AgentMode( str, Enum ):
    SEARCH_PRIMARY = "SEARCH_PRIMARY"       # Exploring assigned primary sector
    SEARCH_SPILLOVER = "SEARCH_SPILLOVER"   # Primary sector done; helping teammates' sectors
    APPROACH_TARGET = "APPROACH_TARGET"     # Moving to pick up a spotted/known person
    EXTRACT_TARGET = "EXTRACT_TARGET"       # Carrying person back to entrance/exit
    RELAY_SYNC = "RELAY_SYNC"               # Returning toward comms zone to sync critical intel
    MISSION_COMPLETE = "MISSION_COMPLETE"   # All N targets accounted for and at exit


@dataclass
class TargetRecord:
    """Tracks the state of a discovered person/objective."""
    target_id: str
    position: Coord
    status: TargetStatus = TargetStatus.DISCOVERED
    assigned_agent: Optional[str] = None
    updated_step: int = 0

    def copy(self) -> TargetRecord:
        return TargetRecord(
            target_id=self.target_id,
            position=self.position,
            status=self.status,
            assigned_agent=self.assigned_agent,
            updated_step=self.updated_step,
        )


@dataclass
class VisibleObservation:
    """
    Output of an agent's centred 5x5 line-of-sight perception at a single step.
    """
    agent_id: str
    center: Coord
    visible_cells: Dict[Coord, int]           # { (r, c): 0 or 1 }
    visible_targets: Dict[str, Coord]         # { target_id: (r, c) } spotted in 5x5
    visible_agents: Dict[str, Coord]          # { other_agent_id: (r, c) } spotted in 5x5


@dataclass
class AgentReport:
    """
    1-to-1 Uplink message sent from a single Agent (A1, A2, or A3) to the Coordinator
    whenever communication is available.
    """
    agent_id: str
    step: int
    position: Coord
    mode: AgentMode
    carrying_target_id: Optional[str]
    newly_cleared_cells: Set[Coord]
    all_cleared_count: int
    known_targets: Dict[str, TargetRecord]
    rescued_target_ids: Set[str]


@dataclass
class CoordinatorDirective:
    """
    1-to-1 Downlink response sent from the Coordinator back to the synced Agent.
    Contains merged group intelligence and dynamic sector/target assignments.
    """
    step: int
    total_targets_goal: int
    total_rescued_count: int
    mission_complete: bool
    global_cleared_cells: Set[Coord]
    global_targets: Dict[str, TargetRecord]
    recommended_sector: int
    priority_target_id: Optional[str] = None
    priority_waypoint: Optional[Coord] = None
    dark_agents: Set[str] = field(default_factory=set)
    other_agent_positions: Dict[str, Coord] = field(default_factory=dict)


@dataclass
class StepDecision:
    """
    Action chosen independently by an Agent at each simulation step.
    Easy for teammates' environment / UI to consume.
    """
    agent_id: str
    current_pos: Coord
    next_pos: Coord
    move_delta: Tuple[int, int]
    mode: AgentMode
    synced_with_coordinator: bool
    carrying_target_id: Optional[str] = None
    picked_up_target_id: Optional[str] = None
    rescued_target_id: Optional[str] = None
    target_goal_pos: Optional[Coord] = None
    visible_cells: Set[Coord] = field(default_factory=set)
    reasoning: str = ""
