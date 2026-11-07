"""
Data models, enums, and communication payloads for the Neptune North
Search & Rescue (SAR) Coordinator and 3 Autonomous Agents (A1, A2, A3).

Maze Elements:
- '0' (0): Walkable path
- '1' (1): Wall (blocks movement and 5x5 line-of-sight)
- 'A1', 'A2', 'A3': Field Agents
- 'C': Outside Coordinator / Entrance Base
- 'P': Person / Target to rescue
- 'D': Heavy Door (blocks movement & sight behind it; requires 2 agents adjacent to unlock)
- 'I': Ground Message Marker ("Invite/Assist") asking any agent seeing it to go to a
       designated Door 'D' coordinate to help unlock it when signal is down
- 'O': Ground Message Marker ("Dead End") indicating the corridor beyond is a searched
       dead end so other offline agents skip it
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

Coord = Tuple[int, int]  # (row, col)


class TileType(int, Enum):
    PATH = 0
    WALL = 1


class MazeSymbol(str, Enum):
    PATH = "0"
    WALL = "1"
    AGENT_PREFIX = "A"
    COORDINATOR = "C"
    PERSON = "P"
    DOOR = "D"
    MSG_INVITE_DOOR = "I"
    MSG_DEAD_END = "O"


class MarkerType(str, Enum):
    INVITE_DOOR = "I"  # Ask another agent to go to designated location to open Door 'D'
    DEAD_END = "O"     # Marks that the road after this tile is a searched dead end


@dataclass
class GroundMarker:
    """
    Physical message left on a maze cell ('I' or 'O') by an agent when signal is down.
    Other agents within 5x5 line-of-sight can read this marker and react autonomously.
    """
    marker_type: MarkerType
    position: Coord
    author_agent: str
    target_door_pos: Optional[Coord] = None          # Used by 'I' (designated door location)
    dead_end_cells: Set[Coord] = field(default_factory=set)  # Used by 'O' (cells behind dead end)
    created_step: int = 0

    def copy(self) -> GroundMarker:
        return GroundMarker(
            marker_type=self.marker_type,
            position=self.position,
            author_agent=self.author_agent,
            target_door_pos=self.target_door_pos,
            dead_end_cells=set(self.dead_end_cells),
            created_step=self.created_step,
        )


class TargetStatus(str, Enum):
    DISCOVERED = "DISCOVERED"   # Spotted in 5x5 sight, not yet picked up
    CARRYING = "CARRYING"       # Currently being escorted/carried by an agent
    RESCUED = "RESCUED"         # Safely delivered to entrance / Coordinator


class AgentMode(str, Enum):
    SEARCH_PRIMARY = "SEARCH_PRIMARY"               # Exploring assigned primary sector
    SEARCH_SPILLOVER = "SEARCH_SPILLOVER"           # Primary sector done; helping teammates' sectors
    APPROACH_TARGET = "APPROACH_TARGET"             # Moving to pick up a spotted/known person
    EXTRACT_TARGET = "EXTRACT_TARGET"               # Carrying person out of maze to entrance
    WAIT_FOR_DOOR_PARTNER = "WAIT_FOR_DOOR_PARTNER" # At/near Door 'D', waiting or dropping 'I' for 2nd agent
    ASSIST_OPEN_DOOR = "ASSIST_OPEN_DOOR"           # Responding to 'I' marker or Coordinator order to open 'D'
    RELAY_SYNC = "RELAY_SYNC"                       # Exiting maze to entrance (where signal is always ON)
    MISSION_COMPLETE = "MISSION_COMPLETE"           # All N targets rescued and agent outside maze


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
    visible_cells: Dict[Coord, int]                  # { (r, c): 0 or 1 }
    visible_targets: Dict[str, Coord]                # { target_id: (r, c) } spotted in 5x5
    visible_agents: Dict[str, Coord]                 # { other_agent_id: (r, c) } spotted in 5x5
    visible_doors: Set[Coord] = field(default_factory=set)                   # Closed 'D' doors in 5x5
    visible_markers: Dict[Coord, GroundMarker] = field(default_factory=dict) # 'I' and 'O' markers in 5x5


@dataclass
class AgentReport:
    """
    1-to-1 Uplink message sent from a single Agent (A1, A2, or A3) to the Coordinator
    whenever communication is available (or when outside the maze at the entrance).
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
    known_closed_doors: Set[Coord] = field(default_factory=set)
    opened_doors: Set[Coord] = field(default_factory=set)


@dataclass
class CoordinatorDirective:
    """
    1-to-1 Downlink response sent from the Coordinator back to the synced Agent.
    Contains merged group intelligence and dynamic sector/target/door assignments.
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
    priority_door_pos: Optional[Coord] = None        # Door 'D' that needs 2-agent breach
    known_closed_doors: Set[Coord] = field(default_factory=set)
    opened_doors: Set[Coord] = field(default_factory=set)
    dark_agents: Set[str] = field(default_factory=set)
    other_agent_positions: Dict[str, Coord] = field(default_factory=dict)


@dataclass
class StepDecision:
    """
    Action chosen independently by an Agent at each simulation step.
    """
    agent_id: str
    current_pos: Coord
    next_pos: Coord
    move_delta: Tuple[int, int]
    mode: AgentMode
    synced_with_coordinator: bool
    is_outside_maze: bool = False
    carrying_target_id: Optional[str] = None
    picked_up_target_id: Optional[str] = None
    rescued_target_id: Optional[str] = None
    unlocked_door_pos: Optional[Coord] = None
    dropped_marker: Optional[GroundMarker] = None
    target_goal_pos: Optional[Coord] = None
    visible_cells: Set[Coord] = field(default_factory=set)
    reasoning: str = ""
