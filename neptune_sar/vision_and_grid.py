"""
Floor plan parsing, centred 5x5 sight with wall/door occlusion raycasting,
shortest-path navigation (BFS/A*), 3-sector spatial partitioning,
2-agent Door ('D') helpers, and offline Ground Marker ('I' and 'O') topology utilities.
"""

from __future__ import annotations
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from models import Coord, GroundMarker, MarkerType, TileType, VisibleObservation


def extract_doors_from_floor_plan(raw_grid: Sequence[Sequence[Any]]) -> Set[Coord]:
    """Extracts all initial 'D' (Door) coordinates from a raw 2D floor plan array."""
    doors: Set[Coord] = set()
    for r, row in enumerate(raw_grid):
        for c, cell in enumerate(row):
            if isinstance(cell, str) and cell.strip().upper() == "D":
                doors.add((r, c))
    return doors


def parse_floor_plan(
    raw_grid: Sequence[Sequence[Any]],
) -> Tuple[List[List[int]], Dict[str, Coord], Dict[str, Coord]]:
    """
    Parses a 2D floor plan array where:
      - 0 or '0' is traversable path
      - 1 or '1' is wall
      - 'A1', 'A2', 'A3' are agent starting positions (also traversable path=0)
      - 'P', 'T', or 'P1', 'P2'... are target/person markers
      - 'D' is a heavy door (underlying tile is 0 once unlocked by 2 agents)
      - 'C' is the Coordinator / entrance marker
      - 'I', 'O' are ground message markers

    Returns:
      - grid: 2D list of ints (0 = path, 1 = wall)
      - agent_positions: {'A1': (r, c), 'A2': (r, c), 'A3': (r, c)}
      - target_positions: {'P1': (r, c), ...} if embedded in raw_grid
    """
    grid: List[List[int]] = []
    agent_positions: Dict[str, Coord] = {}
    target_positions: Dict[str, Coord] = {}
    target_counter = 1

    for r, row in enumerate(raw_grid):
        parsed_row: List[int] = []
        for c, cell in enumerate(row):
            if cell in (1, "1", True, "#"):
                parsed_row.append(TileType.WALL.value)
            elif cell in (0, "0", False, "."):
                parsed_row.append(TileType.PATH.value)
            elif isinstance(cell, str):
                token = cell.strip().upper()
                if token in ("A1", "A2", "A3"):
                    agent_positions[token] = (r, c)
                    parsed_row.append(TileType.PATH.value)
                elif (
                    token.startswith("P")
                    or token.startswith("T")
                    or token.startswith("C")
                ):
                    tid = token if len(token) > 1 else f"C{target_counter}"
                    target_positions[tid] = (r, c)
                    target_counter += 1
                    parsed_row.append(TileType.PATH.value)
                else:
                    # 'D', 'I', 'O' sit on walkable structural cells (0)
                    parsed_row.append(TileType.PATH.value)
            else:
                parsed_row.append(TileType.PATH.value)
        grid.append(parsed_row)

    return grid, agent_positions, target_positions


def to_visualization_schema(
    grid: Sequence[Sequence[int]],
    agent_positions: Optional[Dict[str, Coord]] = None,
    coordinator_positions: Optional[Sequence[Coord]] = None,
    target_positions: Optional[Dict[str, Coord]] = None,
    closed_doors: Optional[Set[Coord]] = None,
    ground_markers: Optional[Dict[Coord, GroundMarker]] = None,
) -> List[List[str]]:
    """
    Exports current grid and entity locations into the 2D string schema expected
    by the Tkinter visualizer (`maze_vis.py`):
      - '0': path
      - '1': wall
      - 'D': closed door (requires 2 agents to unblock)
      - 'I': invite message marker (go to designated door 'D')
      - 'O': dead-end message marker (road ahead is a searched dead end)
      - 'C': coordinator / entrance checkpoint
      - 'P': target / survivor
      - 'A1' / 'A2' / 'A3': agent
    """
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    schema: List[List[str]] = [
        [str(grid[r][c]) for c in range(cols)] for r in range(rows)
    ]

    if ground_markers:
        for (mr, mc), marker in ground_markers.items():
            if 0 <= mr < rows and 0 <= mc < cols and grid[mr][mc] == 0:
                schema[mr][mc] = marker.marker_type.value

    if closed_doors:
        for dr, dc in closed_doors:
            if 0 <= dr < rows and 0 <= dc < cols:
                schema[dr][dc] = "D"

    if coordinator_positions:
        for cr, cc in coordinator_positions:
            if 0 <= cr < rows and 0 <= cc < cols:
                schema[cr][cc] = "C"

    if target_positions:
        for tid, (tr, tc) in target_positions.items():
            if 0 <= tr < rows and 0 <= tc < cols:
                schema[tr][tc] = tid

    if agent_positions:
        for aid, (ar, ac) in agent_positions.items():
            if 0 <= ar < rows and 0 <= ac < cols:
                schema[ar][ac] = aid

    return schema


def has_line_of_sight(
    grid: Sequence[Sequence[int]],
    start: Coord,
    end: Coord,
    closed_doors: Optional[Set[Coord]] = None,
) -> bool:
    """
    Determines whether `end` is visible from `start` within the grid,
    accounting for wall occlusion (value == 1) and closed doors ('D').
    Note: A wall or closed door 'D' at `end` itself is visible if no intermediate
    obstacle blocks the ray; cells strictly *behind* it are occluded.
    """
    r0, c0 = start
    r1, c1 = end
    if (r0, c0) == (r1, c1):
        return True

    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    if not (0 <= r1 < rows and 0 <= c1 < cols):
        return False

    doors = closed_doors or set()

    def _is_opaque(cell: Coord) -> bool:
        return grid[cell[0]][cell[1]] == 1 or cell in doors

    dr = r1 - r0
    dc = c1 - c0
    steps = max(abs(dr), abs(dc))

    if steps == 1:
        if abs(dr) == 1 and abs(dc) == 1:
            if _is_opaque((r0 + dr, c0)) and _is_opaque((r0, c0 + dc)):
                return False
        return True

    num_samples = 20
    prev_cell = (r0, c0)
    for i in range(1, num_samples):
        t = i / num_samples
        rf = r0 + dr * t
        cf = c0 + dc * t
        cr = int(round(rf))
        cc = int(round(cf))
        curr_cell = (cr, cc)

        if curr_cell in ((r0, c0), (r1, c1)):
            prev_cell = curr_cell
            continue

        if _is_opaque(curr_cell):
            return False

        pr, pc = prev_cell
        if abs(cr - pr) == 1 and abs(cc - pc) == 1:
            if _is_opaque((cr, pc)) and _is_opaque((pr, cc)):
                return False

        prev_cell = curr_cell

    sgn_r = (1 if dr > 0 else -1) if dr != 0 else 0
    sgn_c = (1 if dc > 0 else -1) if dc != 0 else 0

    if abs(dr) == 2 and abs(dc) == 1:
        mid_ortho = (r0 + sgn_r, c0)
        mid_diag = (r0 + sgn_r, c0 + sgn_c)
        if _is_opaque(mid_ortho) and _is_opaque(mid_diag):
            return False
    elif abs(dr) == 1 and abs(dc) == 2:
        mid_ortho = (r0, c0 + sgn_c)
        mid_diag = (r0 + sgn_r, c0 + sgn_c)
        if _is_opaque(mid_ortho) and _is_opaque(mid_diag):
            return False

    return True


def compute_5x5_sight(
    grid: Sequence[Sequence[int]],
    center: Coord,
    agent_id: str = "A1",
    active_targets: Optional[Dict[str, Coord]] = None,
    all_agent_positions: Optional[Dict[str, Coord]] = None,
    closed_doors: Optional[Set[Coord]] = None,
    ground_markers: Optional[Dict[Coord, GroundMarker]] = None,
) -> VisibleObservation:
    """
    Computes the centred 5x5 sight around `center = (r, c)` (i.e. r-2..r+2, c-2..c+2)
    with wall and closed-door ('D') occlusion.
    Also detects visible closed doors ('D') and ground markers ('I' and 'O').
    """
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    r0, c0 = center
    doors = closed_doors or set()

    visible_cells: Dict[Coord, int] = {}
    visible_doors: Set[Coord] = set()
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            nr, nc = r0 + dr, c0 + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                cell = (nr, nc)
                if has_line_of_sight(grid, center, cell, closed_doors=doors):
                    if cell in doors:
                        visible_doors.add(cell)
                    else:
                        visible_cells[cell] = grid[nr][nc]

    visible_targets: Dict[str, Coord] = {}
    if active_targets:
        for tid, tpos in active_targets.items():
            if tpos in visible_cells and visible_cells[tpos] == 0:
                visible_targets[tid] = tpos

    visible_agents: Dict[str, Coord] = {}
    if all_agent_positions:
        for aid, apos in all_agent_positions.items():
            if aid != agent_id and (apos in visible_cells or apos in visible_doors):
                visible_agents[aid] = apos

    visible_markers: Dict[Coord, GroundMarker] = {}
    if ground_markers:
        for mpos, marker in ground_markers.items():
            if mpos in visible_cells:
                visible_markers[mpos] = marker.copy()

    return VisibleObservation(
        agent_id=agent_id,
        center=center,
        visible_cells=visible_cells,
        visible_targets=visible_targets,
        visible_agents=visible_agents,
        visible_doors=visible_doors,
        visible_markers=visible_markers,
    )


def get_valid_neighbors(
    grid: Sequence[Sequence[int]],
    pos: Coord,
    closed_doors: Optional[Set[Coord]] = None,
) -> List[Coord]:
    """Returns 4-directional walkable (value == 0 and not a closed door 'D') neighbor coordinates."""
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    r, c = pos
    doors = closed_doors or set()
    neighbors: List[Coord] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if (
            0 <= nr < rows
            and 0 <= nc < cols
            and grid[nr][nc] == 0
            and (nr, nc) not in doors
        ):
            neighbors.append((nr, nc))
    return neighbors


def get_door_stand_positions(
    grid: Sequence[Sequence[int]],
    door_pos: Coord,
    closed_doors: Optional[Set[Coord]] = None,
) -> List[Coord]:
    """
    Returns walkable cells immediately adjacent to `door_pos` where agents can stand
    to cooperatively unlock/remove the door 'D'.
    """
    return get_valid_neighbors(grid, door_pos, closed_doors=closed_doors)


def can_unlock_door(
    door_pos: Coord,
    agent_positions: Dict[str, Coord],
) -> bool:
    """
    Checks if at least 2 agents are adjacent to (or standing on) `door_pos`
    at the same time, fulfilling the 2-agent requirement to remove Door 'D'.
    """
    dr, dc = door_pos
    adjacent_agents = [
        aid
        for aid, (ar, ac) in agent_positions.items()
        if abs(ar - dr) + abs(ac - dc) <= 1
    ]
    return len(adjacent_agents) >= 2


def bfs_shortest_path(
    grid: Sequence[Sequence[int]],
    start: Coord,
    goal: Coord,
    avoid_positions: Optional[Set[Coord]] = None,
    closed_doors: Optional[Set[Coord]] = None,
) -> List[Coord]:
    """
    Finds the shortest path on walkable `0` tiles from `start` to `goal`,
    respecting `closed_doors` ('D') and optional `avoid_positions`.
    If `goal` itself is a closed door 'D', routes to the closest walkable neighbor of `goal`.
    """
    doors = closed_doors or set()
    if goal in doors:
        # Route to a walkable cell adjacent to the door
        adj = get_door_stand_positions(grid, goal, closed_doors=doors)
        if not adj:
            return []
        if start in adj:
            return [start]
        best_path: List[Coord] = []
        for target_adj in adj:
            p = bfs_shortest_path(
                grid, start, target_adj, avoid_positions=avoid_positions, closed_doors=doors
            )
            if p and (not best_path or len(p) < len(best_path)):
                best_path = p
        return best_path

    if start == goal:
        return [start]

    def _run_bfs(blocked: Set[Coord]) -> List[Coord]:
        queue: deque[Coord] = deque([start])
        parent: Dict[Coord, Optional[Coord]] = {start: None}
        while queue:
            curr = queue.popleft()
            if curr == goal:
                path: List[Coord] = []
                node: Optional[Coord] = curr
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path

            for nxt in get_valid_neighbors(grid, curr, closed_doors=doors):
                if nxt not in parent and (nxt == goal or nxt not in blocked):
                    parent[nxt] = curr
                    queue.append(nxt)
        return []

    if avoid_positions:
        path = _run_bfs(avoid_positions)
        if path:
            return path
    return _run_bfs(set())


def bfs_distance_map(
    grid: Sequence[Sequence[int]],
    sources: Sequence[Coord],
    closed_doors: Optional[Set[Coord]] = None,
) -> Dict[Coord, int]:
    """
    Computes shortest topological step distance from any of `sources`
    to all currently reachable `0` cells in the floor plan.
    """
    dist: Dict[Coord, int] = {}
    queue: deque[Coord] = deque()
    doors = closed_doors or set()
    for s in sources:
        if 0 <= s[0] < len(grid) and 0 <= s[1] < len(grid[0]):
            if grid[s[0]][s[1]] == 0 and s not in doors and s not in dist:
                dist[s] = 0
                queue.append(s)

    while queue:
        curr = queue.popleft()
        d = dist[curr]
        for nxt in get_valid_neighbors(grid, curr, closed_doors=doors):
            if nxt not in dist:
                dist[nxt] = d + 1
                queue.append(nxt)
    return dist


def detect_dead_end_branch(
    grid: Sequence[Sequence[int]],
    current_pos: Coord,
    cleared_cells: Set[Coord],
    entrances: Sequence[Coord],
    closed_doors: Set[Coord],
    active_target_positions: Set[Coord],
) -> Optional[Tuple[Coord, Set[Coord]]]:
    """
    Checks if `current_pos` is at the mouth/exit of a dead-end corridor that has been
    100% searched by the agent (contains no entrances, no closed doors 'D', no targets,
    and only 1 exit point back to the rest of the maze).

    Returns `(marker_pos, dead_end_cells)` where an 'O' marker can be placed, or `None`.
    """
    neighbors = get_valid_neighbors(grid, current_pos, closed_doors=closed_doors)
    entrance_set = set(entrances)

    # Check if `current_pos` itself or one of its neighbors is the neck of a cleared dead-end pocket
    for candidate_neck in [current_pos] + neighbors:
        if candidate_neck in entrance_set:
            continue
        adj = get_valid_neighbors(grid, candidate_neck, closed_doors=closed_doors)
        # Try isolating each branch behind `candidate_neck`
        for start_cell in adj:
            # Flood-fill from `start_cell` without crossing `candidate_neck`
            visited: Set[Coord] = {start_cell}
            queue: deque[Coord] = deque([start_cell])
            is_dead_end = True
            has_unsearched_or_poi = False

            while queue:
                curr = queue.popleft()
                if (
                    curr in entrance_set
                    or curr in active_target_positions
                    or curr not in cleared_cells
                ):
                    has_unsearched_or_poi = True
                    break
                # Check if any closed door is adjacent to `curr`
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    if (curr[0] + dr, curr[1] + dc) in closed_doors:
                        has_unsearched_or_poi = True
                        break
                if has_unsearched_or_poi:
                    break

                for nxt in get_valid_neighbors(grid, curr, closed_doors=closed_doors):
                    if nxt == candidate_neck:
                        continue
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append(nxt)
                        if len(visited) > 25:
                            # Too large to be a local dead-end branch
                            is_dead_end = False
                            break
                if not is_dead_end:
                    break

            if is_dead_end and not has_unsearched_or_poi and len(visited) >= 1:
                visited.add(candidate_neck)
                return candidate_neck, visited

    return None


def partition_floor_plan_into_sectors(
    grid: Sequence[Sequence[int]],
    entrances: Sequence[Coord],
    num_sectors: int = 3,
) -> Dict[int, Set[Coord]]:
    """
    Partitions all walkable `0` cells of the known floor plan into
    `num_sectors` (default 3 for A1, A2, A3) balanced spatial regions.
    """
    dist_from_entrance = bfs_distance_map(grid, entrances)
    reachable = sorted(dist_from_entrance.keys())
    if not reachable:
        return {i: set() for i in range(1, num_sectors + 1)}

    seeds: List[Coord] = [
        max(reachable, key=lambda c: (c[0] * -1 + c[1], dist_from_entrance[c]))
    ]
    for _ in range(1, num_sectors):
        dmap = bfs_distance_map(grid, seeds)
        next_seed = max(reachable, key=lambda c: dmap.get(c, 0))
        seeds.append(next_seed)

    seeds.sort(key=lambda c: (c[1], c[0]))
    seed_dists: List[Dict[Coord, int]] = [bfs_distance_map(grid, [s]) for s in seeds]

    sectors: Dict[int, Set[Coord]] = {i: set() for i in range(1, num_sectors + 1)}
    target_size = (len(reachable) // num_sectors) + 1

    def _preference_margin(cell: Coord) -> int:
        dists = sorted(sd.get(cell, 999999) for sd in seed_dists)
        return dists[0] - dists[1] if len(dists) > 1 else 0

    for cell in sorted(reachable, key=_preference_margin):
        ranked_sectors = sorted(
            range(1, num_sectors + 1),
            key=lambda sid: (
                seed_dists[sid - 1].get(cell, 999999)
                + (25 if len(sectors[sid]) >= target_size else 0)
            ),
        )
        chosen = ranked_sectors[0]
        sectors[chosen].add(cell)

    return sectors
