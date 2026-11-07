"""
Floor plan parsing, centred 5x5 sight with wall-occlusion raycasting,
shortest-path navigation (BFS/A*), and 3-sector spatial partitioning.
"""

from __future__ import annotations
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .models import Coord, TileType, VisibleObservation


def parse_floor_plan(
    raw_grid: Sequence[Sequence[Any]],
) -> Tuple[List[List[int]], Dict[str, Coord], Dict[str, Coord]]:
    """
    Parses a 2D floor plan array where:
      - 0 or '0' is traversable path
      - 1 or '1' is wall
      - 'A1', 'A2', 'A3' are agent starting positions (also traversable path=0)
      - 'P', 'T', or 'P1', 'P2'... are optional target/person markers on the floor plan

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
                elif token.startswith("P") or token.startswith("T"):
                    tid = token if len(token) > 1 else f"P{target_counter}"
                    target_positions[tid] = (r, c)
                    target_counter += 1
                    parsed_row.append(TileType.PATH.value)
                else:
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
) -> List[List[str]]:
    """
    Exports current grid and entity locations into the 2D string schema expected
    by the teammate Tkinter visualizer (`maze_vis.py`):
      - '0': path
      - '1': wall
      - 'A' / 'A1' / 'A2' / 'A3': agent
      - 'C': coordinator / entrance checkpoint
      - 'P': target / survivor
    """
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    schema: List[List[str]] = [
        [str(grid[r][c]) for c in range(cols)] for r in range(rows)
    ]

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



def has_line_of_sight(grid: Sequence[Sequence[int]], start: Coord, end: Coord) -> bool:
    """
    Determines whether `end` is visible from `start` within the grid,
    accounting for wall occlusion (value == 1).

    Uses fine-grained ray sampling + corner pinch prevention:
    - Any intermediate cell strictly between `start` and `end` that is a wall (1)
      blocks vision to `end`.
    - Diagonal squeezing between two diagonally adjacent walls also blocks vision.
    """
    r0, c0 = start
    r1, c1 = end
    if (r0, c0) == (r1, c1):
        return True

    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    if not (0 <= r1 < rows and 0 <= c1 < cols):
        return False

    dr = r1 - r0
    dc = c1 - c0
    steps = max(abs(dr), abs(dc))

    # Immediate neighbors (distance 1 in Chebyshev metric)
    if steps == 1:
        # If moving diagonally, check if two orthogonal walls pinch the corner completely shut
        if abs(dr) == 1 and abs(dc) == 1:
            if grid[r0 + dr][c0] == 1 and grid[r0][c0 + dc] == 1:
                return False
        return True

    # For distance == 2 (outer ring of the 5x5 sight window):
    # Sample points along the segment from center of start to center of end.
    # N=20 samples ensures we catch every intermediate cell along the ray.
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

        # If intermediate cell is a wall, ray is blocked
        if grid[cr][cc] == 1:
            return False

        # Check diagonal pinch transition between prev_cell and curr_cell
        pr, pc = prev_cell
        if abs(cr - pr) == 1 and abs(cc - pc) == 1:
            if grid[cr][pc] == 1 and grid[pr][cc] == 1:
                return False

        prev_cell = curr_cell

    # Also check exact midpoint for knight's moves (e.g. (2, 1) or (1, 2))
    # For a (2, 1) offset, the ray passes through both (r0 + sgn(dr), c0) and (r0 + sgn(dr), c0 + sgn(dc)).
    # If BOTH of those intermediate cells are walls, it is certainly blocked;
    # if the direct orthogonal step (r0 + sgn(dr), c0) AND diagonal step (r0 + sgn(dr), c0 + sgn(dc))
    # block the corridor, block LOS.
    sgn_r = (1 if dr > 0 else -1) if dr != 0 else 0
    sgn_c = (1 if dc > 0 else -1) if dc != 0 else 0

    if abs(dr) == 2 and abs(dc) == 1:
        mid_ortho = (r0 + sgn_r, c0)
        mid_diag = (r0 + sgn_r, c0 + sgn_c)
        if grid[mid_ortho[0]][mid_ortho[1]] == 1 and grid[mid_diag[0]][mid_diag[1]] == 1:
            return False
    elif abs(dr) == 1 and abs(dc) == 2:
        mid_ortho = (r0, c0 + sgn_c)
        mid_diag = (r0 + sgn_r, c0 + sgn_c)
        if grid[mid_ortho[0]][mid_ortho[1]] == 1 and grid[mid_diag[0]][mid_diag[1]] == 1:
            return False

    return True


def compute_5x5_sight(
    grid: Sequence[Sequence[int]],
    center: Coord,
    agent_id: str = "A1",
    active_targets: Optional[Dict[str, Coord]] = None,
    all_agent_positions: Optional[Dict[str, Coord]] = None,
) -> VisibleObservation:
    """
    Computes the centred 5x5 sight around `center = (r, c)` (i.e. r-2..r+2, c-2..c+2)
    with wall occlusion.

    Returns a `VisibleObservation` containing:
      - visible_cells: dict of {(r, c): tile_value} for all non-occluded cells in 5x5
      - visible_targets: dict of {target_id: (r, c)} for targets standing on visible path cells
      - visible_agents: dict of {other_agent_id: (r, c)} for peers in visible path cells
    """
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    r0, c0 = center

    visible_cells: Dict[Coord, int] = {}
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            nr, nc = r0 + dr, c0 + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if has_line_of_sight(grid, center, (nr, nc)):
                    visible_cells[(nr, nc)] = grid[nr][nc]

    visible_targets: Dict[str, Coord] = {}
    if active_targets:
        for tid, tpos in active_targets.items():
            if tpos in visible_cells and visible_cells[tpos] == 0:
                visible_targets[tid] = tpos

    visible_agents: Dict[str, Coord] = {}
    if all_agent_positions:
        for aid, apos in all_agent_positions.items():
            if aid != agent_id and apos in visible_cells:
                visible_agents[aid] = apos

    return VisibleObservation(
        agent_id=agent_id,
        center=center,
        visible_cells=visible_cells,
        visible_targets=visible_targets,
        visible_agents=visible_agents,
    )


def get_valid_neighbors(grid: Sequence[Sequence[int]], pos: Coord) -> List[Coord]:
    """Returns 4-directional walkable (value == 0) neighbor coordinates."""
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    r, c = pos
    neighbors: List[Coord] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
            neighbors.append((nr, nc))
    return neighbors


def bfs_shortest_path(
    grid: Sequence[Sequence[int]],
    start: Coord,
    goal: Coord,
    avoid_positions: Optional[Set[Coord]] = None,
) -> List[Coord]:
    """
    Finds the shortest path on walkable `0` tiles from `start` to `goal`.
    Returns a list of coordinates `[start, step1, ..., goal]`, or `[]` if unreachable.
    If `avoid_positions` blocks all paths, falls back to ignoring `avoid_positions`.
    """
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

            for nxt in get_valid_neighbors(grid, curr):
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
) -> Dict[Coord, int]:
    """
    Computes shortest topological step distance from any of `sources`
    to all reachable `0` cells in the floor plan.
    """
    dist: Dict[Coord, int] = {}
    queue: deque[Coord] = deque()
    for s in sources:
        if grid[s[0]][s[1]] == 0 and s not in dist:
            dist[s] = 0
            queue.append(s)

    while queue:
        curr = queue.popleft()
        d = dist[curr]
        for nxt in get_valid_neighbors(grid, curr):
            if nxt not in dist:
                dist[nxt] = d + 1
                queue.append(nxt)
    return dist


def partition_floor_plan_into_sectors(
    grid: Sequence[Sequence[int]],
    entrances: Sequence[Coord],
    num_sectors: int = 3,
) -> Dict[int, Set[Coord]]:
    """
    Partitions all reachable walkable `0` cells of the known floor plan into
    `num_sectors` (default 3 for A1, A2, A3) balanced spatial regions.

    Algorithm:
    1. Run BFS from entrances to find all reachable `0` cells.
    2. Select 3 well-separated seed centroids using Farthest-First Traversal
       (K-Means++ style seeding) on the walkable graph so each agent gets a distinct wing/zone.
    3. Grow balanced Voronoi regions via round-robin multi-source BFS,
       with capacity balancing so sectors have near-equal cell counts.
    """
    dist_from_entrance = bfs_distance_map(grid, entrances)
    reachable = sorted(dist_from_entrance.keys())
    if not reachable:
        return {i: set() for i in range(1, num_sectors + 1)}

    # Step 1: Pick 3 spatially separated seed points
    # Seed 1: cell with median/high distance in one corner/wing
    seeds: List[Coord] = [max(reachable, key=lambda c: (c[0] * -1 + c[1], dist_from_entrance[c]))]
    for _ in range(1, num_sectors):
        dmap = bfs_distance_map(grid, seeds)
        next_seed = max(reachable, key=lambda c: dmap.get(c, 0))
        seeds.append(next_seed)

    # Sort seeds deterministically (top-left, center/bottom, right) so sector 1, 2, 3 are stable
    seeds.sort(key=lambda c: (c[1], c[0]))

    # Step 2: Compute distance from each seed to all reachable cells
    seed_dists: List[Dict[Coord, int]] = [bfs_distance_map(grid, [s]) for s in seeds]

    # Step 3: Assign each cell to its closest seed with soft load-balancing
    sectors: Dict[int, Set[Coord]] = {i: set() for i in range(1, num_sectors + 1)}
    target_size = (len(reachable) // num_sectors) + 1

    # Sort cells by how strongly they prefer a specific seed
    def _preference_margin(cell: Coord) -> int:
        dists = sorted(sd.get(cell, 999999) for sd in seed_dists)
        return dists[0] - dists[1] if len(dists) > 1 else 0

    for cell in sorted(reachable, key=_preference_margin):
        # Rank sectors by distance to this cell, penalized slightly if sector is already full
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
