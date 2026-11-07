"""
Rescue agent for the maze rescue simulation.
Maze format: 2D list of ints, 0 = empty space, 1 = wall. Coordinates are (row, col).
"""
from __future__ import annotations

import re
from collections import deque
from enum import Enum

EMPTY, WALL = 0, 1
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class Link(Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"


class RescueAgent:
    def __init__(self, agent_id, maze, start, vision=2):
        self.id = agent_id
        self.maze = maze
        self.rows, self.cols = len(maze), len(maze[0])
        self.pos = tuple(start)
        self.vision = vision

        self.link = Link.CONNECTED
        self.instruction = {"action": "explore"}  # last order from HQ
        self.carrying = False

        self.known_exits = set()
        self.known_people = set()
        self.other_agents = {}  # agent_id -> last known (row, col)
        self.seen = set()       # open cells that have been within vision

        self.outbox = []        # signals waiting to be sent to HQ
        self.path = []
        self._mark_seen()

    # ---------- link state ----------
    def set_connection(self, connected: bool):
        self.link = Link.CONNECTED if connected else Link.DISCONNECTED

    @property
    def connected(self):
        return self.link is Link.CONNECTED

    # ---------- perception ----------
    def visible_cells(self):
        r0, c0 = self.pos
        v = self.vision
        return {(r, c) for r in range(r0 - v, r0 + v + 1)
                for c in range(c0 - v, c0 + v + 1) if self.is_open((r, c))}

    def observe(self, exits=(), people=()):
        """The simulator calls this with the exits and people currently in vision."""
        exits = {tuple(e) for e in exits}
        people = {tuple(p) for p in people}
        for e in exits - self.known_exits:
            self.known_exits.add(e)
            self._signal("exit_found", pos=e)
        for p in people - self.known_people:
            self.known_people.add(p)
            self._signal("person_found", pos=p)
        # People we remember but can no longer see were rescued by someone else.
        for p in (self.known_people & self.visible_cells()) - people:
            self.known_people.discard(p)

    # ---------- communication ----------
    def exchange(self):
        """Connected: return queued signals plus a status report. Disconnected: []."""
        if not self.connected:
            return []
        msgs = self.outbox + [{"type": "status", "agent": self.id, "pos": self.pos,
                               "carrying": self.carrying, "instruction": self.instruction}]
        self.outbox = []
        return msgs

    def receive(self, instruction=None, agent_locations=None):
        """Take an order and other agents' locations from HQ. Ignored while disconnected."""
        if not self.connected:
            return False
        if instruction is not None:
            new = self.parse_instruction(instruction)
            if new != self.instruction:
                self.instruction = new
                self.path = []
        if agent_locations:
            self.other_agents = {a: tuple(p) for a, p in agent_locations.items() if a != self.id}
        return True

    @staticmethod
    def parse_instruction(instr):
        """Dict {"action": ..., "target": [r, c]} or a vague string like "rescue the person at (4, 7)"."""
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

    # ---------- acting ----------
    def step(self):
        """Move one cell according to the current instruction. Returns the new position."""
        goal = self._choose_goal()
        if goal is not None and goal != self.pos:
            if not self.path or self.path[-1] != goal:
                self.path = self.bfs_path(self.pos, goal)
            if self.path:
                self.pos = self.path.pop(0)
        self._mark_seen()
        self._interact()
        return self.pos

    def _choose_goal(self):
        if self.carrying:
            return self._nearest(self.known_exits) or self._frontier_goal()
        action = self.instruction.get("action")
        target = self.instruction.get("target")
        # Order completed (arrived, nothing to pick up): act on our own.
        if action in ("rescue", "go_to") and target == self.pos:
            self.instruction = self._fallback()
            action, target = self.instruction["action"], self.instruction.get("target")
        if action == "hold":
            return None
        if action in ("rescue", "go_to") and target and self.is_open(target):
            return target
        if action == "return":
            return self._nearest(self.known_exits) or self._frontier_goal()
        return self._frontier_goal(region=target)

    def _fallback(self):
        person = self._nearest(self.known_people)
        return {"action": "rescue", "target": person} if person else {"action": "explore"}

    def _interact(self):
        if not self.carrying and self.pos in self.known_people:
            self.carrying = True
            self.known_people.discard(self.pos)
            self._signal("person_picked_up", pos=self.pos)
            self.instruction = {"action": "return"}
            self.path = []
        elif self.carrying and self.pos in self.known_exits:
            self.carrying = False
            self._signal("person_rescued", pos=self.pos)
            self.instruction = self._fallback()
            self.path = []

    # ---------- pathfinding ----------
    def is_open(self, cell):
        r, c = cell
        return 0 <= r < self.rows and 0 <= c < self.cols and self.maze[r][c] != WALL

    def _neighbours(self, cell):
        r, c = cell
        for dr, dc in MOVES:
            n = (r + dr, c + dc)
            if self.is_open(n):
                yield n

    def distances_from(self, start):
        dist = {start: 0}
        q = deque([start])
        while q:
            cur = q.popleft()
            for n in self._neighbours(cur):
                if n not in dist:
                    dist[n] = dist[cur] + 1
                    q.append(n)
        return dist

    def bfs_path(self, start, goal):
        """Shortest path from start to goal (start excluded). [] if unreachable."""
        prev = {start: None}
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

    def _nearest(self, cells):
        if not cells:
            return None
        dist = self.distances_from(self.pos)
        reachable = [c for c in cells if c in dist]
        return min(reachable, key=lambda c: dist[c]) if reachable else None

    def _frontier_goal(self, region=None):
        """Nearest unseen open cell, spread away from other agents (or toward a region)."""
        dist = self.distances_from(self.pos)
        unseen = [c for c in dist if c not in self.seen]
        if not unseen:
            return self._nearest(self.known_people)

        def score(c):
            s = dist[c]
            if region:
                s += abs(c[0] - region[0]) + abs(c[1] - region[1])
            for p in self.other_agents.values():
                d = abs(c[0] - p[0]) + abs(c[1] - p[1])
                s += max(0, 6 - d) * 2  # avoid exploring where teammates are
            return s

        return min(unseen, key=score)

    # ---------- helpers ----------
    def _mark_seen(self):
        self.seen |= self.visible_cells()

    def _signal(self, kind, **data):
        self.outbox.append({"type": kind, "agent": self.id, **data})