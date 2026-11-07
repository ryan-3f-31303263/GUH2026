from enum import Enum 
import random

class GenerationMethod(Enum):
    DFS = 1
    ITER_STACK = 2
    KRUSKALS = 3

class Maze:
    def __init__(self, n, method=GenerationMethod.DFS):
        self.n = n
        if method == GenerationMethod.DFS:
            self.dfs_gen()
        elif method == GenerationMethod.ITER_STACK:
            self.iter_stack_gen()
        elif method == GenerationMethod.KRUSKALS: 
            self.kruskals_gen()

    def dfs_gen(self):
        maze = [['1' for i in range(self.n)] for j in range(self.n)] 
        visited = [] 
        stack = []

        # make initial cell
        start = random.randint(0, self.n-1)
        maze[0][start] = '0'

        # mark initial cell as visited and push to stack
        visited.append((0, start))
        stack.append((0, start))

        # main body
        while len(stack) > 0: 
            current_cell = stack.pop()
            x = current_cell[0]
            y = current_cell[1]

            neighbours = [(x+2, y), (x-2, y), (x, y+2), (x, y-2)]
            candidates = [] 
            for neighbour in neighbours: 
                nx = neighbour[0]
                ny = neighbour[1]

                if (0 <= nx < self.n) and (0 <= ny < self.n) and neighbour not in visited:
                    candidates.append(neighbour)

            if len(candidates) > 0: 
                chosen = candidates[random.randint(0, len(candidates)-1)]
                wall = (int((x + chosen[0])/2), int((y + chosen[1])/2))
                maze[wall[0]][wall[1]] = '0'
                visited.append(chosen)
                stack.append(current_cell) 
                stack.append(chosen)
            else: 
                pass
        self.maze = maze 
        return maze 








    def iter_stack_gen(self):
        pass

    def kruskals_gen(self):
        pass