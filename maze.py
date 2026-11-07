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
                maze[chosen[0]][chosen[1]] = '0'
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

    def seal(self): 
        new_maze = [['1' for i in range(self.n+2)] for j in range(self.n+2)] 

        for i in range(0, self.n): 
            for j in range(0, self.n):
                new_maze[i+1][j+1] = self.maze[i][j]

        self.maze = new_maze 
        self.n += 2

    def poke(self, top_count, bottom_count):
        top_wall = [(0, i) for i in range(1, self.n-1)] 
        bottom_wall = [(self.n-1, i) for i in range(1, self.n-1)]

        count = 0
        while count < 3:
            candidate = random.choice(top_wall) 
            if self.maze[candidate[0]+1][candidate[1]] == "0": 
                self.maze[candidate[0]][candidate[1]] = "A"+str(count+1)
                top_wall.remove(candidate)  
                count += 1  
            else:
                top_wall.remove(candidate)   

        for i in range(top_count): 
            candidate = random.choice(top_wall) 
            if self.maze[candidate[0]+1][candidate[1]] == "0": 
                self.maze[candidate[0]][candidate[1]] = "0"
                top_wall.remove(candidate) 

        for i in range(bottom_count): 
            candidate = random.choice(bottom_wall) 
            if self.maze[candidate[0]-1][candidate[1]] == "0": 
                self.maze[candidate[0]][candidate[1]] = "0"
                bottom_wall.remove(candidate) 


    def gen_wounded(self, count):
        candidates = [] 
        for x in range(0, self.n): 
            for y in range(0, self.n): 
                if self.maze[x][y] == '0': 
                    candidates.append((x,y))

        for i in range(count): 
            candidate = random.choice(candidates)
            self.maze[candidate[0]][candidate[1]] = 'C'
            candidates.remove(candidate)
