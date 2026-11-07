from maze import * 
from maze_vis import MazeGrid

myMaze = Maze(16)
myMaze.gen_wounded(10)
myMaze.seal()
myMaze.poke(3, 3)
myGrid = MazeGrid(schema=myMaze.maze, rows=myMaze.n, columns=myMaze.n)
myGrid.run()