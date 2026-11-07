from maze import * 
from maze_vis import MazeGrid

myMaze = Maze(51)
myMaze.gen_wounded(10)
myMaze.seal()
myGrid = MazeGrid(schema=myMaze.maze, rows=myMaze.n, columns=myMaze.n)
myGrid.run()