from maze_vis import MazeGrid

# Initial maze
maze = [
    ['0', '1', '0', 'A', 'C'],
    ['1', '0', 'C', '0', '1'],
    ['A', '1', '0', 'C', '0'],
    ['0', 'C', '1', '1', 'A'],
    ['C', '0', 'A', '0', '1']
]

# Create window once
gui = MazeGrid(maze, 5, 5)


# Test mazes
maze1 = [
    ['I', '0', '1', '0', 'A'],
    ['1', '0', '1', 'W', '1'],
    ['0', '0', '0', '1', '0'],
    ['W', '1', '0', 'O', '1'],
    ['0', '1', 'A', '0', '0']
]

maze2 = [
    ['A', '1', '0', 'W', '0'],
    ['0', '1', '0', '1', '0'],
    ['I', '0', 'W', '0', '1'],
    ['1', 'D', '1', '0', 'O'],
    ['0', '0', 'A', '1', '0']
]


# Update maze
def update_maze(new_maze):
    gui.update(new_maze)


# Animation
def animate(i=0):

    if i >= 10:
        return

    update_maze(maze1 if i % 2 == 0 else maze2)

    # Schedule next update in 2 seconds
    gui.root.after(2000, lambda: animate(i + 1))


# Start animation
gui.root.after(2000, animate)

# Run application
gui.run()