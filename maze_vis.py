
import tkinter as tk


class MazeGrid:

    def __init__(self, schema, rows=20, columns=20,
                 bg_colour='#f0dfce', line_colour="black"):

        self.schema = schema
        self.rows = rows
        self.columns = columns
        self.bg_colour = bg_colour
        self.line_colour = line_colour

        # Create window once
        self.root = tk.Tk()
        self.root.title("Maze")
        self.root.geometry("1000x1000")
        self.root.resizable(True, True)

        # Create canvas
        self.canvas = tk.Canvas(
            self.root,
            bg=bg_colour,
            highlightthickness=0
        )
        self.canvas.pack(fill="both", expand=True)

        # Redraw when resized
        self.canvas.bind("<Configure>", self.draw_grid)

        # Initial drawing
        self.root.update_idletasks()
        self.draw_grid()

    # Update maze data
    def update(self, schema):
        self.schema = schema
        self.draw_grid()

    # Draw grid
    def draw_grid(self, event=None):

        self.canvas.delete("all")

        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()

        square_width = width / self.columns
        square_height = height / self.rows

        for row in range(self.rows):
            for col in range(self.columns):

                x1 = col * square_width
                y1 = row * square_height

                x2 = x1 + square_width
                y2 = y1 + square_height

                match self.schema[row][col][0]:

                    case '0':
                        self.canvas.create_rectangle(
                            x1, y1, x2, y2,
                            outline=self.line_colour,
                            fill=self.bg_colour
                        )

                    case '1':
                        self.canvas.create_rectangle(
                            x1, y1, x2, y2,
                            outline=self.line_colour,
                            fill="#918980"
                        )

                    case 'A':
                        padding_x = square_width * 0.3
                        padding_y = square_height * 0.3

                        self.canvas.create_rectangle(
                            x1 + padding_x, y1 + padding_y,
                            x2 - padding_x, y2 - padding_y,
                            outline=self.line_colour,
                            fill="#ff00d4"
                        )

                    case 'W':
                        padding_x = square_width * 0.2
                        padding_y = square_height * 0.2

                        self.canvas.create_rectangle(
                            x1 + padding_x, y1 + padding_y,
                            x2 - padding_x, y2 - padding_y,
                            outline=self.line_colour,
                            fill="#76b07b"
                        )
                    case 'I':
                        padding_x = square_width * 0.4
                        padding_y = square_height * 0.4

                        self.canvas.create_rectangle(
                            x1 + padding_x, y1 + padding_y,
                            x2 - padding_x, y2 - padding_y,
                            outline=self.line_colour,
                            fill="blue"
                        )
                    case 'O':
                        padding_x = square_width * 0.4
                        padding_y = square_height * 0.4

                        self.canvas.create_rectangle(
                            x1 + padding_x, y1 + padding_y,
                            x2 - padding_x, y2 - padding_y,
                            outline=self.line_colour,
                            fill="red"
                        )
                    case 'D':
                        padding_x = square_width 
                        padding_y = square_height 

                        self.canvas.create_rectangle(
                            x1 + padding_x, y1 + padding_y,
                            x2 - padding_x, y2 - padding_y,
                            outline=self.line_colour,
                            fill="#c2c1c0"
                        )

    # Run GUI
    def run(self):
        self.root.mainloop()

    # Close GUI
    def close(self):
        self.root.destroy()
