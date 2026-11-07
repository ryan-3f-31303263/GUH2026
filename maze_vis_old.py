
import tkinter as tk


def create_grid(schma, rows=20, columns=20, bg_colour='#f0dfce',
                line_colour="black"):

    # Create and display window
    root = tk.Tk()
    root.title("Maze")
    root.geometry("1000x1000")
    root.resizable(True, True)

    # Create canvas
    canvas = tk.Canvas(
        root,
        bg=bg_colour,
        highlightthickness=0
    )
    canvas.pack(fill="both", expand=True)

    # Draw grid
    def draw_grid(event=None):

        canvas.delete("all")

        width = canvas.winfo_width()
        height = canvas.winfo_height()

        square_width = width / columns
        square_height = height / rows

        for row in range(rows):
            for col in range(columns):

                x1 = col * square_width
                y1 = row * square_height

                x2 = x1 + square_width
                y2 = y1 + square_height

                match (schma[row][col][0]):
                    case '0':
                        canvas.create_rectangle(
                            x1, y1, x2, y2,
                            outline=line_colour,
                            fill= bg_colour
                        )
                    case '1':
                        canvas.create_rectangle(
                            x1, y1, x2, y2,
                            outline=line_colour,
                            fill= "#918980"
                        )
                    case 'A':
                        padding = square_width * 0.3
                        
                        canvas.create_rectangle(
                            x1 + padding, y1 + padding,
                            x2 - padding, y2 - padding,
                            outline=line_colour,
                            fill="#ff00d4"
                        )
                    case 'C':
                        padding = square_width * 0.2

                        canvas.create_rectangle(
                            x1 + padding, y1 + padding,
                            x2 - padding, y2 - padding,
                            outline=line_colour,
                            fill="#76b07b"
                        )
                

    # Redraw when resized
    canvas.bind("<Configure>", draw_grid)

    # Start GUI
    root.mainloop()

    return canvas


create_grid([['0', '1', '0', 'A', 'C'],
['1', '0', 'C', '0', '1'],
['A', '1', '0', 'C', '0'],
['0', 'C', '1', '1', 'A'],
['C', '0', 'A', '0', '1']],5,5)