# --------------------------
# Grid and window settings
# --------------------------

ROWS = 20
COLS = 30
CELL_SIZE = 40

# Control panel settings
PANEL_WIDTH = 250
PANEL_OPEN = True

# Compute window size from grid size
GRID_WIDTH = COLS * CELL_SIZE
GRID_HEIGHT = ROWS * CELL_SIZE
WINDOW_WIDTH = GRID_WIDTH + PANEL_WIDTH
WINDOW_HEIGHT = GRID_HEIGHT

# Colors in RGB format
COLOR_BG = (15, 18, 25)                 # dark grid background
COLOR_GRID = (50, 55, 75)              # subtle grid lines

COLOR_WALL = (45, 50, 70)              # wall cells
COLOR_START = (70, 200, 170)           # teal start node
COLOR_GOAL = (220, 70, 70)             # red goal node

COLOR_CLOSED = (60, 60, 90)            # explored cells
COLOR_PATH = (0, 230, 255)             # neon cyan path

COLOR_TEXT = (240, 240, 240)           # main text (panel)
COLOR_PANEL = (12, 12, 16)             # matte black control panel

COLOR_BUTTON = (30, 35, 50)            # button idle
COLOR_BUTTON_HOVER = (50, 60, 90)      # button hover
COLOR_BUTTON_ACTIVE = (0, 200, 220)    # active button (cyan)

WHITE = (240, 240, 240)  # text and highlights
GREEN = (90, 220, 120)   # frog color
FROG_RADIUS = 16          # draw size and collision size for the frog
FROG_SPEED  = 100.0       # top speed for the frog in pixels per second