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
COLOR_BG = (30, 30, 40)
COLOR_GRID = (60, 60, 70)
COLOR_WALL = (40, 40, 120)
COLOR_START = (0, 200, 0)
COLOR_GOAL = (200, 0, 0)
COLOR_PATH = (250, 220, 120)
COLOR_CLOSED = (120, 0, 120)
COLOR_TEXT = (230, 230, 230)
COLOR_PANEL = (25, 25, 35)
COLOR_BUTTON = (50, 50, 80)
COLOR_BUTTON_HOVER = (70, 70, 100)
COLOR_BUTTON_ACTIVE = (90, 120, 90)

WHITE = (240, 240, 240)  # text and highlights
GREEN = (90, 220, 120)   # frog color
FROG_RADIUS = 16          # draw size and collision size for the frog
FROG_SPEED  = 100.0       # top speed for the frog in pixels per second