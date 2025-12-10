"""
Complete A* pathfinding playground with control panel.

Controls:
  - Left click: toggle wall on or off on a cell.
  - Right click: set the start cell.
  - Middle click: set the goal cell.
  - Space key: run A* from start to goal and draw the path.
  - Escape or window close: quit the program.
"""

import sys
import heapq
import pygame
import math
from frog import Frog
from pygame.math import Vector2 as V2

from constants import (
    ROWS, COLS, CELL_SIZE,
    PANEL_WIDTH, PANEL_OPEN,
    GRID_WIDTH, GRID_HEIGHT, WINDOW_WIDTH, WINDOW_HEIGHT,
    COLOR_BG, COLOR_GRID, COLOR_WALL, COLOR_START, COLOR_GOAL,
    COLOR_PATH, COLOR_CLOSED, COLOR_TEXT, COLOR_PANEL,
    COLOR_BUTTON, COLOR_BUTTON_HOVER, COLOR_BUTTON_ACTIVE
)

# Set of wall cells stored as (row, col) pairs
walls = set()

# Initial start and goal cells
start = (5, 5)
goal = (10, 20)

# Latest path and closed set produced by A*
current_path = None
current_closed = set()
path_cost = 0.0
g_scores = {}
show_g_scores = True

# Movement settings
allow_diagonal = True


# --------------------------
# Helper functions for grid
# --------------------------

def cell_from_mouse(pos):
    """Convert mouse pixel position to grid cell coordinates."""
    x, y = pos
    c = x // CELL_SIZE
    r = y // CELL_SIZE
    if 0 <= r < ROWS and 0 <= c < COLS:
        return (r, c)
    return None

def draw_grid(surface):
    """Draw the grid cells with all overlays."""
    # First draw base cells
    for r in range(ROWS):
        for c in range(COLS):
            x = c * CELL_SIZE
            y = r * CELL_SIZE
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)

            color = COLOR_BG
            if (r, c) in walls:
                color = COLOR_WALL

            pygame.draw.rect(surface, color, rect)

    # Draw visited cells
    for (r, c) in current_closed:
        x = c * CELL_SIZE
        y = r * CELL_SIZE
        rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(surface, COLOR_CLOSED, rect)

    # Draw the path
    if current_path is not None:
        for (r, c) in current_path:
            x = c * CELL_SIZE
            y = r * CELL_SIZE
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, COLOR_PATH, rect)

    # Draw start and goal on top
    sr, sc = start
    gr, gc = goal
    start_rect = pygame.Rect(sc * CELL_SIZE, sr * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    goal_rect = pygame.Rect(gc * CELL_SIZE, gr * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(surface, COLOR_START, start_rect)
    pygame.draw.rect(surface, COLOR_GOAL, goal_rect)

    # Draw grid lines
    for c in range(COLS + 1):
        x = c * CELL_SIZE
        pygame.draw.line(surface, COLOR_GRID, (x, 0), (x, GRID_HEIGHT))
    for r in range(ROWS + 1):
        y = r * CELL_SIZE
        pygame.draw.line(surface, COLOR_GRID, (0, y), (GRID_WIDTH, y))

    if show_g_scores and g_scores:
        cost_font = pygame.font.SysFont("arial", 16)
        for (r, c), cost in g_scores.items():
            if (r, c) != start:  # Don't show cost on start
                x = c * CELL_SIZE
                y = r * CELL_SIZE
                cost_text = cost_font.render(f"{cost:.1f}", True, (255, 255, 255))
                text_rect = cost_text.get_rect(center=(x + CELL_SIZE // 2, y + CELL_SIZE // 2))
                # Draw shadow for better readability
                shadow = cost_font.render(f"{cost:.1f}", True, (0, 0, 0))
                shadow_rect = shadow.get_rect(center=(x + CELL_SIZE // 2 + 1, y + CELL_SIZE // 2 + 1))
                surface.blit(shadow, shadow_rect)
                surface.blit(cost_text, text_rect)


# --------------------------
# A* algorithm helpers
# --------------------------

def heuristic(a, b):
    """Diagonal distance heuristic for A*."""
    r1, c1 = a
    r2, c2 = b
    
    dr = abs(r1 - r2)
    dc = abs(c1 - c2)
    
    if allow_diagonal:
        diagonal = min(dr, dc)
        straight = max(dr, dc) - diagonal
        return diagonal * math.sqrt(2) + straight
    else:
        return dr + dc  # Manhattan distance

def get_neighbors(cell):
    """Return valid neighbor cells."""
    (r, c) = cell
    neighbors = []

    if allow_diagonal:
        # 8 directions
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]
    else:
        # 4 cardinal directions only
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for dr, dc in directions:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < ROWS and 0 <= nc < COLS:
            if (nr, nc) not in walls:
                neighbors.append((nr, nc))

    return neighbors

def reconstruct_path(came_from, current):
    """Rebuild the path from start to goal."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

def run_astar():
    """Execute A* algorithm."""
    global current_path, current_closed, path_cost, g_scores

    current_path = None
    current_closed = set()
    path_cost = 0.0
    g_scores = {}

    if start in walls or goal in walls:
        return

    open_heap = []
    heapq.heappush(open_heap, (0, start))

    open_set = {start}
    closed_set = set()

    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    came_from = {}

    while open_heap:
        current_f, current = heapq.heappop(open_heap)
        open_set.remove(current)

        if current == goal:
            current_path = reconstruct_path(came_from, current)
            current_closed = closed_set
            path_cost = g_score[goal]
            g_scores = g_score.copy()
            return

        closed_set.add(current)

        for neighbor in get_neighbors(current):
            if neighbor in closed_set:
                continue

            # Calculate step cost
            if neighbor[0] != current[0] and neighbor[1] != current[1]:
                step_cost = math.sqrt(2)  # diagonal
            else:
                step_cost = 1.0  # straight

            tentative_g = g_score[current] + step_cost

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)

                if neighbor not in open_set:
                    heapq.heappush(open_heap, (f_score[neighbor], neighbor))
                    open_set.add(neighbor)

    current_path = None
    current_closed = closed_set
    g_scores = g_score.copy()

# --------------------------
# Control Panel UI
# --------------------------

class Button:
    def __init__(self, x, y, w, h, text, action=None):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.action = action
        self.hovered = False
    
    def draw(self, surface, font, active=False):
        if active:
            color = COLOR_BUTTON_ACTIVE
        elif self.hovered:
            color = COLOR_BUTTON_HOVER
        else:
            color = COLOR_BUTTON
        
        pygame.draw.rect(surface, color, self.rect, border_radius=5)
        pygame.draw.rect(surface, COLOR_GRID, self.rect, 2, border_radius=5)
        
        text_surface = font.render(self.text, True, COLOR_TEXT)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)
    
    def check_hover(self, pos):
        self.hovered = self.rect.collidepoint(pos)
    
    def check_click(self, pos):
        if self.rect.collidepoint(pos):
            if self.action:
                self.action()
            return True
        return False

def toggle_diagonal():
    global allow_diagonal
    allow_diagonal = not allow_diagonal
    reset_search()

def toggle_panel():
    global PANEL_OPEN, WINDOW_WIDTH
    PANEL_OPEN = not PANEL_OPEN
    if PANEL_OPEN:
        WINDOW_WIDTH = GRID_WIDTH + PANEL_WIDTH
    else:
        WINDOW_WIDTH = GRID_WIDTH

def draw_control_panel(surface, font, small_font, buttons):
    """Draw the control panel on the right side."""
    if not PANEL_OPEN:
        return
    
    panel_x = GRID_WIDTH
    
    # Panel background
    panel_rect = pygame.Rect(panel_x, 0, PANEL_WIDTH, WINDOW_HEIGHT)
    pygame.draw.rect(surface, COLOR_PANEL, panel_rect)
    pygame.draw.line(surface, COLOR_GRID, (panel_x, 0), (panel_x, WINDOW_HEIGHT), 2)
    
    y_offset = 20

    # Controls section
    controls_title = font.render("Controls:", True, COLOR_TEXT)
    surface.blit(controls_title, (panel_x + 10, y_offset))
    y_offset += 30
    
    controls = [
        "LMB: Toggle wall",
        "RMB: Set start",
        "MMB: Set goal",
        "SPACE: Run A*",
        "C: Toggle G-scores",
        "R: Reset",
    ]

    
    for control in controls:
        text = small_font.render(control, True, COLOR_TEXT)
        surface.blit(text, (panel_x + 15, y_offset))
        y_offset += 22
    
    y_offset += 10

    # Statistics section
    stats_title = font.render("Statistics:", True, COLOR_TEXT)
    surface.blit(stats_title, (panel_x + 10, y_offset))
    y_offset += 30
    
    # Path info
    if current_path:
        path_len = small_font.render(f"Path length: {len(current_path)} cells", True, COLOR_TEXT)
        surface.blit(path_len, (panel_x + 15, y_offset))
        y_offset += 22
        
        cost_text = small_font.render(f"Total cost: {path_cost:.2f}", True, COLOR_TEXT)
        surface.blit(cost_text, (panel_x + 15, y_offset))
        y_offset += 22
        
        cells_explored = small_font.render(f"Cells explored: {len(current_closed)}", True, COLOR_TEXT)
        surface.blit(cells_explored, (panel_x + 15, y_offset))
        y_offset += 22
    else:
        no_path = small_font.render("No path found", True, (150, 150, 150))
        surface.blit(no_path, (panel_x + 15, y_offset))
        y_offset += 22
    
    y_offset += 20
    
    # Settings section
    settings_title = font.render("Settings:", True, COLOR_TEXT)
    surface.blit(settings_title, (panel_x + 10, y_offset))
    y_offset += 30
    
    movement_text = small_font.render(f"Movement:", True, COLOR_TEXT)
    surface.blit(movement_text, (panel_x + 15, y_offset))
    
    mode_text = "8-directional" if allow_diagonal else "4-directional"
    mode_render = small_font.render(f"({mode_text})", True, (150, 200, 150))
    surface.blit(mode_render, (panel_x + 90, y_offset))
    y_offset += 10
    
    # Draw buttons
    for button in buttons:
        button.draw(surface, small_font, active=(button.text == "Allow Diagonal Movement" and allow_diagonal))

def reset_search():
    """Reset path and closed set when map changes."""
    global current_path, current_closed, path_cost, g_scores
    current_path = None
    current_closed = set()
    path_cost = 0.0
    g_scores = {}


# --------------------------
# Main application loop
# --------------------------

def main():
    global start, goal, WINDOW_WIDTH, show_g_scores

    pygame.init()
    pygame.display.set_caption("A* Pathfinding with Control Panel")
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
    clock = pygame.time.Clock()

    frog = Frog((start[1] * CELL_SIZE + CELL_SIZE/2,
             start[0] * CELL_SIZE + CELL_SIZE/2))

    font = pygame.font.SysFont("arial", 16, bold=True)
    small_font = pygame.font.SysFont("arial", 14)
    
    # Create buttons
    panel_x = GRID_WIDTH
    buttons = [
        Button(panel_x + 20, 370, 210, 35, "Allow Diagonal Movement", toggle_diagonal),
    ]

    running = True
    while running:
        dt = clock.tick(90) / 1000.0

        mouse_pos = pygame.mouse.get_pos()
        
        # Update button hovers
        for button in buttons:
            button.check_hover(mouse_pos)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    run_astar()
                    if current_path:
                        frog.set_path(current_path)
                elif event.key == pygame.K_r:
                    walls.clear()
                    reset_search()
                    start = (5, 5)
                    goal = (10, 20)

                    frog.pos = V2(start[1] * CELL_SIZE + CELL_SIZE/2,
                                  start[0] * CELL_SIZE + CELL_SIZE/2)
                    frog.vel = V2()
                    frog.path = []
                    frog.path_index = 0
                    frog.target = frog.pos

                elif event.key == pygame.K_c:
                    show_g_scores = not show_g_scores


            if event.type == pygame.MOUSEBUTTONDOWN:
                # Check panel buttons
                if PANEL_OPEN:
                    clicked = False
                    for button in buttons:
                        if button.check_click(mouse_pos):
                            clicked = True
                            break
                    if clicked:
                        continue
                
                # Check grid clicks
                cell = cell_from_mouse(mouse_pos)
                if cell is not None:
                    if event.button == 1:  # Left click
                        if cell == start or cell == goal:
                            pass
                        elif cell in walls:
                            walls.remove(cell)
                        else:
                            walls.add(cell)
                        reset_search()
                    elif event.button == 3:  # Right click
                        if cell != goal:
                            start = cell
                            reset_search()
                    elif event.button == 2:  # Middle click
                        if cell != start:
                            goal = cell
                            reset_search()

        frog.update(dt)

        # Draw everything
        screen.fill(COLOR_BG)
        draw_grid(screen)
        draw_control_panel(screen, font, small_font, buttons)
        frog.draw(screen)

        pygame.display.flip()

    pygame.quit()
    sys.exit(0)

if __name__ == "__main__":
    main()