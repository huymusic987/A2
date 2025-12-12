"""
Assignment II - Connect 4 with MCTS and Statistics Panel

Enhanced version with real-time statistics display showing:
- Selected move analysis with win rate and visits
- All moves considered with UCB1 scores
- Column win rates split by player

Run using: python connect4_mcts.py
"""

import pygame
import sys
import math
import random

# ==========================================
#              GAME CONSTANTS AND COLORS
# ==========================================

ROWS = 6
COLS = 7

EMPTY = 0
PLAYER1 = 1
PLAYER2 = 2

SQUARESIZE = 100
RADIUS = SQUARESIZE // 2 - 5

BOARD_COLOR = (15, 20, 45)          # deep navy board
BG_COLOR = (10, 10, 10)             # dark background
PLAYER1_COLOR = (220, 70, 70)       # modern red
PLAYER2_COLOR = (70, 200, 170)      # modern teal
TEXT_COLOR = (240, 240, 240)        # off-white text
HINT_COLOR = (0, 230, 255)          # neon highlight
PANEL_COLOR = (18, 18, 22)          # matte side panel
HEADER_COLOR = (55, 90, 140)        # steel-blue headers

STATS_PANEL_WIDTH = 400
WIDTH = COLS * SQUARESIZE + STATS_PANEL_WIDTH
HEIGHT = (ROWS + 2) * SQUARESIZE
SIZE = (WIDTH, HEIGHT)

FPS = 60

# Global statistics tracking
column_stats = {col: {"player1_wins": 0, "player2_wins": 0} for col in range(COLS)}


# ====================================
#              CONNECT 4 STATE CLASS
# ====================================

class Connect4State:
    def __init__(self, board=None, current_player=PLAYER1):
        if board is None:
            self.board = [[EMPTY for _ in range(COLS)] for _ in range(ROWS)]
        else:
            self.board = [row[:] for row in board]
        self.current_player = current_player

    def clone(self):
        return Connect4State(self.board, self.current_player)

    def get_legal_moves(self):
        moves = []
        for c in range(COLS):
            if self.board[0][c] == EMPTY:
                moves.append(c)
        return moves

    def make_move(self, col):
        for r in range(ROWS - 1, -1, -1):
            if self.board[r][col] == EMPTY:
                self.board[r][col] = self.current_player
                self.current_player = PLAYER1 if self.current_player == PLAYER2 else PLAYER2
                return True
        return False

    def check_winner(self):
        # Horizontal
        for r in range(ROWS):
            for c in range(COLS - 3):
                piece = self.board[r][c]
                if piece != EMPTY:
                    if all(self.board[r][c + i] == piece for i in range(4)):
                        return piece

        # Vertical
        for c in range(COLS):
            for r in range(ROWS - 3):
                piece = self.board[r][c]
                if piece != EMPTY:
                    if all(self.board[r + i][c] == piece for i in range(4)):
                        return piece

        # Diagonal (top-left to bottom-right)
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                piece = self.board[r][c]
                if piece != EMPTY:
                    if all(self.board[r + i][c + i] == piece for i in range(4)):
                        return piece

        # Diagonal (bottom-left to top-right)
        for r in range(3, ROWS):
            for c in range(COLS - 3):
                piece = self.board[r][c]
                if piece != EMPTY:
                    if all(self.board[r - i][c + i] == piece for i in range(4)):
                        return piece

        return None

    def is_full(self):
        return all(self.board[0][c] != EMPTY for c in range(COLS))

    def is_terminal(self):
        return self.check_winner() is not None or self.is_full()


# ============================================================
#              IMPROVED MCTS IMPLEMENTATION
# ============================================================

class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.wins = 0.0
        self.untried_moves = state.get_legal_moves() if not state.is_terminal() else []
    
    def is_fully_expanded(self):
        return len(self.untried_moves) == 0
    
    def is_terminal(self):
        return self.state.is_terminal()
    
    def select_child_uct(self, exploration_constant=1.414):
        best_score = float('-inf')
        best_children = []
        
        for child in self.children:
            if child.visits == 0:
                return child
            
            exploitation = child.wins / child.visits
            exploration = exploration_constant * math.sqrt(
                math.log(self.visits) / child.visits
            )
            
            ucb_score = exploitation + exploration
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_children = [child]
            elif ucb_score == best_score:
                best_children.append(child)
        
        return random.choice(best_children)
    
    def expand(self):
        if not self.untried_moves:
            return None

        move = random.choice(self.untried_moves)
        self.untried_moves.remove(move)

        new_state = self.state.clone()
        new_state.make_move(move)

        child = MCTSNode(new_state, parent=self, move=move)
        self.children.append(child)
        return child
    
    def best_child(self):
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.visits)
    
    def update(self, result):
        self.visits += 1
        self.wins += result


def evaluate_terminal_state(state, player):
    winner = state.check_winner()
    
    if winner is None:
        return 0.5
    elif winner == player:
        return 1.0
    else:
        return 0.0


def simulate_random_playout(state, player):
    simulation_state = state.clone()
    
    while not simulation_state.is_terminal():
        legal_moves = simulation_state.get_legal_moves()
        
        if not legal_moves:
            break
        
        move = random.choice(legal_moves)
        simulation_state.make_move(move)
    
    return evaluate_terminal_state(simulation_state, player)


def mcts_search(root_state, iterations=1000, exploration_constant=1.414):
    if root_state.is_terminal():
        return None, {}
    
    root_player = root_state.current_player
    root_node = MCTSNode(root_state.clone())
    
    for _ in range(iterations):
        node = root_node
        state = root_state.clone()
        
        while node.is_fully_expanded() and not node.is_terminal():
            node = node.select_child_uct(exploration_constant)
            state.make_move(node.move)
        
        if not node.is_terminal() and not node.is_fully_expanded():
            node = node.expand()
            state = node.state.clone()
        
        result = simulate_random_playout(state, root_player)
        
        while node is not None:
            if node.parent is None:
                node.update(result)
            else:
                if node.parent.state.current_player == root_player:
                    node.update(result)
                else:
                    node.update(1.0 - result)
    
            node = node.parent
    
    best_child = root_node.best_child()
    
    # Collect detailed statistics for all children
    move_stats = []
    for child in root_node.children:
        if child.visits > 0:
            win_rate = (child.wins / child.visits) * 100
            exploitation = child.wins / child.visits
            exploration = exploration_constant * math.sqrt(
                math.log(root_node.visits) / child.visits
            )
            ucb_score = exploitation + exploration
            
            move_stats.append({
                'move': child.move,
                'win_rate': win_rate,
                'visits': child.visits,
                'ucb_score': ucb_score
            })
    
    # Sort by visits (descending)
    move_stats.sort(key=lambda x: x['visits'], reverse=True)
    
    stats = {
        'selected_move': best_child.move if best_child else None,
        'selected_win_rate': (best_child.wins / best_child.visits * 100) if best_child and best_child.visits > 0 else 0,
        'selected_visits': best_child.visits if best_child else 0,
        'total_simulations': root_node.visits,
        'all_moves': move_stats
    }
    
    if best_child is None:
        return None, stats
    
    return best_child.move, stats


# ============================================================
#             DRAWING THE GAME WITH PYGAME
# ============================================================

def draw_stats_panel(screen, font, small_font, tiny_font, stats, column_stats, selected_col):
    """Draw the statistics panel on the right side"""
    panel_x = COLS * SQUARESIZE

    # Draw panel background
    pygame.draw.rect(screen, PANEL_COLOR, (panel_x, 0, STATS_PANEL_WIDTH, HEIGHT))

    # Draw "STATS" title
    title_text = font.render("STATS", True, TEXT_COLOR)
    screen.blit(title_text, (panel_x + 10, 10))

    y_offset = 60

    # === AI MOVE ANALYSIS SECTION ===
    pygame.draw.rect(screen, HEADER_COLOR, (panel_x + 10, y_offset, STATS_PANEL_WIDTH - 20, 30))
    section1_text = small_font.render("AI Move Analysis", True, TEXT_COLOR)
    screen.blit(section1_text, (panel_x + 20, y_offset + 5))

    y_offset += 40

    # Display selected move info
    if stats and 'selected_move' in stats and stats['selected_move'] is not None:
        sel_move = stats['selected_move']
        sel_wr = stats.get('selected_win_rate', 0)
        sel_visits = stats.get('selected_visits', 0)
        total_sims = stats.get('total_simulations', 0)

        selected_text = tiny_font.render(f"SELECTED: Column {sel_move}] Win Rate: {sel_wr:.1f}% | Visits: {sel_visits}", True, HINT_COLOR)
        screen.blit(selected_text, (panel_x + 20, y_offset))
        y_offset += 20

        total_text = tiny_font.render(f"Total Simulations: {total_sims}", True, TEXT_COLOR)
        screen.blit(total_text, (panel_x + 20, y_offset))
        y_offset += 25

        # Display all moves considered
        all_moves_text = tiny_font.render("All Moves Considered:", True, TEXT_COLOR)
        screen.blit(all_moves_text, (panel_x + 20, y_offset))
        y_offset += 20

        for move_data in stats.get('all_moves', []):
            col = move_data['move']
            wr = move_data['win_rate']
            ucb = move_data['ucb_score']
            visits = move_data['visits']

            # Highlight if this is the selected move
            color = HINT_COLOR if col == sel_move else TEXT_COLOR

            move_text = tiny_font.render(f"* Col {col}: {wr:.1f}% | UCB1 {ucb:.3f} ({visits})", True, color)
            screen.blit(move_text, (panel_x + 25, y_offset))
            y_offset += 18
    else:
        no_data_text = small_font.render("No move data yet", True, TEXT_COLOR)
        screen.blit(no_data_text, (panel_x + 20, y_offset))
        y_offset += 30

    y_offset += 20

    # === COLUMN WIN RATES SECTION ===
    pygame.draw.rect(screen, HEADER_COLOR, (panel_x + 10, y_offset, STATS_PANEL_WIDTH - 20, 30))
    section2_text = small_font.render("Column Win Rates", True, TEXT_COLOR)
    screen.blit(section2_text, (panel_x + 20, y_offset + 5))

    y_offset += 40

    # Draw column win rates as split bars
    bar_width = STATS_PANEL_WIDTH - 140
    bar_height = 25

    # Build MCTS lookup if available: map col -> win_rate (0..100)
    mcts_by_col = {}
    if stats and 'all_moves' in stats:
        for move_data in stats['all_moves']:
            mcts_by_col[move_data['move']] = move_data['win_rate']

    for col in range(COLS):
        # Column label
        col_text = tiny_font.render(f"Col {col}:", True, TEXT_COLOR)
        screen.blit(col_text, (panel_x + 20, y_offset + 5))

        bar_x = panel_x + 70

        if mcts_by_col:
            # Use MCTS win rate (interpreted as "current player" win %)
            p1_pct = mcts_by_col.get(col, 0.0)
            p2_pct = 100.0 - p1_pct
            total_games_present = True
        else:
            # Fallback to historical column_stats (counts)
            p1_wins = column_stats[col]["player1_wins"]
            p2_wins = column_stats[col]["player2_wins"]
            total = p1_wins + p2_wins
            if total > 0:
                p1_pct = (p1_wins / total) * 100
                p2_pct = (p2_wins / total) * 100
                total_games_present = True
            else:
                total_games_present = False
                p1_pct = p2_pct = 0.0

        if total_games_present:
            # Draw split bar: left = p1_pct, right = p2_pct
            p1_width = int((p1_pct / 100.0) * bar_width)
            p2_width = bar_width - p1_width

            # Player 1 portion (red)
            if p1_width > 0:
                pygame.draw.rect(screen, PLAYER1_COLOR, (bar_x, y_offset, p1_width, bar_height))
            # Player 2 portion (yellow)
            if p2_width > 0:
                pygame.draw.rect(screen, PLAYER2_COLOR, (bar_x + p1_width, y_offset, p2_width, bar_height))

            # Border
            pygame.draw.rect(screen, TEXT_COLOR, (bar_x, y_offset, bar_width, bar_height), 1)

            # Highlight if this is the selected column
            if col == selected_col:
                pygame.draw.rect(screen, HINT_COLOR, (bar_x, y_offset, bar_width, bar_height), 2)

            # Percentages (rounded)
            pct_text = tiny_font.render(f"{p1_pct:.0f}% / {p2_pct:.0f}%", True, TEXT_COLOR)
            screen.blit(pct_text, (bar_x + bar_width + 10, y_offset + 5))
        else:
            # No data yet
            pygame.draw.rect(screen, (60, 60, 60), (bar_x, y_offset, bar_width, bar_height))
            pygame.draw.rect(screen, TEXT_COLOR, (bar_x, y_offset, bar_width, bar_height), 1)

            no_data = tiny_font.render("No data", True, (150, 150, 150))
            screen.blit(no_data, (bar_x + bar_width + 10, y_offset + 5))

        y_offset += 30


def draw_board(screen, state, font, small_font, tiny_font, message="", stats=None, selected_col=None):
    """Draw the game board and stats panel"""
    screen.fill(BG_COLOR)
    
    # Draw message
    text_surface = font.render(message, True, TEXT_COLOR)
    screen.blit(text_surface, (10, 5))
    
    # Draw board
    for c in range(COLS):
        for r in range(ROWS):
            pygame.draw.rect(
                screen,
                BOARD_COLOR,
                (c * SQUARESIZE, (r + 2) * SQUARESIZE, SQUARESIZE, SQUARESIZE)
            )
            
            pygame.draw.circle(
                screen,
                BG_COLOR,
                (c * SQUARESIZE + SQUARESIZE // 2, (r + 2) * SQUARESIZE + SQUARESIZE // 2),
                RADIUS
            )
    
    # Draw pieces
    for c in range(COLS):
        for r in range(ROWS):
            piece = state.board[r][c]
            
            if piece != EMPTY:
                color = PLAYER1_COLOR if piece == PLAYER1 else PLAYER2_COLOR
                
                pygame.draw.circle(
                    screen,
                    color,
                    (c * SQUARESIZE + SQUARESIZE // 2, (r + 2) * SQUARESIZE + SQUARESIZE // 2),
                    RADIUS
                )
    
    # Draw stats panel
    if stats is None:
        stats = {}
    draw_stats_panel(screen, font, small_font, tiny_font, stats, column_stats, selected_col)
    
    pygame.display.update()


def update_column_stats(col, winner):
    """Update statistics after a game ends"""
    if winner == PLAYER1:
        column_stats[col]["player1_wins"] += 1
    elif winner == PLAYER2:
        column_stats[col]["player2_wins"] += 1


def reset_column_stats():
    """Reset all column statistics"""
    global column_stats
    column_stats = {col: {"player1_wins": 0, "player2_wins": 0} for col in range(COLS)}


# ============================================================
#              MAIN GAME LOOP
# ============================================================

def main():
    pygame.init()
    
    screen = pygame.display.set_mode(SIZE)
    pygame.display.set_caption("MCTS Connect 4 with Stats")
    
    clock = pygame.time.Clock()
    
    font = pygame.font.SysFont("arial", 24)
    small_font = pygame.font.SysFont("arial", 18)
    tiny_font = pygame.font.SysFont("arial", 14)
    
    # Mode selection
    mode = None
    selecting = True
    
    while selecting:
        screen.fill(BG_COLOR)
        
        t1 = font.render("Press 1 for Player vs AI", True, TEXT_COLOR)
        t2 = font.render("Press 2 for AI vs AI", True, TEXT_COLOR)
        screen.blit(t1, (50, 100))
        screen.blit(t2, (50, 150))
        pygame.display.update()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    mode = 1
                    selecting = False
                elif event.key == pygame.K_2:
                    mode = 2
                    selecting = False
    
    state = Connect4State()
    game_over = False
    message = "Red turn"
    stats = {}
    last_move = None
    
    running = True
    while running:
        clock.tick(FPS)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    state = Connect4State()
                    game_over = False
                    message = "Red turn"
                    stats = {}
                    last_move = None
            
            if mode == 1 and not game_over and state.current_player == PLAYER1:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, _ = event.pos
                    col = x // SQUARESIZE
                    
                    if col < COLS and col in state.get_legal_moves():
                        last_move = col
                        state.make_move(col)
                        
                        winner = state.check_winner()
                        if winner:
                            message = "Red wins! Press R to restart" if winner == PLAYER1 else "Yellow wins! Press R to restart"
                            game_over = True
                            if last_move is not None:
                                update_column_stats(last_move, winner)
                        elif state.is_full():
                            message = "Draw! Press R to restart"
                            game_over = True
                        else:
                            message = "AI thinking..."
        
        if not game_over:
            if mode == 1:
                if state.current_player == PLAYER2:
                    ai_move, stats = mcts_search(state)
                    
                    if ai_move is not None:
                        last_move = ai_move
                        state.make_move(ai_move)
                        
                        winner = state.check_winner()
                        if winner:
                            message = "Red wins! Press R to restart" if winner == PLAYER1 else "Yellow wins! Press R to restart"
                            game_over = True
                            if last_move is not None:
                                update_column_stats(last_move, winner)
                        elif state.is_full():
                            message = "Draw! Press R to restart"
                            game_over = True
                        else:
                            message = "Red turn"
            
            elif mode == 2:
                ai_move, stats = mcts_search(state)
                
                if ai_move is not None:
                    last_move = ai_move
                    current_player = state.current_player
                    state.make_move(ai_move)
                    
                    winner = state.check_winner()
                    if winner:
                        message = "Red wins! Press R to restart" if winner == PLAYER1 else "Yellow wins! Press R to restart"
                        game_over = True
                        if last_move is not None:
                            update_column_stats(last_move, winner)
                    elif state.is_full():
                        message = "Draw! Press R to restart"
                        game_over = True
                    else:
                        message = "Red turn" if state.current_player == PLAYER1 else "Yellow turn"
                
                pygame.time.wait(500)
        
        draw_board(screen, state, font, small_font, tiny_font, message, stats, last_move)
    
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()